# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar


class _RefCount:
    def __init__(self, initial_count: int = 0):
        self.count = initial_count

    def inc(self) -> None:
        self.count += 1

    def dec(self) -> bool:
        self.count -= 1
        return self.count == 0

    def __repr__(self) -> str:
        return f"RC({self.count})"


K = TypeVar("K")
O = TypeVar("O")


@dataclass
class _Entry(Generic[O]):
    obj: O
    refcount: _RefCount
    prefetching: bool


class LRUPool(Generic[K, O]):
    def __init__(
        self,
        objs: List[O],
        num_frames: int,
        max_prefetch: int = 0,
    ):
        # (Key, Generation) -> Entry: holds all entries in use
        self.lookup: OrderedDict[Tuple[K, int], _Entry[O]] = OrderedDict()

        # Object -> (Key, Generation): holds all free objects. If the object
        # is not yet associated with any key (if it was never used before)
        # the value for that item will be None.
        self.lru: OrderedDict[O, Optional[Tuple[K, int]]] = OrderedDict()

        # ((Key, Generation), Entry): list of entries in flight with their key and generation
        self.in_flight: List[Optional[Tuple[Tuple[K, int], _Entry[O]]]] = [None] * num_frames

        # Key -> Generation: current generation of in use keys. If a key is not
        # in use, it will not be in this map and its generation will be implicitly 0.
        #
        # The generation index is increased when requested by the user.
        # It will also be reset when there is no entry associated with that
        # key in the lookup anymore (this implies no reference also in the lru,
        # in_flight list, and prefetch_store).
        self.current_generation: Dict[K, int] = {}

        # Maximum number of prefetched items at a time
        self.max_prefetch: int = max_prefetch

        # Object -> (Key, Generation): stores state for each prefetching item
        self.prefetch_store: Dict[O, Tuple[K, int]] = {}

        # Initialize LRU to empty objects
        for o in objs:
            self.lru[o] = None

    def get(
        self,
        k: K,
        load: Callable[[K, O], None],
        ensure_fetched: Optional[Callable[[K, O], None]] = None,
    ) -> O:
        key = (k, self.current_generation.get(k, 0))
        cached = self.lookup.get(key)

        # Check if already loaded
        if cached is None:
            # Grab a free object from the lru
            obj = self._lru_pop_free()

            # Load
            load(key[0], obj)

            # Register buffer as loaded for future use
            self.lookup[key] = _Entry(obj, _RefCount(), prefetching=False)
        else:
            # If this was a prefetched buffer wait for it to be loaded
            obj = cached.obj
            if cached.prefetching:
                # Realize prefetch
                if ensure_fetched:
                    ensure_fetched(key[0], obj)

                # Item is still in the prefetching list here.
                # It will be removed by the next prefetch cleanup
                cached.prefetching = False

            # If the buffer is in the LRU remove it to mark it as in use
            try:
                self.lru.pop(obj)
            except KeyError:
                pass
        return obj

    def is_available(self, k: K) -> bool:
        cached = self.lookup.get((k, self.current_generation.get(k, 0)))
        return cached is not None and not cached.prefetching

    def is_available_or_prefetching(self, k: K) -> bool:
        cached = self.lookup.get((k, self.current_generation.get(k, 0)))
        return cached is not None

    def increment_generation(self, k: K) -> None:
        gen = self.current_generation.get(k, 0)
        if (o := self.lookup.get((k, gen))) is not None:
            self.current_generation[k] = gen + 1
            try:
                self.lru.move_to_end(o.obj, last=False)
            except KeyError:
                pass

    def increment_all_generations(self) -> None:
        for k, o in self.lookup.items():
            current_gen = self.current_generation.get(k[0], 0)
            if current_gen == k[1]:
                self.current_generation[k[0]] = k[1] + 1
            try:
                self.lru.move_to_end(o.obj, last=False)
            except KeyError:
                pass

    def use_frame(self, frame_index: int, k: K) -> None:
        key = (k, self.current_generation.get(k, 0))

        entry = self.lookup[key]
        entry.refcount.inc()
        self.in_flight[frame_index] = (key, entry)

    def use_manual(self, k: K) -> None:
        key = (k, self.current_generation.get(k, 0))

        entry = self.lookup[key]
        entry.refcount.inc()

    def release_manual(self, k: K) -> None:
        key = (k, self.current_generation.get(k, 0))

        entry = self.lookup[key]

        # Decrement refcount
        if entry.refcount.dec():
            # If refcount is 0 add buffer back to LRU
            self.lru[entry.obj] = key

            # If this is an old generation of this buffer move it
            # to the end of the LRU to make sure it will be evicted soon.
            if key[1] < self.current_generation.get(key[0], 0):
                self.lru.move_to_end(entry.obj, last=False)

    def release_frame(self, frame_index: int) -> None:
        if old := self.in_flight[frame_index]:
            key, entry = old

            # Decrement refcount
            if entry.refcount.dec():
                # If refcount is 0 add buffer back to LRU
                self.lru[entry.obj] = key

                # If this is an old generation of this buffer move it
                # to the end of the LRU to make sure it will be evicted soon.
                if key[1] < self.current_generation.get(key[0], 0):
                    self.lru.move_to_end(entry.obj, last=False)

            # Mark nothing in flight yet for this frame.
            self.in_flight[frame_index] = None

    def give_back(self, k: K, obj: O) -> None:
        self.lru[obj] = (k, self.current_generation.get(k, 0))

    def prefetch(
        self,
        useful_range: Sequence[K],
        cleanup: Callable[[K, O], bool],
        submit_load: Callable[[K, O], None],
    ) -> None:
        if self.max_prefetch <= 0:
            return

        for obj, key in list(self.prefetch_store.items()):
            if cleanup(key[0], obj):
                # Insert back in the LRU if not yet claimed
                if self.lookup[key].prefetching:
                    self.lru[obj] = key

                    # Check if the buffer is old or it's not in the window
                    if key[1] < self.current_generation.get(key[0], 0) or key[0] not in useful_range:
                        # Move to end of LRU to ensure this buffer is reused soon
                        self.lru.move_to_end(obj, last=False)

                    # Mark as ready
                    self.lookup[key].prefetching = False

                # Remove from prefetching dict
                del self.prefetch_store[obj]

            # TODO: we could also cancel prefetch work here, if possible

        bump = []
        prefetch_count = self.max_prefetch - len(self.prefetch_store)

        # Submit prefetch requests and allocate buffers in ascending distance.
        for k in useful_range[:prefetch_count]:
            key = (k, self.current_generation.get(k, 0))
            cached = self.lookup.get(key)
            if cached is None:
                # Grab a free object from the lru
                obj = self._lru_pop_free()

                # Submit for load
                submit_load(key[0], obj)

                # Register buffer as loading for future use
                self.lookup[key] = _Entry(obj, _RefCount(), prefetching=True)
                self.prefetch_store[obj] = key
            else:
                # If already loaded just bump in front of LRU
                bump.append(cached.obj)

        # Bump to front in reverse order to ensure the closest element
        # will be evicted as late as possible.
        for o in reversed(bump):
            try:
                # Refresh entry in LRU cache
                self.lru.move_to_end(o)
            except KeyError:  # noqa: PERF203
                pass

    def clear(self) -> None:
        self.lru.clear()
        self.lookup.clear()
        self.in_flight.clear()
        self.prefetch_store.clear()
        self.current_generation.clear()

    def _lru_pop_free(self) -> O:
        obj, old_key = self.lru.popitem(last=False)
        if old_key is not None:
            # Remove the bufer from the lookup
            self.lookup.pop(old_key)

            # If we are removing the buffer with the current generation,
            # reset its generation number to 0 by removing it.
            if self.current_generation.get(old_key[0]) == old_key[1]:
                del self.current_generation[old_key[0]]
        return obj
