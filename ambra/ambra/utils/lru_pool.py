from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)


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
    def __init__(self, objs: List[O], num_frames: int, max_prefetch: int = 0):
        self.lru: OrderedDict[O, Optional[K]] = OrderedDict()
        for b in objs:
            self.lru[b] = None
        self.lookup: OrderedDict[K, _Entry[O]] = OrderedDict()
        self.in_flight: List[Optional[Tuple[K, _Entry[O]]]] = [None] * num_frames

        self.max_prefetch: int = max_prefetch
        self.prefetch_store: Dict[O, K] = {}

    def get(
        self,
        key: K,
        load: Callable[[K, O], None],
        ensure_fetched: Optional[Callable[[K, O], None]] = None,
    ) -> O:
        cached = self.lookup.get(key)

        # Check if already loaded
        if cached is None:
            # Grab a free buffer
            obj, old_key = self.lru.popitem(last=False)
            if old_key is not None:
                # Remove the bufer from the lookup
                self.lookup.pop(old_key)

            # Load
            load(key, obj)

            # Register buffer as loaded for future use
            self.lookup[key] = _Entry(obj, _RefCount(), prefetching=False)
        else:
            # If this was a prefetched buffer wait for it to be loaded
            obj = cached.obj
            if cached.prefetching:
                # Realize prefetch
                if ensure_fetched:
                    ensure_fetched(key, obj)

                # Item is still in the prefetching list here.
                # It will be removed by the next prefetch cleanup
                cached.prefetching = False

            # If the buffer is in the LRU remove it to mark it as in use
            try:
                self.lru.pop(obj)
            except KeyError:
                pass
        return obj

    def is_available(self, key: K) -> bool:
        cached = self.lookup.get(key)
        return cached is not None and not cached.prefetching

    def is_available_or_prefetching(self, key: K) -> bool:
        return key in self.lookup

    def use_frame(self, frame_index: int, key: K) -> None:
        entry = self.lookup[key]
        entry.refcount.inc()
        self.in_flight[frame_index] = (key, entry)

    def use_manual(self, key: K) -> None:
        entry = self.lookup[key]
        entry.refcount.inc()

    def release_manual(self, key: K) -> None:
        entry = self.lookup[key]

        # Decrement refcount
        if entry.refcount.dec():
            # If refcount is 0 add buffer back to LRU
            self.lru[entry.obj] = key

    def release_frame(self, frame_index: int) -> None:
        if old := self.in_flight[frame_index]:
            key, entry = old

            # Decrement refcount
            if entry.refcount.dec():
                # If refcount is 0 add buffer back to LRU
                self.lru[entry.obj] = key

            # Mark nothing in flight yet for this frame.
            self.in_flight[frame_index] = None

    def give_back(self, k: K, obj: O) -> None:
        self.lru[obj] = k

    def prefetch(
        self,
        useful_range: Sequence[K],
        cleanup: Callable[[K, O], bool],
        submit_load: Callable[[K, O], None],
    ) -> None:
        if self.max_prefetch <= 0:
            return

        for obj, key in list(self.prefetch_store.items()):
            if cleanup(key, obj):
                # Insert back in the LRU if not yet claimed
                if self.lookup[key].prefetching:
                    self.lru[obj] = key

                    # Check if the buffer is still in the window
                    if key not in useful_range:
                        # Move to end of LRU to ensure this buffer is reused soon
                        self.lru.move_to_end(obj, last=False)

                    # Mark as ready
                    self.lookup[key].prefetching = False

                # Remove from prefetching dict
                del self.prefetch_store[obj]

            # TODO: we could also cancel prefetch work here, if possible

        bump = []
        prefetch_count = self.max_prefetch - len(self.prefetch_store)
        for key in useful_range[:prefetch_count]:
            cached = self.lookup.get(key)
            if cached is None:
                # Grab a free buffer
                obj, old_key = self.lru.popitem(last=False)
                if old_key is not None:
                    # Remove the bufer from the lookup
                    self.lookup.pop(old_key)

                # Submit for load
                submit_load(key, obj)

                # Register buffer as loading for future use
                self.lookup[key] = _Entry(obj, _RefCount(), prefetching=True)
                self.prefetch_store[obj] = key
            else:
                # If already loaded just bump in front of LRU
                bump.append(cached.obj)
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
