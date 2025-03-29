import watchfiles
from queue import Queue
from typing import List, Dict, Callable
from threading import Thread
from pathlib import Path
import atexit
from platformdirs import user_cache_path
import hashlib
import pickle
import abc

from pyxpg import slang

def compile(file: Path, entry: str):
    # TODO:
    # [ ] This cache does not currently consider
    #   [x] imported files / modules  -> highest importance
    #       [x] Export list of deps in prog
    #       [x] Use this in pipeline cache creation for deps
    #   [ ] compiler version          -> should expose this as an API
    #   [-] slang target              -> only spirv currently supported
    #   [-] preprocessor defines      -> not supported currently
    #   [-] params (if any)           -> not supported currently
    # [ ] No obvious way to clear the cache, maybe should have some LRU with max size? e.g. touch files when using them
    name = f"{hashlib.sha256(file.read_bytes()).digest().hex()}_{entry}.shdr"

    # Check cache
    cache_dir = user_cache_path("pyxpg")
    path = Path(cache_dir, name)
    if path.exists():
        try:
            pkl = pickle.load(open(path, "rb"))
            prog: slang.Shader = pkl[0]
            old_hashes: List[str] = pkl[1]

            # Check if any of the dependent files changed from when the shader
            # was serialized.
            assert len(prog.dependencies) == len(old_hashes)
            new_hashes = [ hashlib.sha256(Path(d).read_bytes()).digest().hex() for d in sorted(prog.dependencies) ]
            if new_hashes == old_hashes:
                print(f"Cache hit: {file}")
                return prog
        except Exception as e:
            print(f"Shader cache hit invalid: {e}")
            pass

    # Create prog
    print(f"Shader cache miss: {file}")
    prog = slang.compile(str(file), entry)

    hashes = [ hashlib.sha256(Path(d).read_bytes()).digest().hex() for d in sorted(prog.dependencies) ]

    # Populate cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump((prog, hashes), open(path, "wb"))
    return prog

class Pipeline:
    def __init__(self):
        items = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        self.__shaders: Dict[str, Path] = { a: Path(getattr(self, a)) for a in items }
        self.__compiled_shaders: Dict[str, slang.Shader] = {}
        self._update()
    
    @abc.abstractmethod
    def create(self, **kwargs):
        pass

    def _mark_dirty(self) -> bool:
        if self.__dirty:
            return False

        self.__dirty = True
        return True
    
    def _update(self):
        compiled_shaders: Dict[str, slang.Shader] = {}
        for k, v in self.__shaders.items():
            try:
                prog = compile(v, "main")
                compiled_shaders[k] = prog
            except slang.CompilationError as e:
                if k in self.__compiled_shaders:
                    # We still have the old version of this shader. Report the error and use that instead.

                    # TODO: maybe should bubble up all compilation here instead of printing here.
                    print(f'Shader compilation eror: {v} | Entry point: "main"')
                    print(e)

                    compiled_shaders[k] = self.__compiled_shaders[k]
                else:
                    raise e

        self.__compiled_shaders = compiled_shaders
        self._deps = []
        for s in self.__compiled_shaders.values():
            for d in s.dependencies:
                self._deps.append(d)
        self.create(**self.__compiled_shaders)
        self.__dirty = False
    
ALIVE_CACHES = []

class PipelineWatch:
    def __init__(self, pipelines: List[Pipeline]):
        self.queue = Queue()
        self.reload: Dict[Path, List[Pipeline]] = {}
        for pipe in pipelines:
            for dep in pipe._deps:
                self.reload.setdefault(Path(dep).absolute(), []).append(pipe)
        self.should_stop = False
        self.thread = Thread(target=self.__thread_entry, daemon=True, name="pipeline-watch")
        self.thread.start()

    def __thread_entry(self):
        ALIVE_CACHES.append(self)

        if not len(self.reload):
            return

        for changes in watchfiles.watch(*self.reload.keys(), debounce=200):
            if self.should_stop:
                break

            for change, path in changes:
                if change == watchfiles.Change.modified:
                    for pipe in self.reload[Path(path).absolute()]:
                        if pipe._mark_dirty():
                            self.queue.put(pipe)

    def refresh(self, before_any_update: Callable):
        if len(self.reload) and not self.thread.is_alive():
            raise Exception("Pipeline thread not running")

        called = False
        while not self.queue.empty():
            if not called:
                before_any_update()
                called = True
            pipe: Pipeline = self.queue.get()
            pipe._update()

    def stop(self):
        if len(self.reload):
            self.should_stop = True
            for p in self.reload.keys():
                p.touch()
                break
        self.thread.join()

        ALIVE_CACHES.remove(self)

def __close_all():
    while ALIVE_CACHES:
       ALIVE_CACHES[0].stop()

atexit.register(__close_all)