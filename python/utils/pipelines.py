import watchfiles
from queue import Queue
from typing import List, Dict, Callable, Optional
from threading import Thread, Event
from pathlib import Path
import atexit
from platformdirs import user_cache_path
import hashlib
import pickle
import shutil
from typing import Tuple, Union

from pyxpg import slang
from pyxpg import Window

CACHE_DIR = user_cache_path("pyxpg")

CACHE_VERSION_MAJOR = 0
CACHE_VERSION_MINOR = 1
CACHE_VERSION_PATCH = 1
CACHE_VERSION = f"{CACHE_VERSION_MAJOR}.{CACHE_VERSION_MINOR}.{CACHE_VERSION_PATCH}"

def clear_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)

def vulkan_version_to_minimum_supported_spirv_version(version: Tuple[int, int]) -> str:
    if version[0] == 0:
        raise ValueError("Unsupported Vulkan version < 1")
    if version[0] == 1:
        if version[1] == 0:
            return "spirv_1_0"
        elif version[1] == 1:
            return "spirv_1_3"
        elif version[1] == 2:
            return "spirv_1_5"
        elif version[1] == 3 or version[1] == 4:
            return "spirv_1_6"
    return "spirv_1_6"

def compile(file: Path, entry: str = "main", target: str = "spirv_1_3") -> slang.Shader:
    # [x] This cache does not currently consider
    #   [x] imported files / modules  -> highest importance
    #       [x] Export list of deps in prog
    #       [x] Use this in pipeline cache creation for deps
    #   [x] compiler version          -> should expose this as an API
    #   [x] slang target              -> only spirv currently supported
    #   Fill need if ever supported:
    #   - preprocessor defines
    #   - specialization constants
    #   - compilation options
    # [ ] No automatic way to clear the cache, maybe should have some LRU with max size? e.g. touch files when using them
    name = f"{hashlib.sha256(file.read_bytes()).digest().hex()}_{entry}_{target}_{CACHE_VERSION}.shdr"

    # Check cache
    path = Path(CACHE_DIR, name)
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
    prog = slang.compile(str(file), entry, target)
    hashes = [ hashlib.sha256(Path(d).read_bytes()).digest().hex() for d in sorted(prog.dependencies) ]

    # Populate cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump((prog, hashes), open(path, "wb"))
    return prog

class Pipeline:
    def __init__(self):
        items = [a for a in dir(type(self)) if not a.startswith('__') and not callable(getattr(type(self), a))]
        self.__shaders: Dict[str, Union[Path, Tuple[Path, str]]] = { a: getattr(type(self), a) for a in items }
        self.__compiled_shaders: Dict[str, slang.Shader] = {}
        self._update(True)
    
    def init(self, **kwargs):
        pass

    def create(self, **kwargs):
        pass

    def _mark_dirty(self) -> bool:
        if self.__dirty:
            return False

        self.__dirty = True
        return True
    
    def _update(self, init):
        compiled_shaders: Dict[str, slang.Shader] = {}
        for k, v in self.__shaders.items():
            try:
                if isinstance(v, list) or isinstance(v, tuple):
                    file, entry = v
                else:
                    file, entry = v, "main"
                prog = compile(Path(file), entry)
                compiled_shaders[k] = prog
            except slang.CompilationError as e:
                if k in self.__compiled_shaders:
                    # We still have the old version of this shader. Report the error and use that instead.

                    # TODO: maybe should bubble up all compilation errors here instead of printing here.
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
        if init:
            self.init(**self.__compiled_shaders)
        self.create(**self.__compiled_shaders)
        self.__dirty = False
    
ALIVE_CACHES = []

class PipelineWatch:
    def __init__(self, pipelines: List[Pipeline], window: Optional[Window]=None):
        self.window = window
        self.queue = Queue()
        self.reload: Dict[Path, List[Pipeline]] = {}
        for pipe in pipelines:
            for dep in pipe._deps:
                self.reload.setdefault(Path(dep).absolute(), []).append(pipe)
        self.stop_event = Event()
        self.thread = Thread(target=self.__thread_entry, daemon=True, name="pipeline-watch")
        self.thread.start()

    def __thread_entry(self):
        ALIVE_CACHES.append(self)

        if not len(self.reload):
            return

        for changes in watchfiles.watch(*self.reload.keys(), debounce=200, stop_event=self.stop_event):
            any_change = False
            for change, path in changes:
                if change == watchfiles.Change.modified:
                    for pipe in self.reload[Path(path).absolute()]:
                        if pipe._mark_dirty():
                            self.queue.put(pipe)
                            any_change = True
            if any_change:
                if self.window is not None:
                    self.window.post_empty_event()

    def refresh(self, before_any_update: Callable):
        if len(self.reload) and not self.thread.is_alive():
            raise Exception("Pipeline thread not running")

        called = False
        while not self.queue.empty():
            if not called:
                before_any_update()
                called = True
            pipe: Pipeline = self.queue.get()
            pipe._update(False)

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        ALIVE_CACHES.remove(self)

def __close_all():
    while ALIVE_CACHES:
       ALIVE_CACHES[0].stop()

atexit.register(__close_all)