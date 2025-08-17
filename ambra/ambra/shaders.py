import hashlib
import pickle
import shutil
from pathlib import Path
from typing import List, Tuple

from platformdirs import user_cache_path
from pyxpg import slang

# TODO: logic related to shader loading:
# [ ] internal vs external shaders
# [ ] how are internal shaders compiled / shipped?
#     [ ] Do we use reflection and hot realoding for internal stuff? or only offered as a tool?
#     [ ] Do we have an exprt / development mode? Pre-exported definitely helps with first startup time

CACHE_DIR = user_cache_path("ambra")

CACHE_VERSION_MAJOR = 0
CACHE_VERSION_MINOR = 1
CACHE_VERSION_PATCH = 1
CACHE_VERSION = f"{CACHE_VERSION_MAJOR}.{CACHE_VERSION_MINOR}.{CACHE_VERSION_PATCH}"


def clear_cache() -> None:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def vulkan_version_to_minimum_supported_spirv_version(
    version: Tuple[int, int],
) -> str:
    if version[0] == 0:
        raise ValueError("Unsupported Vulkan version < 1")
    if version[0] == 1:
        if version[1] == 0:
            return "spirv_1_0"
        if version[1] == 1:
            return "spirv_1_3"
        if version[1] == 2:
            return "spirv_1_5"
        if version[1] == 3 or version[1] == 4:
            return "spirv_1_6"
    return "spirv_1_6"


def compile(file: Path, entry: str = "main", target: str = "spirv_1_3") -> slang.Shader:
    name = f"{hashlib.sha256(file.read_bytes()).hexdigest()}_{entry}_{target}_{CACHE_VERSION}.shdr"

    # Check cache
    path = Path(CACHE_DIR, name)
    if path.exists():
        try:
            with path.open("rb") as f:
                pkl = pickle.load(f)
            prog: slang.Shader = pkl[0]
            old_hashes: List[str] = pkl[1]

            # Check if any of the dependent files changed from when the shader
            # was serialized.
            assert len(prog.dependencies) == len(old_hashes)
            new_hashes = [hashlib.sha256(Path(d).read_bytes()).hexdigest() for d in sorted(prog.dependencies)]
            if new_hashes == old_hashes:
                print(f"Cache hit: {file}")
                return prog
        except Exception as e:
            print(f"Shader cache hit invalid: {e}")

    # Create prog
    print(f"Shader cache miss: {file}")
    prog = slang.compile(str(file), entry, target)
    hashes = [hashlib.sha256(Path(d).read_bytes()).hexdigest() for d in sorted(prog.dependencies)]

    # Populate cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump((prog, hashes), f)
    return prog
