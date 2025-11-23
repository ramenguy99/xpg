# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

import hashlib
import logging
import pickle
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

from platformdirs import user_cache_path
from pyxpg import slang

logger = logging.getLogger(__name__)

CACHE_DIR = user_cache_path("ambra")

CACHE_VERSION_MAJOR = 0
CACHE_VERSION_MINOR = 2
CACHE_VERSION_PATCH = 0
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
        elif version[1] == 1:
            return "spirv_1_3"
        elif version[1] == 2:
            return "spirv_1_5"
        else:
            return "spirv_1_6"
    else:
        return "spirv_1_6"


def compile(
    file: Path,
    entry: str = "main",
    target: str = "spirv_1_3",
    defines: Optional[List[Tuple[str, str]]] = None,
    include_paths: Optional[List[Union[Path, str]]] = None,
) -> slang.Shader:
    defines_list = defines or []
    include_paths_list = [str(p) for p in include_paths] if include_paths is not None else []

    defines_bytes = b"".join([f"{k}\0{v}\0".encode() for k, v in sorted(defines_list)])
    include_paths_bytes = b"".join([f"{p}\0".encode() for p in sorted(include_paths_list)])

    hexdigest = hashlib.sha256(b"\0".join([defines_bytes, include_paths_bytes, file.read_bytes()])).hexdigest()
    name = f"{hexdigest}_{entry}_{target}_{CACHE_VERSION}.shdr"

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
                logger.info("Shader cache hit: %s -> %s ", file, path)
                return prog
        except Exception as e:
            logger.warning("Shader cache hit invalid: %s (%s)", file, e)

    # Create prog
    logger.info("Shader cache miss: %s", file)
    prog = slang.compile(str(file), entry, target, defines_list, include_paths_list)
    hashes = [hashlib.sha256(Path(d).read_bytes()).hexdigest() for d in sorted(prog.dependencies)]

    # Populate cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump((prog, hashes), f)
    return prog
