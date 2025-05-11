from pipelines import Thread, Task, Pipeline
from colors import *

def double_buffered_async_cpu_load():
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", 1, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("Commands", 3, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", 3, GREEN), Task("Present", 1.5, VIOLET)]),
    }
    p = Pipeline(threads)

    N = 2
    p.add("load", "Load", "cpubuf", count=N)
    p.add("cpu", "Commands", "cmd", count=N)
    p.produces(("load", "Load"), ("cpu", "Commands"), "cpubuf")
    p.produces(("cpu", "Commands"), ("load", "Load"), "cpubuf")
    p.produces(("cpu", "Commands"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    p.produces(("gpu", "Present"), ("cpu", "Commands"), "cmd")
    return p

def double_buffered_async_gpu_upload():
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", 2, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("Commands", 2, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", 2.75, GREEN), Task("Present", 1.25, VIOLET)]),
        "copy": Thread("GPU - COPY", [Task("Upload", 2, BLUE)]),
    }
    p = Pipeline(threads)

    N = 2
    p.add("load", "Load", "cpubuf", count=N)
    p.add("cpu", "Commands", "cmd", count=N)
    p.add("copy", "Upload", "gpubuf", count=N)

    p.produces(("load", "Load"), ("copy", "Upload"), "cpubuf")
    p.produces(("gpu", "Draw"), ("copy", "Upload"), "gpubuf")
    p.produces(("copy", "Upload"), ("gpu", "Draw"), "gpubuf")
    p.produces(("cpu", "Commands"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    p.produces(("gpu", "Draw"), ("load", "Load"), "cpubuf")
    p.produces(("gpu", "Present"), ("cpu", "Commands"), "cmd")
    return p

def double_buffered_async_gpu_upload_sync_submit():
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", 4, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("CopyCmd", 1, ORANGE), Task("Cmd", 1, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", 3, GREEN), Task("Present", 1, VIOLET)]),
        "copy": Thread("GPU - COPY", [Task("Upload", 5, BLUE)]),
    }
    p = Pipeline(threads)

    p.add("load", "Load", "cpubuf", count=3)
    p.add("cpu", "CopyCmd", "cmd", count=2)
    p.add("copy", "Upload", "gpubuf", count=2)

    p.produces(("load", "Load"), ("cpu", "CopyCmd"), "copybuf")
    p.produces(("cpu", "CopyCmd"), ("copy", "Upload"), "copybuf")
    p.produces(("cpu", "CopyCmd"), ("cpu", "Cmd"), "exec")
    p.produces(("gpu", "Present"), ("cpu", "CopyCmd"), "cmd")
    p.produces(("copy", "Upload"), ("gpu", "Draw"), "copybuf")
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    p.produces(("gpu", "Draw"), ("copy", "Upload"), "gpubuf")
    p.produces(("cpu", "Cmd"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Present"), ("load", "Load"), "cpubuf")
    return p

def double_buffered_async_gpu_upload_sync_submit_half_rate():
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", 4, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("Copy", 1, YELLOW), Task("Cmd", 1, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", 3, GREEN), Task("Present", 1, VIOLET)]),
        "copy": Thread("GPU - COPY", [Task("Upload", 5, BLUE)]),
    }
    p = Pipeline(threads)

    N = 2
    F = 2
    p.add("load", "Load", "cpubuf", count=N)
    p.add("cpu", "Copy", "cmd", count=N)
    # p.add("copy", "Upload", "gpubuf", count=N)

    p.produces(("load", "Load"), ("cpu", "Copy"), "copybuf", (F, 1))
    p.produces(("cpu", "Copy"), ("copy", "Upload"), "copybuf", (1, F))
    p.produces(("cpu", "Copy"), ("cpu", "Cmd"), "exec")
    p.produces(("gpu", "Present"), ("cpu", "Copy"), "cmd")
    p.produces(("copy", "Upload"), ("gpu", "Draw"), "copybuf", (F, 1))
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    # p.produces(("gpu", "Draw"), ("copy", "Upload"), "gpubuf", (1, F))
    p.produces(("cpu", "Cmd"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Present"), ("load", "Load"), "cpubuf", (1, F))
    return p
