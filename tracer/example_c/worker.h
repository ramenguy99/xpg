#pragma once

#define TRACER_PRIVATE_API
#include "tracer.h"

typedef struct WorkerContext {
    Thread thread;
    _Atomic(int) running;
} WorkerContext;

void worker_start(WorkerContext* ctx);
void worker_stop(WorkerContext* ctx);
