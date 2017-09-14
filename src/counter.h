#pragma once

#include "result.h"

class counter {
  public:
    virtual ~counter() {}

    virtual void start() = 0;
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual void stop() = 0;
    virtual void clear() = 0;

    virtual result_array total() const = 0;
    virtual result_array imbalance() const = 0;

    virtual int threads() const = 0;
    virtual result_array thread_total(int thread) const = 0;
};
