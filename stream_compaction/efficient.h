#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, bool calculateTime = true);

        int compact(int n, int *odata, const int *idata);

        void optimizedScan(int n, int* odata, const int* idata, bool calculateTime = true);

        int optimizedCompact(int n, int* odata, const int* idata);

        void sharedMemoryOptimizedScan(int n, int* odata, const int* idata, bool calculateTime = true);
        int sharedMemoryOptimizedCompact(int n, int* odata, const int* idata);

    }
}
