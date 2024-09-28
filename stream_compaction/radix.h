#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();

        void sort(int n, int* odata, int* idata);

        int getMaxBit(int* idata, int n);
    }
}