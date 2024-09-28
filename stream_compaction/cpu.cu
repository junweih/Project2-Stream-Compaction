#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int* odata, const int* idata, bool calculateTime/* = true*/) {
            if (calculateTime)
            {
                timer().startCpuTimer();
            }

            int sum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = sum;
                int val = idata[i];
                sum += val;
            }

            if (calculateTime)
            {
                timer().endCpuTimer();
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         * goal: closely and neatly packed the elements != 0
         * Simply loop through the input array, meanwhile maintain a pointer indicating which address shall we put the next non-zero element
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; ++i) {
                int val = idata[i];
                if (val != 0) {
                    odata[count++] = val;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int* odata, const int* idata) {
            int* temp = new int[n];

            timer().startCpuTimer();
            // Step 1: map
            // map original data array (integer, Light Ray, etc) to a bool array
            for (int i = 0; i < n; ++i) {
                odata[i] = idata[i] ? 1 : 0;
            }
            int last = odata[n - 1];

            // Step 2: scan
            //scan(n, odata, odata);
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                temp[i] = sum;
                int val = odata[i];
                sum += val;
            }
            int count = last + temp[n - 1];

            // Step 3: scatter
            // preserve non-zero elements and compact them into a new array
            // for each element input[i] in original array
            //  if it's non-zero (given by mapped array)
            //    then put it at output[index], where index = scanned[i]
            for (int i = 0; i < n; ++i) {
                if (odata[i]) {
                    odata[temp[i]] = idata[i];
                }
            }
            timer().endCpuTimer();

            delete[](temp); 
            return count;
        }
    }
}
