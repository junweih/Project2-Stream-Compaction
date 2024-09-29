#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            // Create device vectors
            thrust::device_vector<int> d_input(idata, idata + n);
            thrust::device_vector<int> d_output(n); 

            timer().startGpuTimer();

            // Perform exclusive scan
            thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
            timer().endGpuTimer();

            // Copy result back to host
            thrust::copy(d_output.begin(), d_output.end(), odata);

        }
    }
}
