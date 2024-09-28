#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScanNaive(int n, int offset, int* odata, const int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index >= offset)
            {
                odata[index] = idata[index - offset] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            timer().startGpuTimer();

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            // Allocate device memory
            int* dev_in;
            int* dev_out;

            size_t arrSize = n * sizeof(int);
            cudaMalloc((void**)&dev_in, arrSize);
            checkCUDAError("cudaMalloc dev_in failed!");
            cudaMalloc((void**)&dev_out, arrSize);
            checkCUDAError("cudaMalloc dev_out failed!");

            // Copy input data to device
            cudaMemcpy(dev_in, idata, arrSize, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_in failed!");


            for (int d = 1; d <= ilog2ceil(n); d++)
            {
                int offset = 1 << (d - 1);
                kernScanNaive << <blocksPerGrid, blockSize >> > (n, offset, dev_out, dev_in);
                checkCUDAError("kernScanNaive failed!");

                // Swap input and output buffers
                std::swap(dev_in, dev_out);
            }


            // Copy result back to host
            //odata[0] = 0;
            cudaMemcpy(odata + 1, dev_in, arrSize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            // Free device memory
            cudaFree(dev_in);
            cudaFree(dev_out);

            timer().endGpuTimer();

        }
    }
}
