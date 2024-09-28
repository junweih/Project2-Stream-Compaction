#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient 
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#pragma region base
        __global__ void kernUpSweep(int n, int d, int* data)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            int offset1 = 1 << d + 1;
            int offset2 = 1 << d;

            if (index % offset1 == 0)
            {
                data[index + offset1 - 1] += data[index + offset2 - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* data)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int offset1 = 1 << d + 1;
            int offset2 = 1 << d;
            if (index % offset1 == 0)
            {
                int t = data[index + offset2 - 1];
                data[index + offset2 - 1] = data[index + offset1 - 1];
                data[index + offset1 - 1] += t;

            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata, bool calculateTime /*= true*/) {
            if (calculateTime)
            {
                timer().startGpuTimer();
            }

            int* dev_buf;
            int power2 = ilog2ceil(n);

            int chunk = 1 << power2;

            // Calculate grid dimensions
            dim3 blockPerGrid((chunk + blockSize - 1) / blockSize);

            // Allocate device memory
            cudaMalloc((void**)&dev_buf, chunk * sizeof(int));

            // Copy input data to device
            cudaMemcpy(dev_buf, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Zero-pad the rest of the array if necessary
            if (chunk > n)
            {
                cudaMemset(dev_buf + n, 0, (chunk - n) * sizeof(int));
            }



            // Up-sweep phase
            for (int d = 0; d < power2 - 1; d++)
            {
                kernUpSweep << < blockPerGrid, blockSize >> > (chunk, d, dev_buf);
                cudaDeviceSynchronize();
            }

            cudaMemset(dev_buf + chunk - 1, 0, sizeof(int)); // set root to zero

            // Down-sweep phase
            for (int d = power2 - 1; d >= 0; d--)
            {
                kernDownSweep << < blockPerGrid, blockSize >> > (chunk, d, dev_buf);
                cudaDeviceSynchronize();
            }



            // Copy result back to host
            cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_buf);

            if (calculateTime)
            {
                timer().endGpuTimer();
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            int* dev_bools;
            int* dev_idata;
            int* dev_odata;
            int* dev_indices;
            int* host_indices = new int[n];
            int count = 0;

            // Calculate grid dimensions
            dim3 blockPerGrid((n + blockSize - 1) / blockSize);

            // Allocate device memory
            size_t arrSize = n * sizeof(int);
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));

            // Copy input data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Step 1: map
            Common::kernMapToBoolean << < blockPerGrid, blockSize >> > (n, dev_bools, dev_idata);


            // Step 2: Exclusive scan
            scan(n, host_indices, dev_bools, false);
            cudaMemcpy(dev_indices, host_indices, n * sizeof(int), cudaMemcpyHostToDevice);

            // Step 3: scatter
            Common::kernScatter << < blockPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            // Copy over last elements from GPU to local to return
            cudaMemcpy(&count, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            count += (int)(idata[n - 1] != 0);

            cudaMemcpy(odata, dev_odata, arrSize, cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            delete[] host_indices;
            return count;
        }
#pragma endregion

#pragma region optimize 
        __global__ void kernOptimizedUpSweep(int n, int d, int* data)
        {
            int thid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = 1 << (d + 1);

            if (thid < n / stride)
            {
                int i = (thid + 1) * stride - 1;
                if (i < n)
                {
                    data[i] += data[i - (stride >> 1)];
                }
            }
        }

        __global__ void kernOptimizedDownSweep(int n, int d, int* data)
        {
            int thid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = 1 << (d + 1);

            if (thid < n / stride)
            {
                int i = (thid + 1) * stride - 1;
                if (i < n)
                {
                    int t = data[i];
                    data[i] += data[i - (stride >> 1)];
                    data[i - (stride >> 1)] = t;
                }
            }
        }

        void optimizedScan(int n, int* odata, const int* idata, bool calculateTime/* = true*/) {
            if (calculateTime)
            {
                timer().startGpuTimer();
            }

            int* dev_buf;
            int power2 = ilog2ceil(n);
            int chunk = 1 << power2;

            // Allocate device memory
            cudaMalloc((void**)&dev_buf, chunk * sizeof(int));

            // Copy input data to device
            cudaMemcpy(dev_buf, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Zero-pad the rest of the array if necessary
            if (chunk > n)
            {
                cudaMemset(dev_buf + n, 0, (chunk - n) * sizeof(int));
            }

            // Up-sweep phase
            for (int d = 0; d < power2; d++)
            {
                int num_threads = chunk / (2 << d);
                dim3 gridSize((num_threads + blockSize - 1) / blockSize);  
                kernOptimizedUpSweep << <gridSize, blockSize >> > (chunk, d, dev_buf);
                cudaDeviceSynchronize();
            }

            cudaMemset(dev_buf + chunk - 1, 0, sizeof(int)); // set root to zero

            // Down-sweep phase
            for (int d = power2 - 1; d >= 0; d--)
            {
                int num_threads = chunk / (2 << d);
                dim3 gridSize((num_threads + blockSize - 1) / blockSize);
                kernOptimizedDownSweep << <gridSize, blockSize >> > (chunk, d, dev_buf);
                cudaDeviceSynchronize();
            }

            // Copy result back to host
            cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_buf);

            if (calculateTime)
            {
                timer().endGpuTimer();
            }
        }

        // You can add a new optimized compact function here if needed
        int optimizedCompact(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            int* dev_bools;
            int* dev_idata;
            int* dev_odata;
            int* dev_indices;
            int* host_indices = new int[n];
            int count = 0;

            // Calculate grid dimensions
            dim3 blockPerGrid((n + blockSize - 1) / blockSize);

            // Allocate device memory
            size_t arrSize = n * sizeof(int);
            cudaMalloc((void**)&dev_bools, arrSize);
            cudaMalloc((void**)&dev_idata, arrSize);
            cudaMalloc((void**)&dev_odata, arrSize);
            cudaMalloc((void**)&dev_indices, arrSize);

            // Copy input data to device
            cudaMemcpy(dev_idata, idata, arrSize, cudaMemcpyHostToDevice);

            // Step 1: map
            Common::kernMapToBoolean << <blockPerGrid, blockSize >> > (n, dev_bools, dev_idata);

            // Step 2: Exclusive scan (using optimized version)
            optimizedScan(n, host_indices, dev_bools, false);
            cudaMemcpy(dev_indices, host_indices, arrSize, cudaMemcpyHostToDevice);

            // Step 3: scatter
            Common::kernScatter << <blockPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            // Copy over last elements from GPU to local to return
            cudaMemcpy(&count, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            count += (int)(idata[n - 1] != 0);

            cudaMemcpy(odata, dev_odata, arrSize, cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            delete[] host_indices;
            return count;
        }
#pragma endregion


    }
}