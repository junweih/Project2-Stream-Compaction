#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

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


            if (calculateTime)
            {
                timer().startGpuTimer();
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

            if (calculateTime)
            {
                timer().endGpuTimer();
            }


            // Copy result back to host
            cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_buf);


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

            timer().startGpuTimer();

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

            if (calculateTime)
            {
                timer().startGpuTimer();
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

            if (calculateTime)
            {
                timer().endGpuTimer();
            }

            // Copy result back to host
            cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_buf);

        }

        // You can add a new optimized compact function here if needed
        int optimizedCompact(int n, int* odata, const int* idata) {

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

            timer().startGpuTimer();

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

#pragma region sharedOptimized

        __global__ void kernSharedMemScan(int* g_odata, const int* g_idata, int n, int* blockSums) {
            extern __shared__ int temp[];

            int thid = threadIdx.x;
            int globalId = blockIdx.x * blockDim.x + threadIdx.x;
            int offset = 1;

            // Load input into shared memory
            temp[thid] = (globalId < n) ? g_idata[globalId] : 0;
            __syncthreads();

            // Build sum in place up the tree
            for (int d = blockSize >> 1; d > 0; d >>= 1) {
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
                __syncthreads();
            }

            // Store the total sum of this block
            if (thid == 0) {
                blockSums[blockIdx.x] = temp[blockSize - 1];
                temp[blockSize - 1] = 0;
            }
            __syncthreads();

            // Traverse down the tree building the scan in place
            for (int d = 1; d < blockSize; d *= 2) {
                offset >>= 1;
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
                __syncthreads();
            }

            // Write results to device memory
            if (globalId < n) {
                g_odata[globalId] = temp[thid];
            }
        }

        __global__ void kernAddBlockSums(int* data, int* blockSums, int n) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < n && blockIdx.x > 0) {
                data[tid] += blockSums[blockIdx.x - 1];
            }
        }

        void sharedMemoryOptimizedScan(int n, int* odata, const int* idata, bool calculateTime/* = true*/) {
            int numBlocks = (n + blockSize - 1) / blockSize;

            int* d_input, * d_output, * d_blockSums;
            cudaMalloc(&d_input, n * sizeof(int));
            cudaMalloc(&d_output, n * sizeof(int));
            cudaMalloc(&d_blockSums, numBlocks * sizeof(int));

            cudaMemcpy(d_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            if (calculateTime)
            {
                timer().startGpuTimer();
            }

            // Step 1: Perform scan on each block
            kernSharedMemScan << <numBlocks, blockSize, blockSize * sizeof(int) >> > (
                d_output, d_input, n, d_blockSums);

            // Step 2: Scan block sums
            int* h_blockSums = new int[numBlocks];
            cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 1; i < numBlocks; i++) {
                h_blockSums[i] += h_blockSums[i - 1];
            }

            cudaMemcpy(d_blockSums, h_blockSums, numBlocks * sizeof(int), cudaMemcpyHostToDevice);

            // Step 3: Add block sums to each element
            kernAddBlockSums << <numBlocks, blockSize >> > (d_output, d_blockSums, n);

            if (calculateTime)
            {
                timer().endGpuTimer();
            }

            // Copy result back to host
            cudaMemcpy(odata, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Clean up
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_blockSums);
            delete[] h_blockSums;
        }


        int sharedMemoryOptimizedCompact(int n, int* odata, const int* idata) {
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

            timer().startGpuTimer();

            // Step 1: map
            Common::kernMapToBoolean << <blockPerGrid, blockSize >> > (n, dev_bools, dev_idata);

            // Step 2: Exclusive scan (using shared memory optimized version)
            sharedMemoryOptimizedScan(n, host_indices, dev_bools, false);
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