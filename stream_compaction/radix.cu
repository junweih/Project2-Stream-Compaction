#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"

namespace StreamCompaction {
    namespace RadixSort
    {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __device__ inline int findMSB(int n) {
            return 31 - __clz(n);
        }

        __global__ void calculateMaxBit(int* d_input, int* d_max_bit, int n) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            __shared__ int shared_max_bit;
            if (threadIdx.x == 0) {
                shared_max_bit = 0;
            }
            __syncthreads();

            for (int i = tid; i < n; i += stride) {
                int local_max_bit = findMSB(d_input[i]);
                atomicMax(&shared_max_bit, local_max_bit);
            }

            __syncthreads();

            if (threadIdx.x == 0) {
                atomicMax(d_max_bit, shared_max_bit);
            }
        }

        __global__ void computeHistogram(int* d_input, int* d_histogram, int n, int bit) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            __shared__ int local_histogram[RADIX];
            if (threadIdx.x < RADIX) {
                local_histogram[threadIdx.x] = 0;
            }
            __syncthreads();

            for (int i = tid; i < n; i += stride) {
                int bin = (d_input[i] >> bit) & 1;
                atomicAdd(&local_histogram[bin], 1);
            }

            __syncthreads();

            if (threadIdx.x < RADIX) {
                atomicAdd(&d_histogram[threadIdx.x], local_histogram[threadIdx.x]);
            }
        }

        __global__ void scan(int* d_histogram, int* d_scanned_histogram) {
            __shared__ int temp[RADIX];

            int tid = threadIdx.x;
            temp[tid] = (tid > 0) ? d_histogram[tid - 1] : 0;
            __syncthreads();

            for (int stride = 1; stride < RADIX; stride *= 2) {
                int index = (tid + 1) * 2 * stride - 1;
                if (index < RADIX) {
                    temp[index] += temp[index - stride];
                }
                __syncthreads();
            }

            for (int stride = RADIX / 4; stride > 0; stride /= 2) {
                __syncthreads();
                int index = (tid + 1) * 2 * stride - 1;
                if ((index + stride) < RADIX) {
                    temp[index + stride] += temp[index];
                }
            }

            __syncthreads();
            d_scanned_histogram[tid] = temp[tid];
        }

        __global__ void reorder(int* d_input, int* d_output, int* d_scanned_histogram, int n, int bit) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = tid; i < n; i += stride) {
                int bin = (d_input[i] >> bit) & 1;
                int pos = atomicAdd(&d_scanned_histogram[bin], 1);
                d_output[pos] = d_input[i];
            }
        }

        void sort(int n, int* odata, int* idata) {
            timer().startGpuTimer();

            // Allocate device memory
            int* d_input, * d_output, * d_histogram, * d_scanned_histogram, * d_max_bit;
            cudaMalloc((void**)&d_input, n * sizeof(int));
            cudaMalloc((void**)&d_output, n * sizeof(int));
            cudaMalloc((void**)&d_histogram, RADIX * sizeof(int));
            cudaMalloc((void**)&d_scanned_histogram, RADIX * sizeof(int));
            cudaMalloc((void**)&d_max_bit, sizeof(int));

            // Copy input data to device
            cudaMemcpy(d_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Calculate max bit
            int gridSize = (n + blockSize - 1) / blockSize;
            cudaMemset(d_max_bit, 0, sizeof(int));
            calculateMaxBit << <gridSize, blockSize >> > (d_input, d_max_bit, n);

            int h_max_bit;
            cudaMemcpy(&h_max_bit, d_max_bit, sizeof(int), cudaMemcpyDeviceToHost);
            h_max_bit++; // Add 1 because we count from 0

            // Main radix sort loop
            for (int bit = 0; bit < h_max_bit; ++bit) {
                // Compute histogram
                cudaMemset(d_histogram, 0, RADIX * sizeof(int));
                computeHistogram << <gridSize, blockSize >> > (d_input, d_histogram, n, bit);

                // Scan histogram
                scan << <1, RADIX >> > (d_histogram, d_scanned_histogram);

                // Reorder elements
                reorder << <gridSize, blockSize >> > (d_input, d_output, d_scanned_histogram, n, bit);

                // Swap input and output pointers
                int* temp = d_input;
                d_input = d_output;
                d_output = temp;
            }

            // Copy result back to host
            cudaMemcpy(odata, d_input, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_histogram);
            cudaFree(d_scanned_histogram);
            cudaFree(d_max_bit);

            timer().endGpuTimer();
        }
    }
}