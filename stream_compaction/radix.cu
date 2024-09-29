#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "radix.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernMapBitToBoolean(int n, int* dev_idata, int* dev_bitValueArray, int bitPos) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int bit = (dev_idata[index] >> bitPos) & 1;
            dev_bitValueArray[index] = bit == 1 ? 1 : 0;
        }

        __global__ void kernComputeComplementBitArray(int n, int* dev_bitValueArray, int* dev_complementBitArray) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            dev_complementBitArray[index] = 1 - dev_bitValueArray[index];
        }

        __global__ void kernComputeOnesTargetPosition(int n, int* dev_complementBitArray, int* dev_prefixSumArray, int* dev_onesTargetPositionArray) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int totalZeros = dev_complementBitArray[n - 1] + dev_prefixSumArray[n - 1];
            dev_onesTargetPositionArray[index] = index - dev_prefixSumArray[index] + totalZeros;
        }

        __global__ void kernComputeDestinationIndex(int n, int* dev_bitValueArray, int* dev_prefixSumArray, int* dev_onesTargetPositionArray, int* dev_destinationIndexArray) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            dev_destinationIndexArray[index] = dev_bitValueArray[index] ? dev_onesTargetPositionArray[index] : dev_prefixSumArray[index];
        }

        __global__ void kernScatter(int n, int* dev_idata, int* dev_destinationIndexArray, int* dev_odata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            dev_odata[dev_destinationIndexArray[index]] = dev_idata[index];
        }

        void printArray(const char* name, int* arr, int n) {
            std::cout << name << ": ";
            for (int i = 0; i < n; ++i) {
                std::cout << arr[i] << " ";
            }
            std::cout << std::endl;
        }

        void sort(int n, int* odata, int* idata) {

            timer().startGpuTimer();
            int* dev_idata;
            int* dev_odata;
            int* dev_bitValueArray;
            int* dev_complementBitArray;
            int* dev_prefixSumArray;
            int* dev_onesTargetPositionArray;
            int* dev_destinationIndexArray;

            int arrSize = n * sizeof(int);
            int blockPerGrid = (n + blockSize - 1) / blockSize;

            cudaMalloc((void**)&dev_idata, arrSize);
            cudaMalloc((void**)&dev_odata, arrSize);
            cudaMalloc((void**)&dev_bitValueArray, arrSize);
            cudaMalloc((void**)&dev_complementBitArray, arrSize);
            cudaMalloc((void**)&dev_prefixSumArray, arrSize);
            cudaMalloc((void**)&dev_onesTargetPositionArray, arrSize);
            cudaMalloc((void**)&dev_destinationIndexArray, arrSize);

            cudaMemcpy(dev_idata, idata, arrSize, cudaMemcpyHostToDevice);

            int* host_complementBitArray = new int[n];
            int* host_prefixSumArray = new int[n];

            for (int bitPos = 0; bitPos < 31; bitPos++) {
                // 1. Get bitValueArray
                kernMapBitToBoolean << <blockPerGrid, blockSize >> > (n, dev_idata, dev_bitValueArray, bitPos);

                // 2. Get complementBitArray
                kernComputeComplementBitArray << <blockPerGrid, blockSize >> > (n, dev_bitValueArray, dev_complementBitArray);
                cudaMemcpy(host_complementBitArray, dev_complementBitArray, arrSize, cudaMemcpyDeviceToHost);

                // 3. Compute prefix sum using optimized scan
                Efficient::optimizedScan(n, host_prefixSumArray, host_complementBitArray, false);
                cudaMemcpy(dev_prefixSumArray, host_prefixSumArray, arrSize, cudaMemcpyHostToDevice);

                // 4 & 5. Compute onesTargetPositionArray
                kernComputeOnesTargetPosition << <blockPerGrid, blockSize >> > (n, dev_complementBitArray, dev_prefixSumArray, dev_onesTargetPositionArray);

                // 6. Compute destinationIndexArray
                kernComputeDestinationIndex << <blockPerGrid, blockSize >> > (n, dev_bitValueArray, dev_prefixSumArray, dev_onesTargetPositionArray, dev_destinationIndexArray);

                // 7. Scatter elements to output array
                kernScatter << <blockPerGrid, blockSize >> > (n, dev_idata, dev_destinationIndexArray, dev_odata);

                // Swap input and output for next iteration
                std::swap(dev_idata, dev_odata);
            }

            // Copy result back to host
            cudaMemcpy(odata, dev_idata, arrSize, cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bitValueArray);
            cudaFree(dev_complementBitArray);
            cudaFree(dev_prefixSumArray);
            cudaFree(dev_onesTargetPositionArray);
            cudaFree(dev_destinationIndexArray);

            // Free temporary arrays
            delete[] host_complementBitArray;
            delete[] host_prefixSumArray;

            timer().endGpuTimer();
        }
    }
}