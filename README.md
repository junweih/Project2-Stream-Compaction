CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Junwei Huang
* Tested on: Windows 10, i7-13790F @ 5.20 GHz 32GB, RTX 4080 16GB

## Overview

This project implements a suite of GPU-accelerated algorithms for stream compaction, scan operations, and sorting. The main features include:

1. **CPU Scan Algorithm**
   - A straightforward implementation CPU prefix sum (scan) operation using loop.
2. **Naive GPU Scan and Compact Algorithm**
   - A straightforward implementation of parallel prefix sum (scan) operation.
3. **Work-Efficient GPU Scan Algorithm**
   - An optimized version of the scan algorithm designed for better GPU utilization.
4. **Work-Efficient Optimized GPU Scan Algorithm**
   - Further optimization of the work-efficient algorithm to improve performance.
5. **Work-Efficient Shared-Memory & Hardware GPU Scan Algorithm**
   - Utilizes shared memory for faster operations within GPU blocks.
6. **Radix Sort GPU Parallel Algorithm**
   - Implements a parallel radix sort algorithm leveraging GPU capabilities.



## Description


### 1. **CPU Scan**

Use a for loop to compute an exclusive prefix sum.

![cpu_scan](/img/cpu_scan.png)

### 2. **Naive GPU Scan**

Use double-buffer to scan two array. First do exclusive scan, then do shift right to get inclusive scan array.![figure-39-2](/img/figure-39-2.jpg)

Naïve Parallel Scan needs O(log2(n)) passes. Each pass is O(n) in terms of total work, though in practice it can be less due to factors like early warp retirement on GPUs. The overall work complexity is O(n log2(n)).

### 3. **Work-Efficient GPU Scan**

#### **Step 1: Up-Sweep Phase**

We first traverse the tree from leaves to the root computing partial sums at internal nodes of the tree.

![img](/img/upsweep.png)

#### **Step 2: Down-Sweep Phase**

Then we traverse back down the tree from the root, using the partial sums from the reduce phase to build the scan in place on the array. We start by inserting zero at the root of the tree, and on each step, each node at the current level passes its own value to its left child, and the sum of its value and the former value of its left child to its right child.

![img](/img/downsweep.png)

### 4. **Work-Efficient Optimized GPU Scan Algorithm**

#### 4.1. Thread Utilization 

Original:

```c++
 cpp dim3 blockPerGrid((chunk + blockSize - 1) / blockSize); 
// This launches the same number of threads for each level
```

Optimized: 

```C++
int num_threads = chunk / (2 << d); 
dim3 gridSize((num_threads + blockSize - 1) / blockSize); 
    // This reduces the number of threads at each level
```

Explanation:

- In the original version, we launched the same number of threads for each level of the sweep.
- In the optimized version, we reduce the number of threads at each level.
- This is because, at deeper levels of the tree, fewer elements need to be processed.
- By launching fewer threads, we improve occupancy and reduce unnecessary work.



#### 4.**2. Kernel Indexing**

Original:
```cpp
int offset1 = 1 << d + 1;
int offset2 = 1 << d;
if (index % offset1 == 0)
{
    // Process data
}
```

Optimized:
```cpp
int stride = 1 << (d + 1);
if (thid < n / stride)
{
    int i = (thid + 1) * stride - 1;
    // Process data
}
```

Explanation:
- The original version used modulo operations to determine which threads should work.
- The optimized version uses a stride-based approach.
- This new approach ensures that only the necessary threads are doing work at each level.
- It also compacts the active threads, improving memory access patterns.

#### **4.3. Early Termination**

Original:
```cpp
if (index >= n)
{
    return;
}
```

Optimized:
```cpp
if (thid < n / stride)
{
    // Process data
}
```

Explanation:
- The original version checked if the thread index was out of bounds.
- The optimized version checks if the thread should be active at the current level.
- This allows for early termination of unnecessary threads, reducing wasted work.

#### **4.4. Memory Access Pattern**

Original:
```cpp
data[index + offset1 - 1] += data[index + offset2 - 1];
```

Optimized:
```cpp
int i = (thid + 1) * stride - 1;
if (i < n)
{
    data[i] += data[i - (stride >> 1)];
}
```

Explanation:
- The original version had a less predictable memory access pattern.
- The optimized version uses a more regular stride-based pattern.
- This can lead to better memory coalescing, improving memory bandwidth utilization.

### 5. Work-Efficient Shared Memory Optimized GPU Scan Algorithm

#### 5.1. Shared Memory Usage

The `kernSharedMemScan` kernel utilizes shared memory to store intermediate results:

```cuda
extern __shared__ int temp[];
```

This allows for faster access to data within a thread block, reducing global memory traffic.

#### 5.2. Block-level Scan

The algorithm first performs a scan operation within each thread block:

- Data is loaded from global memory into shared memory.
- An up-sweep (reduction) phase builds partial sums.
- A down-sweep phase distributes the partial sums.

This block-level scan is efficient because it operates entirely in shared memory.

#### 5.3. Handling Large Arrays

For arrays larger than a single block can handle, the algorithm uses a multi-step approach:

a) Each block performs a local scan and stores its total sum.
b) These block sums are then scanned (on the CPU in this implementation, but could be done on the GPU for larger datasets).
c) The scanned block sums are then added back to each element in the corresponding block.

#### 5.4. Kernel Structure

The `kernSharedMemScan` kernel is structured as follows:

1. Load data into shared memory
2. Perform up-sweep (reduction) phase
3. Store block sum and reset last element
4. Perform down-sweep phase
5. Write results back to global memory

#### 5.5. Block Sum Handling

The `kernAddBlockSums` kernel is used to add the scanned block sums back to each element:

```cuda
__global__ void kernAddBlockSums(int* data, int* blockSums, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && blockIdx.x > 0) {
        data[tid] += blockSums[blockIdx.x - 1];
    }
}
```

This step ensures that the scan is correct across block boundaries.

#### 5.6. Optimization Benefits

- Reduced global memory accesses
- Utilization of fast shared memory
- Parallel processing within blocks
- Efficient handling of large arrays through block-level processing

### 6. **GPU Parallel Radix Sort Algorithm**

#### Step 1. Sort on each digit, right to left

In this step, the algorithm processes each digit of the numbers, starting from the least significant digit (rightmost) and moving towards the most significant digit (leftmost).

![](/img/sortdigit.png)
For each digit:

Extract the digit from each number.
Group numbers based on the value of this digit (0-9 for decimal, 0-1 for binary, etc.).
Maintain the relative order of numbers within each group.

####  Step 2. For each digit, sort using Scan
For each digit, the sorting process utilizes a parallel scan operation. Here's how it works:

![radixDigitSort](/img/radixDigitSort.jpg)

1. Create a Bit Array: For each possible value of the digit (e.g., 0 and 1 for binary), create a bit array where 1 indicates the presence of that digit value at a position, and 0 indicates its absence.
2. Perform Scan: Use the shared-memory optimized scan algorithm on these bit arrays. This step computes the prefix sum, which effectively determines the output position for each element.
3. Scatter Elements: Based on the scan results, scatter the elements to their new positions in the output array.



## Results

Test case configuration: blockSize = 256, array size = **2^27 (134,217,728)**

```

****************
** SCAN TESTS **
****************
    [  14  11  26  15   7   6  45  48  43   0  48  16  21 ...   9   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 54.3735ms    (std::chrono Measured)
    [   0  14  25  51  66  73  79 124 172 215 215 263 279 ... -1008022159 -1008022150 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 54.4342ms    (std::chrono Measured)
    [   0  14  25  51  66  73  79 124 172 215 215 263 279 ... -1008022215 -1008022203 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 55.945ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 52.6863ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 31.3832ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 28.8604ms    (CUDA Measured)
    passed
==== work-efficient scan optimized, power-of-two ====
   elapsed time: 19.514ms    (CUDA Measured)
    passed
==== work-efficient scan optimized, non-power-of-two ====
   elapsed time: 19.9905ms    (CUDA Measured)
    passed
==== work-efficient scan shared memory optimized, power-of-two ====
   elapsed time: 5.90288ms    (CUDA Measured)
    passed
==== work-efficient scan shared memory optimized, non-power-of-two ====
   elapsed time: 5.89277ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.34464ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 2.4017ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   1   2   2   1   3   1   3   1   3   0   2   1 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 227.823ms    (std::chrono Measured)
    [   1   1   2   2   1   3   1   3   1   3   2   1   3 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 228.304ms    (std::chrono Measured)
    [   1   1   2   2   1   3   1   3   1   3   2   1   3 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 395.781ms    (std::chrono Measured)
    [   1   1   2   2   1   3   1   3   1   3   2   1   3 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 183.654ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 163.749ms    (CUDA Measured)
    passed
==== work-efficient compact optimized, power-of-two ====
   elapsed time: 164.7ms    (CUDA Measured)
    passed
==== work-efficient compact optimized, non-power-of-two ====
   elapsed time: 151.003ms    (CUDA Measured)
    passed
==== work-efficient compact shared memory optimized, power-of-two ====
   elapsed time: 158.275ms    (CUDA Measured)
    passed
==== work-efficient compact shared memory optimized, non-power-of-two ====
   elapsed time: 164.255ms    (CUDA Measured)
    passed

*****************************
** RADIX SORT TEST **
*****************************
    [ 193 7143 14963 21127 9234 17229 18039 1303 28875 30503 1816 12262 9863 ... 15770 3834 ]
==== cpu std::sort, power-of-two ====
   elapsed time: 5792.38ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]
==== cpu std::sort, non-power-of-two ====
   elapsed time: 5757ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]
==== radix sort, power-of-two ====
   elapsed time: 6011.1ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]
    passed
==== radix sort, non-power-of-two ====
   elapsed time: 5991.25ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]
    passed
```

## Performance Analysis

### 1.  All GPU Scan (Naive, Work-Efficient, Work-Efficient Optimized, Work-Efficient Shared Memory and Thrust) vs serial CPU Scan Comparison

Test case configuration: block size: 256. array size: [2^20, 2^30]

![Runtime vs. Array Size for CPU and GPU Scan Algorithms](/img/Runtime vs. Array Size for CPU and GPU Scan Algorithms.png)

*Figure 1: Runtime vs Array Size for CPU and GPU Efficient Scan*

| Scan Performance Analysis (MS) | CPU Scan | GPU Naive Scan | GPU Efficient Scan | GPU Optimized Efficient Scan | GPU Shared Memory Efficient Scan | Thrust Scan |
| ------------------------------ | -------- | -------------- | ------------------ | ---------------------------- | -------------------------------- | ----------- |
| 1 << 30                        | 456      | 498            | 252                | 160                          | 47                               | 15          |
| 1 << 28                        | 105      | 114            | 61                 | 40                           | 11                               | 4           |
| 1 << 25                        | 14       | 13             | 31.9               | 5.2                          | 1.9                              | 1           |
| 1<<22                          | 1.78     | 0.6            | 1.42               | 0.83                         | 0.34                             | 0.38        |
| 1<<20                          | 0.393    | 0.333          | 0.779              | 0.887                        | 0.121                            | 0.287       |



#### **Large Arrays (2^30 elements)**

Performance order (fastest to slowest):

1. Thrust
2. GPU Shared Memory Efficient Scan
3. GPU Optimized Efficient Scan
4. GPU Efficient Scan
5. GPU Naive Scan
6. CPU Scan

#### Explanation:

1. Thrust:
   - Thrust is a highly optimized CUDA library that leverages advanced GPU optimization techniques.
   - It's specifically designed for parallel algorithms like scan, making it extremely efficient for large datasets.
   - The simple implementation using `thrust::exclusive_scan` allows for maximum performance with minimal code.
2. GPU Shared Memory Efficient Scan:
   - This implementation uses shared memory, which is much faster than global memory.
   - It performs a block-level scan in shared memory and then combines the results, reducing global memory accesses.
   - The `kernSharedMemScan` kernel efficiently utilizes the GPU's parallel architecture.
3. GPU Optimized Efficient Scan:
   - This version improves upon the basic efficient scan by optimizing thread usage and memory access patterns.
   - The `kernOptimizedUpSweep` and `kernOptimizedDownSweep` kernels reduce thread divergence and improve coalesced memory access.
4. GPU Efficient Scan:
   - This implementation uses the work-efficient parallel scan algorithm.
   - It performs up-sweep and down-sweep phases on the GPU, which is more efficient than the naive approach for large arrays.
5. GPU Naive Scan:
   - While this runs on the GPU, it's not as efficient as the other GPU methods for large arrays.
   - It requires multiple kernel launches, one for each level of the scan, which can be costly for large arrays.
6. CPU Scan:
   - The CPU version is a simple sequential implementation.
   - For large arrays, it cannot compete with parallel GPU implementations due to limited parallelism.

#### Small Arrays (2^20 elements)

Performance order (fastest to slowest):

1. GPU Shared Memory Efficient Scan
2. Thrust
3. GPU Naive Scan
4. CPU Scan
5. GPU Optimized Efficient Scan
6. GPU Efficient Scan

#### Explanation:

1. GPU Shared Memory Efficient Scan:
   - For smaller arrays, this method benefits from reduced kernel launch overhead and efficient use of shared memory.
   - The entire array or large portions of it can fit into shared memory, maximizing performance.
2. Thrust:
   - While still very fast, Thrust's overhead might be more noticeable for smaller arrays.
   - It's still highly efficient but may not outperform a well-implemented custom shared memory solution for this size.
3. GPU Naive Scan:
   - For smaller arrays, the simplicity of this method and reduced number of iterations make it surprisingly effective.
   - The overhead of multiple kernel launches is less significant for smaller datasets.
4. CPU Scan:
   - With smaller arrays, the CPU's cache can be more effectively utilized.
   - The overhead of GPU memory transfers becomes more significant, making CPU performance relatively better.
5. GPU Optimized Efficient Scan and GPU Efficient Scan:
   - These methods, while efficient for large arrays, may have more overhead than simpler methods for smaller datasets.
   - The complexity of these algorithms doesn't pay off for smaller arrays where simpler methods can be more direct and effective.

#### Conclusion

The performance characteristics demonstrate that the most suitable algorithm depends on the size of the input data. For very large datasets, highly optimized GPU methods like Thrust and shared memory implementations excel. For smaller datasets, simpler GPU methods or even CPU implementations can be competitive due to reduced overhead and better cache utilization. The choice of algorithm should be based on the expected size of the input data and the specific hardware available.

### 2. Block size optimization for minimal runtime

Test case configuration: power-of-two array size = **2^27 (134,217,728)**

![Block size optimization for minimal runtime](/img/Block size optimization for minimal runtime.png)

| Block Size | GPU Naive Scan | GPU Efficient Scan | GPU Optimized Efficient Scan | GPU Shared Memory Efficient Scan | Thrust Scan |
| :--------: | :------------: | :----------------: | :--------------------------: | :------------------------------: | :---------: |
|     32     |      119       |        258         |              37              |                28                |      4      |
|     64     |      107       |        129         |              37              |                17                |      4      |
|    128     |      111       |         90         |              39              |                13                |      4      |
|    256     |      112       |         59         |              39              |                11                |      4      |
|    512     |      107       |         80         |              37              |                10                |      4      |
|    1024    |      101       |        105         |              37              |                13                |      4      |

1. GPU Naive Scan:
   - Performance is relatively consistent across block sizes, ranging from 101 to 119 ms.
   - This suggests that the naive approach doesn't benefit much from increased parallelism with larger block sizes.
2. GPU Efficient Scan:
   - Shows significant improvement as block size increases, from 258 ms at 32 to 59 ms at 256.
   - However, performance degrades slightly for very large block sizes (512 and 1024).
   - This indicates that the efficient algorithm benefits from increased parallelism up to a point, after which overhead may increase.
3. GPU Optimized Efficient Scan:
   - Maintains very consistent performance (37-39 ms) across all block sizes.
   - This suggests that the optimizations applied make the algorithm less sensitive to block size variations.
4. GPU Shared Memory Efficient Scan:
   - Shows the most dramatic improvement with increasing block size, from 28 ms at 32 to 10 ms at 512.
   - Slight performance degradation at 1024, possibly due to shared memory limitations.
   - This algorithm benefits greatly from larger block sizes due to more efficient use of shared memory.
5. Thrust Scan:
   - Consistently performs at 4 ms regardless of block size.
   - This indicates that Thrust's implementation is highly optimized and likely uses advanced techniques that are not affected by the block size parameter in this range.

Key Observations:

1. Thrust consistently outperforms all other implementations, showcasing the benefits of using a highly optimized library.
2. The shared memory approach shows the best scaling with block size among the custom implementations.
3. The naive approach doesn't scale well with block size, highlighting the importance of more sophisticated algorithms for GPU computing.
4. There's often a "sweet spot" for block size (around 256-512 here) where performance is optimal before degrading due to various overheads.
