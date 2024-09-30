CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Junwei Huang
* Tested on: Windows 10, i7-13790F @ 5.20 GHz 32GB, RTX 4080 16GB

## Overview
------

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

------

### 1. **CPU Scan**

Use a for loop to compute an exclusive prefix sum.

![cpu_scan](\img\cpu_scan.png)

### 2. **Naive GPU Scan**

Use double-buffer to scan two array. First do exclusive scan, then do shift right to get inclusive scan array.![figure-39-2](\img\figure-39-2.jpg)

Naïve Parallel Scan needs O(log2(n)) passes. Each pass is O(n) in terms of total work, though in practice it can be less due to factors like early warp retirement on GPUs. The overall work complexity is O(n log2(n)).

### 3. **Work-Efficient GPU Scan**

#### **Step 1: Up-Sweep Phase**

We first traverse the tree from leaves to the root computing partial sums at internal nodes of the tree.

![img](/img/upsweep.png)

#### **Step 2: Down-Sweep Phase**

Then we traverse back down the tree from the root, using the partial sums from the reduce phase to build the scan in place on the array. We start by inserting zero at the root of the tree, and on each step, each node at the current level passes its own value to its left child, and the sum of its value and the former value of its left child to its right child.

![img](/img/downsweep.png)

### 4. **Work-Efficient Optimized GPU Scan Algorithm**

#### 4.**1. Thread Utilization **

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

