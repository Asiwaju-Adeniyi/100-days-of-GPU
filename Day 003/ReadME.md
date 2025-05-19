# Day 3: Vector Addition with CUDA  
**100 Days of GPU Challenge**

Today, I worked through a foundational CUDA example: vector addition. This is often considered the "Hello World" of parallel computing with GPUs, and it’s a great way to get hands-on with the basics of data parallelism, memory management, and kernel execution.

---

## Concepts Covered
- Difference between task and data parallelism  
- How CUDA organizes threads and memory  
- How to launch and manage kernels  
- Handling edge cases and errors  
- Writing a complete vector addition program in CUDA

---

## Parallelism in CUDA

### Task Parallelism  
Involves running different tasks or functions in parallel—ideal when operations vary but can be processed independently.

### Data Parallelism  
Runs the same operation on different pieces of data. GPUs are built for this kind of workload. Vector addition is a classic example: adding corresponding elements from two arrays in parallel.

---

## Vector Addition Breakdown

Given:

x = [x₀, x₁, ..., xₙ]
y = [y₀, y₁, ..., yₙ]
z = [z₀, z₁, ..., zₙ] where zᵢ = xᵢ + yᵢ

yaml
Copy
Edit

This operation can be distributed across thousands of GPU threads.

---

## Sequential CPU Version

```cpp
void vecadd(float* x, float* y, float* z, int N) {
    for(int i = 0; i < N; ++i) {
        z[i] = x[i] + y[i];
    }
}
Memory Management in CUDA
CUDA programs must explicitly allocate and copy memory between host (CPU) and device (GPU):

cudaMalloc allocates memory on the GPU

cudaMemcpy transfers data between CPU and GPU

cudaFree deallocates GPU memory

Memory Management Example
cpp
Copy
Edit
float *x_d, *y_d, *z_d;
cudaMalloc((void**)&x_d, N * sizeof(float));
cudaMalloc((void**)&y_d, N * sizeof(float));
cudaMalloc((void**)&z_d, N * sizeof(float));

cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

// Later: Copy result back
cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(x_d);
cudaFree(y_d);
cudaFree(z_d);
CUDA Execution Model
Threads are organized in a hierarchy:

Thread – the smallest execution unit

Block – a group of threads

Grid – a group of blocks

Each thread gets a unique index using:

cpp
Copy
Edit
int i = blockDim.x * blockIdx.x + threadIdx.x;
Launching the Kernel
cpp
Copy
Edit
const unsigned int threadsPerBlock = 256;
const unsigned int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

vecadd_kernel<<<blocks, threadsPerBlock>>>(x_d, y_d, z_d, N);
CUDA Kernel for Vector Addition
cpp
Copy
Edit
__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        z[i] = x[i] + y[i];
    }
}
The if (i < N) condition ensures we don’t access out-of-bounds memory.

Function Qualifiers
CUDA functions can have different qualifiers:

__global__: called from host, runs on device

__device__: runs and is called on the device

__host__: runs and is called on the host (default)

You can combine them if needed:

cpp
Copy
Edit
__host__ __device__ float add(float a, float b) {
    return a + b;
}
Error Handling in CUDA
Always check for CUDA errors:

cpp
Copy
Edit
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
Full Program
cpp
Copy
Edit
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N) {
        z[i] = x[i] + y[i];
    }
}

void vecadd(float* x, float* y, float* z, int N) {
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    vecadd_kernel<<<blocks, threads>>>(x_d, y_d, z_d, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main() {
    const int N = 1000000;
    float *x = (float*)malloc(N*sizeof(float));
    float *y = (float*)malloc(N*sizeof(float));
    float *z = (float*)malloc(N*sizeof(float));

    for(int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    vecadd(x, y, z, N);

    bool correct = true;
    for(int i = 0; i < N; i++) {
        if(z[i] != 3.0f) {
            correct = false;
            break;
        }
    }

    printf("Vector addition %s\n", correct ? "PASSED" : "FAILED");

    free(x);
    free(y);
    free(z);
    return 0;
}
