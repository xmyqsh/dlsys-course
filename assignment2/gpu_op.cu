#include "./c_runtime_api.h"
#include <iostream>
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define MAX_THREADS_NUM 512
#define MAX_BLOCKS_NUM 4096
#define BLOCK_NUM(count) min(((count + MAX_THREADS_NUM - 1) / MAX_THREADS_NUM), MAX_BLOCKS_NUM)
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

__global__ void matrix_array_set_kernel(int count,
                                        float *arr,
                                        float value) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    arr[index] = value;
  }
}

__global__ void matrix_relu_kernel(int count,
                                   const float *input,
                                   float *output) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    output[index] = input[index] > 0 ? input[index] : 0;
  }
}

__global__ void matrix_relu_gradient_kernel(int count,
                                            const float *input,
                                            const float *in_grad,
                                            float *output) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    output[index] = input[index] > 0 ? in_grad[index] : 0;
  }
}

__global__ void matrix_softmax_kernel(int nrow, int ncol,
                                      const float *input,
                                      float *output) {
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input += y * ncol;
  output += y * ncol;
  float maxval = *input;
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input[x]);
  }
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input[x] - maxval);
  }
  for (int x = 0; x < ncol; ++x) {
    output[x] = exp(input[x] - maxval) / sum;
  }
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  assert(arr->ndim == 2);
  int count = arr->shape[0] * arr->shape[1];
  float *arr_data = (float *)arr->data;
  matrix_array_set_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, arr_data, value);
  return 0;
}

__global__ void matrix_broadcast_to_kernel(int count, int input_count,
                                   const float *input_data,
                                   float *output_data) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    output_data[index] = input_data[index % input_count];
  }
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 3);
  assert(input->shape[0] == output->shape[1] &&
         input->shape[1] == output->shape[2]);
  int input_count = input->shape[0] * input->shape[1];
  int count = output->shape[0] * output->shape[1] * output->shape[2];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  matrix_broadcast_to_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, input_count, input_data, output_data);
  return 0;
}

/*
__global__ void matrix_reduce_sum_axix_zero_kernel(int count, int output_count,
                                                   const float *input_data,
                                                   float *output_data) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    output_data[index % output_count] += input_data[index];
  }
}
*/

__global__ void matrix_reduce_sum_axix_zero_kernel(int count, int input_axiz_zero,
                                                   const float *input_data,
                                                   float *output_data) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    float sum = 0;
    for (int i = 0; i < input_axiz_zero; ++i) {
      sum += input_data[i * count + index];
    }
    output_data[index] = sum;
  }
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 3);
  assert(output->ndim == 2);
  assert(input->shape[1] == output->shape[0] &&
         input->shape[2] == output->shape[1]);
  int count = output->shape[0] * output->shape[1];
  int input_axiz_zero = input->shape[0];
  const float *input_data = (const float*)input->data;
  float *output_data = (float *)output->data;
  matrix_reduce_sum_axix_zero_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, input_axiz_zero, input_data, output_data);
  return 0;
}

__global__ void matrix_elementwise_add_kernel(int count,
                                              const float *matA_data,
                                              const float *matB_data,
                                              float *output_data) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    output_data[index] = matA_data[index] + matB_data[index];
  }
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(output->ndim == 2);
  assert(matA->shape[0] == output->shape[0] &&
         matA->shape[1] == output->shape[1]);
  assert(matB->shape[0] == output->shape[0] &&
         matB->shape[1] == output->shape[1]);
  int count = output->shape[0] * output->shape[1];
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  matrix_elementwise_add_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, matA_data, matB_data, output_data);
  return 0;
}

__global__ void matrix_elementwise_add_by_const_kernel(int count,
                                              const float *input_data,
                                              float val,
                                              float *output_data) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    output_data[index] = input_data[index] + val;
  }
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int count = output->shape[0] * output->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  matrix_elementwise_add_by_const_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, input_data, val, output_data);
  return 0;
}

__global__ void matrix_elementwise_multiply_kernel(int count,
                                              const float *matA_data,
                                              const float *matB_data,
                                              float *output_data) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    output_data[index] = matA_data[index] * matB_data[index];
  }
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(output->ndim == 2);
  assert(matA->shape[0] == output->shape[0] &&
         matA->shape[1] == output->shape[1]);
  assert(matB->shape[0] == output->shape[0] &&
         matB->shape[1] == output->shape[1]);
  int count = output->shape[0] * output->shape[1];
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  matrix_elementwise_multiply_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, matA_data, matB_data, output_data);
  return 0;
}

__global__ void matrix_multiply_by_const_kernel(int count,
                                              const float *input_data,
                                              float val,
                                              float *output_data) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    output_data[index] = input_data[index] * val;
  }
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int count = output->shape[0] * output->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  matrix_multiply_by_const_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *matC_data = (float *)matC->data;
  float alpha = 1, beta = 0;

  /*
  float *h_A, *h_B, *h_C;
  h_A = (float*)malloc(sizeof(float) * matA->shape[0] * matA->shape[1]);
  h_B = (float*)malloc(sizeof(float) * matB->shape[0] * matB->shape[1]);
  h_C = (float*)malloc(sizeof(float) * matC->shape[0] * matC->shape[1]);
  cudaMemcpy(h_A, matA_data,sizeof(float) * matA->shape[0] * matA->shape[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, matB_data,sizeof(float) * matB->shape[0] * matB->shape[1], cudaMemcpyDeviceToHost);

  std::cout << h_A[0] << std::endl;
  std::cout << "print A:" << std::endl;
  std::cout << matA->shape[0] << std::endl;
  std::cout << matA->shape[1] << std::endl;

  for (int i = 0; i < matA->shape[0]; ++i) {
    for (int j = 0; j < matA->shape[1]; ++j) {
      std::cout << h_A[i * matA->shape[1] + j] << '\t';
    }
    std::cout << std::endl;
  }

  std::cout << "print B:" << std::endl;
  std::cout << matB->shape[0] << std::endl;
  std::cout << matB->shape[1] << std::endl;
  for (int i = 0; i < matB->shape[0]; ++i) {
    for (int j = 0; j < matB->shape[1]; ++j) {
      std::cout << h_B[i * matB->shape[1] + j] << '\t';
    }
    std::cout << std::endl;
  }
  */

  /*
  cublasSgemm(
          handle,
          (transposeA ? CUBLAS_OP_N : CUBLAS_OP_T),
          //(transposeA ? CUBLAS_OP_T : CUBLAS_OP_N),
          (transposeB ? CUBLAS_OP_N : CUBLAS_OP_T),
          //(transposeB ? CUBLAS_OP_T : CUBLAS_OP_N),
          matC->shape[0],
          matC->shape[1],
          matA->shape[1],
          &alpha,
          matA_data,
          (transposeA ? matA->shape[0] : matA->shape[1]),
          //(transposeA ? matA->shape[1] : matA->shape[0]),
          matB_data,
          (transposeB ? matB->shape[0] : matB->shape[1]),
          //(transposeB ? matB->shape[1] : matB->shape[0]),
          &beta,
          matC_data,
          matC->shape[0]
  );
  */
  cublasSgemm(
          handle,
          //(transposeB ? CUBLAS_OP_N : CUBLAS_OP_T),
          (transposeB ? CUBLAS_OP_T : CUBLAS_OP_N),
          //(transposeA ? CUBLAS_OP_N : CUBLAS_OP_T),
          (transposeA ? CUBLAS_OP_T : CUBLAS_OP_N),
          //matB->shape[1],
          (transposeB ? matB->shape[0] : matB->shape[1]),
          //matA->shape[0],
          (transposeA ? matA->shape[1] : matA->shape[0]),
          //matB->shape[0],
          (transposeB ? matB->shape[1] : matB->shape[0]),
          &alpha,
          matB_data,
          matB->shape[1],
          //(transposeB ? matB->shape[0] : matB->shape[1]),
          //(transposeB ? matB->shape[1] : matB->shape[0]),
          matA_data,
          matA->shape[1],
          //(transposeA ? matA->shape[0] : matA->shape[1]),
          //(transposeA ? matA->shape[1] : matA->shape[0]),
          &beta,
          matC_data,
          //matB->shape[1]
          (transposeB ? matB->shape[0] : matB->shape[1])
          //matB->shape[1]
          //matB->shape[0] // == matC->shape[1]
  );
  /*
  cudaMemcpy(h_C, matC_data,sizeof(float) * matC->shape[0] * matC->shape[1], cudaMemcpyDeviceToHost);

  std::cout << "print C:" << std::endl;
  std::cout << matC->shape[0] << std::endl;
  std::cout << matC->shape[1] << std::endl;
  for (int i = 0; i < matC->shape[0]; ++i) {
    for (int j = 0; j < matC->shape[1]; ++j) {
      std::cout << h_C[i * matC->shape[1] + j] << '\t';
    }
    std::cout << std::endl;
  }
  */

  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int count = input->shape[0] * input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  matrix_relu_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, input_data, output_data);
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(in_grad->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == in_grad->shape[0] &&
         input->shape[1] == in_grad->shape[1]);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int count = input->shape[0] * input->shape[1];
  const float *input_data = (const float *)input->data;
  const float *in_grad_data = (const float *)in_grad->data;
  float *output_data = (float *)output->data;
  matrix_relu_gradient_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
      count, input_data, in_grad_data, output_data);
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  matrix_softmax_kernel<<<1, threads>>>(
      nrow, ncol, input_data, output_data);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
