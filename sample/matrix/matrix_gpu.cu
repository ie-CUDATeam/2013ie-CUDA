#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <cutil_inline.h>

#define MATRIX_SIZE 1024/*行列１辺の数*/
#define BLOCK_SIZE 16

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC);

int main(int argc, char** argv){
unsigned int matrixSize = sizeof(unsigned int) * MATRIX_SIZE * MATRIX_SIZE;

  int* hMatrixA;
  int* hMatrixB;
  int* hMatrixC;
  hMatrixA = (int*)malloc(matrixSize);
  hMatrixB = (int*)malloc(matrixSize);

/* Initial value setting */
  unsigned int col_idx, row_idx;
  for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++){
      for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++){
          hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024*1024);
          hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024*1024);
      }
  }

/* Variable settings on the device side */
  int* dMatrixA;
  int* dMatrixB;
  int* dMatrixC;
 
/* Ensure the device memory area */
  cutilSafeCall(cudaMalloc((void**)&dMatrixA, matrixSize));
  cutilSafeCall(cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMalloc((void**)&dMatrixB, matrixSize));
  cutilSafeCall(cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMalloc((void**)&dMatrixC, matrixSize));

/*Setting the grid size and block size */
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(MATRIX_SIZE/BLOCK_SIZE, MATRIX_SIZE/BLOCK_SIZE);

/* Measurement create and start a timer */
  unsigned int timer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timer));
  CUT_SAFE_CALL( cutStartTimer( timer));

/* Starting the kernel */
  matrixMul<<<grid, block>>>(dMatrixA, dMatrixB, dMatrixC);
  cudaThreadSynchronize();

/* Memory transfer from the device side and secure area of the results */
  hMatrixC = (int*)malloc(matrixSize);
  cutilSafeCall(cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost));

/* Show time we came to stop the timer */
  CUT_SAFE_CALL( cutStopTimer( timer));
  printf("Processing time: %f (msec)\n", cutGetTimerValue( timer));
  CUT_SAFE_CALL( cutDeleteTimer( timer));

/* The opening of the host device memory */
  free(hMatrixA);
  free(hMatrixB);
  free(hMatrixC);
  cutilSafeCall(cudaFree(dMatrixA));
  cutilSafeCall(cudaFree(dMatrixB));
  cutilSafeCall(cudaFree(dMatrixC));
 
/* Processing end　*/
  cudaThreadExit();
  cutilExit(argc, argv);
 }
 
__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC){
  unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int scan_idx;
  unsigned int target = 0;

/* Line the calculation of the matrix */
 for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
   target +=inMatrixA[col_idx * MATRIX_SIZE + scan_idx] * inMatrixB[scan_idx * MATRIX_SIZE + row_idx];
   __syncthreads();
 }
 inMatrixC[col_idx * MATRIX_SIZE + row_idx] = target;
}
