/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 */
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
 
/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 512 // 2^9
#define BLOCKS 32768 // 2^15
#define NUM_VALS THREADS*BLOCKS
 
void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}
 
float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}
 
void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}
 
void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}
 
__global__ void bitonic_sort_step(float *dev_values, int idx, int block)
{
  unsigned int i, e; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  e = i ^ idx;
 
  /* The threads with the lowest ids sort the array. */
  if ( i < e ) {
    if (( i & block)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[e]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[e];
        dev_values[e] = temp;
      }
    }
    if (( i & block)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[e]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[e];
        dev_values[e] = temp;
      }
    }
  }
}
 
/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);
 
  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
 
  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
 
  int idx, block;
  /* Major step */
  for (block = 2; block <= NUM_VALS; block <<= 1) {
    /* Minor step */
    for (idx=block>>1; idx>0; idx=idx>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, idx, block);
    }
  }
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}
 
int main(void)
{
  clock_t start, stop;
 
  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);
 
  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();
 
  print_elapsed(start, stop);
}
