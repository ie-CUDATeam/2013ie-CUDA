/*
 *Combined from merge sort and CUDA Sample Quicksort
 *
 *Compiles fine but is otherwise untested.
 *
 */
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cstdio>

using namespace std;

#define THREADS 16 // 2^9
#define BLOCKS 8 // 2^15
#define NUM_VALS THREADS*BLOCKS

////////////////////////////////////////////////////////////////////////////////
//Merge the chunks
////////////////////////////////////////////////////////////////////////////////
__device__ void cdp_merge(unsigned int *data, int l, int m, int h)
{
  int arr1[20], arr2[20];
  int n1,n2,i,j,k;
  n1=m-l+1;
  n2=h-m;

  for(i=0; i<n1; i++)
    arr1[i]=data[l+i];
  for(j=0; j<n2; j++)
    arr2[j]=arr1[m+j+1];

  arr1[i]=9999;
  arr2[j]=9999;

  i=0;
  j=0;
  for(k=l; k<=h; k++) { //process of combining two sorted arrays
    if(arr1[i]<=arr2[j])
      data[k]=arr1[i++];
    else
      data[k]=arr2[j++];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic mergesort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_mergesort(unsigned int *data, int left, int right, int depth)
{
  int mid;

  if(left<right) {
    mid=(left+right)/2;
		cudaStream_t s;
		cdp_mergesort<<< 1, 1, 0, s >>>(data, left, mid, depth+1);
    cdp_mergesort<<< 1, 1, 0, s1 >>>(data, mid, right, depth+1);
    cdp_merge(data,left,mid,right);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Call the mergesort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_msort(unsigned int *data, unsigned int num_items)
{
  // Launch on device
  int left = 0;
  int right = num_items-1;
  cout << "Launching kernel on the GPU" << endl;
  cdp_mergesort<<< 1, 1 >>>(data, left, right, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(unsigned int *dst, unsigned int num_items)
{
  // Fixed seed for illustration
  srand(2047);

  // Fill dst with random values
  for (unsigned i = 0 ; i < num_items ; i++)
    dst[i] = rand() % num_items ;
}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(int n, unsigned int *results_d)
{
  unsigned int *results_h = new unsigned[n];
  cudaMemcpy(results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost);

  for (int i = 1 ; i < n ; ++i)
    if (results_h[i-1] > results_h[i])
      {
        cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << endl;
        exit(EXIT_FAILURE);
      }

  cout << "OK" << endl;
  delete[] results_h;
}

////////////////////////////////////////////////////////////////////////////////
// Random array.
////////////////////////////////////////////////////////////////////////////////
int randomInt() {
  return (int)(((float)rand()/(float)RAND_MAX) * 1000000);
}

void arrayPrint(int *arr, int length) {
  int i;
  for (i = 0; i < length; ++i) {
    printf("%d ",  arr[i]);
  }
  printf("\n");
}


////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  // Create data
  unsigned int *h_data = 0;
  unsigned int *d_data = 0;


  // Allocate CPU memory and initialize data.
  cout << "Initializing data:" << endl;
  h_data =(unsigned int *)malloc( NUM_VALS * sizeof(unsigned int));
  initialize_data(h_data, NUM_VALS);

  // Allocate GPU memory.
  cudaMalloc((void **)&d_data, NUM_VALS * sizeof(unsigned int));
  cudaMemcpy(d_data, h_data, NUM_VALS * sizeof(unsigned int), cudaMemcpyHostToDevice);

  // Execute
  cout << "Running mergesort on " << NUM_VALS << " elements" << endl;
  run_msort(d_data, NUM_VALS);

  // Check result
  cout << "Validating results: ";
  //check_results(NUM_VALS, d_data);
  cudaFree(d_data);
  cudaFree(h_data);
}

