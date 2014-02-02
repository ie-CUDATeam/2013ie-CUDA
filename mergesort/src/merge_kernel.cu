/*
∗ mergekernel.cu
∗
∗
∗
*CUDA parallel thread mergesort kernel
∗
∗ Jim Kukunas and James Devine
∗ http://jamesdevine.info/wp-content/uploads/2009/05/writeup.pdf
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef _MERGE_KERNEL_CU_
#define _MERGE_KERNEL_CU_

//template structure to past thrust vector to kernel
template <typename T>
struct KernelArray
{
	T* _array;
	int _size;
};
//function to convert device vector to structure
template <typename T>
KernelArray<T> convertoKernel( thrust::device_vector<T> &dVec)
{
	KernelArray<T> kArray;
	kArray._array = thrust::raw_pointer_cast( &dVec[0] );
	kArray._size  = ( int ) dVec.size();

	return kArray;
}

__device__ inline void Merge (int* dvalues, KernelArray<int> results, int l, int r, int u)
	{
	int i, j, k;
		i=l ; j=r ; k=l ;
		while ( i <r && j <u ) {
			if ( dvalues[i]<= dvalues[j] ) { results._array[k]= dvalues[i] ; i ++;}
			else{ results._array[k]= dvalues[j]; j ++;}
			k++;
		}
		while ( i<r ) {
			results._array[k]= dvalues[i] ; i ++; k++;
		}

		while ( j<u ) {
			results._array[k]= dvalues[j] ; j ++; k++;
		}
		for ( k=l; k<u; k++) {
			dvalues[k]= results._array[k];
		}
}


__global__ static void
__launch_bounds__(1024)
MergeSort (KernelArray<int> dvalues, KernelArray<int> results, int num)
{
	//extern __shared__ thrust::device_vector<int> shared;
	extern __shared__ int shared[];

	const unsigned int tid = threadIdx.x;
	int i, k, u; //k=window size, i = left, u = right

	// Copy input to shared mem
	shared[tid] = dvalues._array[tid];
	//thrust::copy(dvalues.begin(), dvalues.end(), shared.begin());

	__syncthreads();

	k = 1;

	while ( k <num)
	{
		i = 0;
		while ( i+k <= num)
		{
			u = i+k*2;
			if ( u> num)
			{
				u = num+1;
			}

			Merge( shared, results, i, i+k, u);
			i = i+k*2 ;
		}
		k = k*2;
		__syncthreads();
	}
	dvalues._array[tid] = shared[tid];
}

#endif
