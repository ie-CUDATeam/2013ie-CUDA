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

#ifndef _MERGE_KERNEL_CU_
#define _MERGE_KERNEL_CU_

#define NUM 64

__device__ inline void Merge (int* values, int* results, int l, int r, int u)
	{
	int i, j, k;
		i=l ; j=r ; k=l ;
		while ( i <r && j <u ) {
			if ( values[i]<= values[j] ) { results[k]= values[i] ; i ++;}
			else{ results[k]= values[j]; j ++;}
			k++;
		}
		while ( i<r ) {
			results[k]= values[i] ; i ++; k++;
		}

		while ( j<u ) {
			results[k]= values[j] ; j ++; k++;
		}
		for ( k=l; k<u; k++) {
			values[k]= results[k];
		}
}


__global__ static void MergeSort ( int* values, int* results)
{
	extern __shared__ int shared[ ];

	const unsigned int tid = threadIdx.x;
	int i, k, u;

	// Copy input to shared mem
	shared [tid] = values[tid] ;

	__syncthreads();

	k = 1;

	while ( k <NUM)
	{
		i = 1;
		while ( i+k <= NUM)
		{
			u = i+k*2;
			if ( u> NUM)
			{
				u = NUM+1;
			}

			Merge( shared, results, i, i+k, u);
			i = i+k*2 ;
		}
		k = k*2;
		__syncthreads();
	}
	values[tid] = shared[tid] ;
}

#endif

