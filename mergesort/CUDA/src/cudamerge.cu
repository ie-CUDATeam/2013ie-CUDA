/*
* cudamerge.cu
*
* CUDA parallel thread mergesort
*
∗ Jim Kukunas and James Devine
∗ http://jamesdevine.info/wp-content/uploads/2009/05/writeup.pdf
*/

#include <helper_cuda.h>
#include <helper_timer.h>

#include "merge_kernel.cu"

int main ( int argc, char** argv )
{
	if( checkCmdLineFlag( argc,  (const char**)argv, "device" ) )
		gpuDeviceInit( argc);
	else
		cudaSetDevice( gpuGetMaxGflopsDeviceId());

	int values[NUM];

	/* initialize a random data set */
	for( int i=0; i<NUM; i++)
	{
		values[i] = rand();
	}

	int* dvalues,
	   * results;

	checkCudaErrors( cudaMalloc(( void**)& dvalues, sizeof( int)* NUM));
	checkCudaErrors( cudaMemcpy( dvalues, values, sizeof( int)* NUM, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMalloc(( void**)& results, sizeof( int)* NUM)) ;
	checkCudaErrors( cudaMemcpy( results, values, sizeof( int)* NUM, cudaMemcpyHostToDevice));

	MergeSort <<<1, NUM, sizeof( int)* NUM*2>>>(dvalues, results);

	// c h e c k f o r any e r r o r s

	getLastCudaError("Kernel execution failed");

	checkCudaErrors( cudaFree(dvalues));
	checkCudaErrors( cudaMemcpy( values, results, sizeof( int)* NUM, cudaMemcpyDeviceToHost));
	checkCudaErrors( cudaFree( results));

	bool passed = true;
	for ( int i=1 ; i<NUM; i++)
	{
		if (values[i-1] > values[i])
		{
			passed = false ;
		}
	}
	printf( "Test %s\n", passed? "PASSED" : "FAILED");
	cudaThreadExit();
}
