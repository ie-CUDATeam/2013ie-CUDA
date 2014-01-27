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
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "merge_kernel.cu"


void read_benchmarks(int num, int dataset, thrust::host_vector<int>& values) //reads the inputs for the current datasize and set #
{
	int i=0;
	int a;

	std::stringstream filename;
	filename << "input_N" << (char)num << "_" << (char)dataset;
	std::ifstream input(filename.str().c_str());

	while(input >> a)
	{
		values[i] = a;
		i++;
	}

}

void test_benchmarks(int num, thrust::host_vector<int>& values, int dataset, thrust::device_vector<int> dvalues, thrust::device_vector<int> results, std::ofstream& out)
{
	clock_t start;
	double elapsed;
	int i;

	out.open("log.txt", std::ios::app);
	out << "\n\t------- CUDA Merge sorting, unsorted array, benchmark N" << num << "_" << dataset << "-------\n\n";
	for(i=0; i<num; i++)
		out << " " << values[i]; //* output unsorted values to the log file

	start = clock(); //start the clock and call the CUDA kernel
	MergeSort <<<1, num, sizeof( int)* num*2>>>(convertoKernel(dvalues), convertoKernel(results), num);
	elapsed = ( clock() - start) / CLOCKS_PER_SEC;

	//check for errors
	getLastCudaError("Kernel execution failed");

	thrust::copy(dvalues.begin(), dvalues.end(), values.begin());
	//values = dvalues;

	checkCudaErrors( cudaFree(&dvalues));
	//checkCudaErrors( cudaMemcpy( values, results, sizeof( int)* num, cudaMemcpyDeviceToHost));

	checkCudaErrors( cudaFree(&results));

	out << "\n\t------- Merge-sorted elements-------\n\n"; //output sorted array to log file
	for(i=0; i<num; i++)
		out << " " << values[i];

	bool passed = true; //check to see if array is sorted properly
	for ( int i=1 ; i<num; i++)
	{
		if (values[i-1] > values[i])
		{
			passed = false;
		}
	}
	out << "Test " << (passed? "PASSED" : "FAILED") << "\n"; //print to screen pass/fail status of sorting algorithm and elapsed time
	out << "Elapsed time: " << elapsed << "\n";
	out.close();
	cudaThreadExit();
}


int main ( int argc, char** argv )
{

	if( checkCmdLineFlag( argc,  (const char**)argv, "device" ) ) //see if a GPU is already set, if not set the CUDA device to the GPU with highest Gigaflops
		gpuDeviceInit( argc);
	else
		cudaSetDevice( gpuGetMaxGflopsDeviceId());

	int num = 100;
	thrust::host_vector<int> values(100,0);
	int dataset=0;
	int e=2; //e is exponent counter for num
	thrust::device_vector<int> dvalues = values;
	thrust::device_vector<int> results = values;

	int* raw_dval_ptr = thrust::raw_pointer_cast( &dvalues[0]); //pointers for the device memory values and results
	int* raw_res_ptr = thrust::raw_pointer_cast( &results[0]);  //to be wrapped in thrust device ptrs

	cudaMalloc(( void**)& raw_dval_ptr, sizeof( int)* num);
	cudaMalloc(( void**)& raw_res_ptr, sizeof( int)* num);

	thrust::device_ptr<int> dvalues_ptr(raw_dval_ptr);
	thrust::device_ptr<int> results_ptr(raw_res_ptr);


	std::ofstream out("log.txt"); //creates an output object, essentially a text log
	out.close();

	while(num<1000000 && dataset<10)
	{
		while(dataset<10)
		{
			read_benchmarks(num, dataset, values);

			checkCudaErrors( cudaMalloc(( void**)& dvalues, sizeof( int)* num));//allocates device memory, device vector = host vector, automatically calls cudamemcpy
			dvalues = values;

			//checkCudaErrors( cudaMemcpy( dvalues, values, sizeof( int)* num, cudaMemcpyHostToDevice));
			checkCudaErrors( cudaMalloc(( void**)& results, sizeof( int)* num)) ;
			//checkCudaErrors( cudaMemcpy( results, values, sizeof( int)* num, cudaMemcpyHostToDevice)); */


			test_benchmarks(num, values, dataset, dvalues, results, out);
			dataset +=1;
		}
		e +=1;
		num = pow( 10, e);
		values.resize(num);
		dataset=0;
	}
}
