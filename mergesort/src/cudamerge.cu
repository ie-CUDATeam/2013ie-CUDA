/*
* cudamerge.cu
* Noble713
* 3 Feb 2014
* CUDA parallel thread mergesort
*
* Original Algorithm from Jim Kukunas and James Devine
* http://jamesdevine.info/wp-content/uploads/2009/05/writeup.pdf
*/

#include <helper_cuda.h>
#include <helper_timer.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
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
	filename << "input_N" << num << "_" << dataset;
	std::cout << "Read-In From: " << filename.str().c_str() <<std::endl;
	std::ifstream input(filename.str().c_str());

	std::string skipline;
	getline(input, skipline);

	if(input.is_open())
	{
		while(input >> a)
		{
			values[i] = a;
			//std::cout << values[i] << " ";
			i++;
		}
	}
	else std::cout << "Unable to open file";
	//std::cout << std::endl;
}

int main ( int argc, char** argv )
{

	if( checkCmdLineFlag( argc,  (const char**)argv, "device" ) ) //see if a GPU is already set, if not set the CUDA device to the GPU with highest Gigaflops
		gpuDeviceInit( argc);
	else
		cudaSetDevice( gpuGetMaxGflopsDeviceId());

	clock_t start;
	double elapsed;
	int i;

	int num = 100;
	int dataset=0;

	std::string userinput = " ";
	int usernumber = 0;
	std::cout << "Enter Dataset size";
	getline(std::cin, userinput);
	std::stringstream userstream(userinput);
	userstream >> usernumber;
	std::cout << "Will now benchmark datasets input_N" << usernumber <<std::endl;

	num = usernumber;
	thrust::host_vector<int> values(num,0);

	thrust::device_vector<int> dvalues = values;
	thrust::device_vector<int> results = values;


	std::ofstream out("log.txt"); //creates an output object, essentially a text log
	out.close();

	while(dataset<10)
	{
		read_benchmarks(num, dataset, values);


		//checkCudaErrors( cudaMalloc(( void**)& dvalues, sizeof( int)* num));//allocates device memory, device vector = host vector, automatically calls cudamemcpy
		dvalues = values;

		//checkCudaErrors( cudaMalloc(( void**)& results, sizeof( int)* num));

		out.open("log.txt", std::ios::app);
		out << "\n------- CUDA Merge sorting, unsorted array, benchmark N" << num << "_" << dataset << "-------\n";
		for(i=0; i<num; i++)
			out << " " << values[i]; //* output unsorted values to the log file

		start = clock(); //start the clock and call the CUDA kernel
		MergeSort <<<1, num, sizeof( int)* num*2>>>(convertoKernel(dvalues), convertoKernel(results), num);
		elapsed = ( clock() - start) / CLOCKS_PER_SEC;

		std::cout << std::endl << cudaGetErrorString(cudaGetLastError());

		values=dvalues; //thrust::copy(dvalues.begin(), dvalues.end(), values.begin());

		out << "\n------- Merge-sorted elements-------\n"; //output sorted array to log file
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
		out << "\nTest " << (passed? "PASSED" : "FAILED") << "\n"; //print to screen pass/fail status of sorting algorithm and elapsed time
		out << "Elapsed time: " << elapsed << "\n";
		out.close();

		dataset +=1;
		//cudaDeviceReset();
	}
}
