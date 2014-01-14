/*
*Combined from merge sort and CUDA Sample Quicksort
*
*Compiles fine but is otherwise untested.
*
*/
#include <iostream>
#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>

#define MAX_DEPTH       16
#define INSERTION_SORT  32

////////////////////////////////////////////////////////////////////////////////
//Merge the chunks
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_merge(unsigned int *data, int l, int m, int h)
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

		// Divide the array into left and right pieces and launch new blocks
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		cdp_mergesort<<< 1, 1, 0, s >>>(data, left, mid, depth+1);
		cudaStreamDestroy(s);
		
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_mergesort<<< 1, 1, 0, s1 >>>(data, mid, right, depth+1);
        cudaStreamDestroy(s1);		
		

		// Sort and Combine the chunks
		cudaStream_t s2;
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		cdp_merge<<< 1, 1, 0, s2 >>>(data,left,mid,right);
        cudaStreamDestroy(s2);
	}
	
	
}

////////////////////////////////////////////////////////////////////////////////
// Call the mergesort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_msort(unsigned int *data, unsigned int num_items)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    int left = 0;
    int right = num_items-1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_mergesort<<< 1, 1 >>>(data, left, right, 0);
    checkCudaErrors(cudaDeviceSynchronize());
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
    checkCudaErrors(cudaMemcpy(results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int i = 1 ; i < n ; ++i)
        if (results_h[i-1] > results_h[i])
        {
            std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
            exit(EXIT_FAILURE);
        }

    std::cout << "OK" << std::endl;
    delete[] results_h;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int num_items = 128;
    bool verbose = false;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h"))
    {
        std::cerr << "Usage: " << argv[0] << " num_items=<num_items>\t where num_items is the number of items to sort" << std::endl;
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "v"))
    {
        verbose = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "num_items"))
    {
        num_items = getCmdLineArgumentInt(argc, (const char **)argv, "num_items");

        if (num_items < 1)
        {
            std::cerr << "ERROR: num_items has to be greater than 1" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Get device properties
    int device_count = 0, device = -1;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    for (int i = 0 ; i < device_count ; ++i)
    {
        cudaDeviceProp properties;
        checkCudaErrors(cudaGetDeviceProperties(&properties, i));

        if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
        {
            device = i;
            std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
            break;
        }

        std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
    }

    if (device == -1)
    {
        std::cerr << "cdpSimpleQuicksort requires GPU devices with compute SM 3.5 or higher.  Exiting..." << std::endl;
        exit(EXIT_SUCCESS);
    }

    cudaSetDevice(device);

    // Create input data
    unsigned int *h_data = 0;
    unsigned int *d_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
    h_data =(unsigned int *)malloc(num_items*sizeof(unsigned int));
    initialize_data(h_data, num_items);

    if (verbose)
    {
        for (int i=0 ; i<num_items ; i++)
            std::cout << "Data [" << i << "]: " << h_data[i] << std::endl;
    }

    // Allocate GPU memory.
    checkCudaErrors(cudaMalloc((void **)&d_data, num_items * sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Execute
    std::cout << "Running mergesort on " << num_items << " elements" << std::endl;
    run_msort(d_data, num_items);

    // Check result
    std::cout << "Validating results: ";
    check_results(num_items, d_data);

    free(h_data);
    checkCudaErrors(cudaFree(d_data));
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}

