/*
* mergesort.c
*
* Created on: Nov 23, 2013
* Author: Noble713
* renamed cpumerge.cpp, updated 3 Feb 2014
*/

//#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <time.h>
#include <vector>


void mergesort (std::vector<int>& values, std::vector<int>& results, int num)
{
    int right, rend;
    int i,j,k,m;
    int left =0;

    for (k=1; k < num; k *= 2 ) {
        for (left=0; left+k < num; left += k*2 ) {
            right = left + k;
            rend = right + k;
            if (rend > num) rend = num;
            m = left; i = left; j = right;
            while (i < right && j < rend) {
                if (values[i] <= values[j]) {
                    results[m] = values[i]; i++;
                } else {
                    results[m] = values[j]; j++;
                }
                m++;
            }
            while (i < right) {
                results[m]=values[i];
                i++; m++;
            }
            while (j < rend) {
                results[m]=values[j];
                j++; m++;
            }
            for (m=left; m < rend; m++) {
                values[m] = results[m];
            }
        }
    }
}

void read_benchmarks(int num, int dataset, std::vector<int>& values) //reads the inputs for the current datasize and set #
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
			std::cout << values[i] << " ";
			i++;
		}
	}
	else std::cout << "Unable to open file";
	std::cout << std::endl;
}

int main()
{
	clock_t starttime, stoptime;
	double elapsed;
	int i;

	int num = 100;
	int dataset=0;
	int usernumber = 0;

	std::string userinput = " ";
	std::cout << "Enter Dataset size";
	getline(std::cin, userinput);
	std::stringstream userstream(userinput);
	userstream >> usernumber;
	std::cout << "Will now benchmark datasets input_N" << usernumber <<std::endl;

	num = usernumber;
	std::vector<int> values(num,0);
	std::vector<int> results(values);

	std::ofstream out("cpulog.txt"); //creates an output object, essentially a text log
	out.close();

	while(dataset<10)
	{
		read_benchmarks(num, dataset, values);

		out.open("cpulog.txt", std::ios::app);
		out << "\n------- CPU Merge sorting, unsorted array, benchmark N" << num << "_" << dataset << "-------\n";
		for(i=0; i<num; i++)
			out << " " << values[i]; //* output unsorted values to the log file

		starttime = clock(); //start the clock and call the CUDA kernel
		mergesort(values, results, num);
		stoptime = clock();
		elapsed = (double)( stoptime - starttime) / CLOCKS_PER_SEC;

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
	}
}
