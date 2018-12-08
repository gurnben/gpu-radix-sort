#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>

#include "sort.h"
#include "utils.h"

double test_cpu_vs_gpu(unsigned int* h_in, unsigned int num_elems)
{
    std::clock_t start;

    unsigned int* h_out_cpu = new unsigned int[num_elems];
    unsigned int* h_out_gpu = new unsigned int[num_elems];

    unsigned int* d_in;
    unsigned int* d_out;
    checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * num_elems));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * num_elems));
    checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice));
    start = std::clock();
    radix_sort(d_out, d_in, num_elems);
    double gpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    checkCudaErrors(cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_in));

    delete[] h_out_gpu;
    delete[] h_out_cpu;

    return gpu_duration;
}

/*printUsage
*
* Prints the usage information for this application.
*/
void printUsage()
{
    printf("This application takes as input a lower and upper bound for\n");
    printf("data sizes.  The input lower and upper bounds are taken in as\n");
    printf("powers of 2.  For example, input 24 as a lower bound will sort\n");
    printf("2^24 elements.\n");
    printf("\nusage: radix_sort <lower bound> <upper bound>\n");
    printf("\t<lower_bound> will be treated as a power of 2 and is inclusive\n");
    printf("\t<upper_bound> will be treated as a power of 2 and is inclusive\n");
    printf("Examples:\n");
    printf("\t./hybrid_sort 24 26\n");
}

/*parseCommandArgs
*
* This function processes the command line arguments given to the program.
*
* the proper use is:
*   ./hybrid_sort <lower_bound> <upper_bound>
*
* @params:
*   argc        - the number of arguments in argv
*   argv        - the arguments to the utility
*   lower_bound - a pointer to a lower_bound variable
*   upper_bound - a pointer to an upper_bound variable
*/
void parseCommandArgs(int argc, char * argv[], int * lower_bound,
                      int * upper_bound) {
    if (argc < 3) {
      printUsage();
      //exit because the input was incorrect
      exit(EXIT_FAILURE);
    }
    else {
      (*lower_bound) = atoi(argv[argc - 2]);
      (*upper_bound) = atoi(argv[argc - 1]);
    }
}

int main(int argc, char * argv[])
{
    // Set up clock for timing comparisons
    srand(1);

    int lower_bound = 0, upper_bound = 0;

    parseCommandArgs(argc, argv, &lower_bound, &upper_bound);

    for (int i = lower_bound; i <= upper_bound; ++i)
    {
        unsigned int num_elems = (1 << i);
        unsigned int* h_in = new unsigned int[num_elems];
        unsigned int* h_in_rand = new unsigned int[num_elems];

        for (unsigned int j = 0; j < num_elems; j++)
        {
            h_in[j] = (num_elems - 1) - j;
            h_in_rand[j] = rand() % num_elems;
        }
        double time = 0;
        for (unsigned int j = 0; j < 2; ++j) {
            time = test_cpu_vs_gpu(h_in_rand, num_elems);
        }

        printf("Four-way Radix Sort took %f milliseconds to sort %e (2^%d) numbers.\n", time, pow(2, i), i);

        delete[] h_in;
        delete[] h_in_rand;
    }
}
