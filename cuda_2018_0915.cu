/**************************************************
	> File Name:  cuda_2018_0915.cu
	> Author:     Leuckart
	> Time:       2018-09-15 01:53
**************************************************/

#include<stdio.h>
#include<stdlib.h>

#define KERNEL_LOOP 10240
unsigned int cpu_data[KERNEL_LOOP];
__device__ unsigned int * packed_array;//[KERNEL_LOOP];

__global__ void test_gpu_register(unsigned int * const data,const unsigned int num_elements)
{
    const unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x;
    if(tid<num_elements)
    {
        unsigned int d_tmp=0;
        for(int i=0;i<KERNEL_LOOP;i++)
        {
            d_tmp|=(packed_array[i]<<i);
        }
        data[tid]=d_tmp;
    }
}

__device__ static unsigned int d_tmp=0;
__global__ void test_gpu_gmem(unsigned int * const data,const unsigned int num_elements)
{
    const unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x;
    if(tid<num_elements)
    {
        for(int i=0;i<KERNEL_LOOP;i++)
        {
            d_tmp|=(packed_array[i]<<i);
        }
        data[tid]=d_tmp;
    }
}


int main()
{
    const unsigned int num_blocks=2;
    const unsigned int num_threads=64;

    unsigned int *gpu_data;
    cudaMalloc((void **)&gpu_data,KERNEL_LOOP);
    cudaMalloc((void **)&packed_array,KERNEL_LOOP);

    //test_gpu_register<<<num_blocks,num_threads>>>(gpu_data,KERNEL_LOOP);
    test_gpu_gmem<<<num_blocks,num_threads>>>(gpu_data,KERNEL_LOOP);

    cudaMemcpy(cpu_data,gpu_data,KERNEL_LOOP,cudaMemcpyDeviceToHost);
    cudaFree(gpu_data);
    
    return 0;
}
