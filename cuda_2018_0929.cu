/**************************************************
	> File Name:  cuda_2018_0929.cu
	> Author:     Leuckart
	> Time:       2018-09-29 01:18
**************************************************/

//#include "const_common.h"
//#include "conio.h"
#include <stdio.h>
#include <assert.h>

#define CUDA_CALL(x) {const cudaError_t a=(x);if(a!=cudaSuccess){printf("\nCUDA ERROR: %s (error_num: %d )\n",cudaGetErrorString(a),a);cudaDeviceReset();assert(0);}}
#define KERNEL_LOOP 65536

__constant__ static const unsigned int const_data_1=0x55555555;
__constant__ static const unsigned int const_data_2=0x77777777;
__constant__ static const unsigned int const_data_3=0x33333333;
__constant__ static const unsigned int const_data_4=0x11111111;

__global__ void const_test_gpu_literal(unsigned int * const data,const unsigned int num_elements)
{
	const unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<num_elements)
	{
		unsigned int d=0x55555555;
		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d^=0x55555555;
			d|=0x77777777;
			d&=0x33333333;
			d|=0x11111111;
		}
		data[tid]=d;
	}
}

__global__ void const_test_gpu_const(unsigned int * const data,const unsigned int num_elements)
{
	const unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<num_elements)
	{
		unsigned int d=const_data_1;
		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d^=const_data_1;
			d|=const_data_2;
			d&=const_data_3;
			d|=const_data_4;
		}
		data[tid]=d;
	}
}

__device__ static unsigned int data_1=0x55555555;
__device__ static unsigned int data_2=0x77777777;
__device__ static unsigned int data_3=0x33333333;
__device__ static unsigned int data_4=0x11111111;

__global__ void const_test_gpu_gmem(unsigned int * const data,const unsigned int num_elements)
{
	const unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<num_elements)
	{
		unsigned int d=0x55555555;
		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d^=data_1;
			d|=data_2;
			d&=data_3;
			d|=data_4;
		}
		data[tid]=d;
	}
}

__constant__ static const unsigned int const_data[4]={0x55555555,0x77777777,0x33333333,0x11111111};

__global__ void const_test_gpu_const_new(unsigned int * const data,const unsigned int num_elements)
{
	const unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<num_elements)
	{
		unsigned int d=const_data[0];
		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d^=const_data[0];
			d|=const_data[1];
			d&=const_data[2];
			d|=const_data[3];
		}
		data[tid]=d;
	}
}

__host__ void cuda_error_check(const char *prefix,const char *postfix)
{
	if(cudaPeekAtLastError()!=cudaSuccess)
	{
		printf("\n%s%s%s",prefix,cudaGetErrorString(cudaGetLastError()),postfix);
		cudaDeviceReset();
		//wait_exit();
		exit(1);
	}
}

__host__ void gpu_kernel()
{
	const unsigned int num_elements=128*1024;
	const unsigned int num_threads=256;
	const unsigned int num_blocks=(num_elements+(num_threads-1))/num_threads;
	const unsigned int num_bytes=num_elements*sizeof(unsigned int);
	int max_device_num;
	const int max_runs=6;

	CUDA_CALL(cudaGetDeviceCount(&max_device_num));

	for(int device_num=0;device_num<max_device_num;device_num++)
	{
		CUDA_CALL(cudaSetDevice(device_num));
		for(int num_test=0;num_test<max_runs;num_test++)
		{
			unsigned int *data_gpu;
			cudaEvent_t kernel_start1,kernel_stop1;
			cudaEvent_t kernel_start2,kernel_stop2;
			cudaEvent_t kernel_start3,kernel_stop3;
			cudaEvent_t kernel_start4,kernel_stop4;
			float delta_time1=0.0F,delta_time2=0.0F,delta_time3=0.0F,delta_time4=0.0F;
			struct cudaDeviceProp device_prop;
			char device_prefix[261];

			CUDA_CALL(cudaMalloc(&data_gpu,num_bytes));
			CUDA_CALL(cudaEventCreate(&kernel_start1));
			CUDA_CALL(cudaEventCreate(&kernel_start2));
			CUDA_CALL(cudaEventCreate(&kernel_start3));
			CUDA_CALL(cudaEventCreate(&kernel_start4));
			CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync));
			CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop2,cudaEventBlockingSync));
			CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop3,cudaEventBlockingSync));
			CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop4,cudaEventBlockingSync));

			//printf("\nLaunching %u blocks, %u threads.",num_blocks,num_threads);
			CUDA_CALL(cudaGetDeviceProperties(&device_prop,device_num));
			sprintf(device_prefix,"ID: %d %s: ",device_num,device_prop.name);

			/**********/

			//printf("\nLauching literal kernel warm-up");
			const_test_gpu_literal<<<num_blocks,num_threads>>>(data_gpu,num_elements);
			cuda_error_check("Error "," returned from literal startup kernel");

			//printf("\nLauning literal kernel");
			CUDA_CALL(cudaEventRecord(kernel_start1,0));
			const_test_gpu_literal<<<num_blocks,num_threads>>>(data_gpu,num_elements);
			cuda_error_check("Error "," returned from literal rumtime kernel");

			CUDA_CALL(cudaEventRecord(kernel_stop1,0));
			CUDA_CALL(cudaEventSynchronize(kernel_stop1));
			CUDA_CALL(cudaEventElapsedTime(&delta_time1,kernel_start1,kernel_stop1));
			//printf("\nLiteral Elapsed time: %.3fms",delta_time1);

			/**********/

			//printf("\nLaunching constant kernel warm-up");
			const_test_gpu_const<<<num_blocks,num_threads>>>(data_gpu,num_elements);
			cuda_error_check("Error "," returned from constant startup kernel");

			//printf("\nLaunching constant kernel");
			CUDA_CALL(cudaEventRecord(kernel_start2,0));
			const_test_gpu_const<<<num_blocks,num_threads>>>(data_gpu,num_elements);
			cuda_error_check("Error "," returned from constant runtime kernel");

			CUDA_CALL(cudaEventRecord(kernel_stop2,0));
			CUDA_CALL(cudaEventSynchronize(kernel_stop2));
			CUDA_CALL(cudaEventElapsedTime(&delta_time2,kernel_start2,kernel_stop2));
			//printf("\nConst Elapsed Time: %.3fms",delta_time2);

			/**********/

			//printf("\nLaunching gmem kernel warm-up");
			const_test_gpu_gmem<<<num_blocks,num_threads>>>(data_gpu,num_elements);
			cuda_error_check("Error "," returned from constant startup kernel");

			//printf("\nLaunching gmem kernel");
			CUDA_CALL(cudaEventRecord(kernel_start3,0));
			const_test_gpu_const<<<num_blocks,num_threads>>>(data_gpu,num_elements);
			cuda_error_check("Error "," returned from constant runtime kernel");

			CUDA_CALL(cudaEventRecord(kernel_stop3,0));
			CUDA_CALL(cudaEventSynchronize(kernel_stop3));
			CUDA_CALL(cudaEventElapsedTime(&delta_time3,kernel_start3,kernel_stop3));
			//printf("\nConst Elapsed Time: %.3fms",delta_time3);

			/**********/

			//printf("\nLaunching new_const kernel warm-up");
			const_test_gpu_const_new<<<num_blocks,num_threads>>>(data_gpu,num_elements);
			cuda_error_check("Error "," returned from constant startup kernel");

			//printf("\nLaunching new_const kernel");
			CUDA_CALL(cudaEventRecord(kernel_start4,0));
			const_test_gpu_const_new<<<num_blocks,num_threads>>>(data_gpu,num_elements);
			cuda_error_check("Error "," returned from constant runtime kernel");

			CUDA_CALL(cudaEventRecord(kernel_stop4,0));
			CUDA_CALL(cudaEventSynchronize(kernel_stop4));
			CUDA_CALL(cudaEventElapsedTime(&delta_time4,kernel_start4,kernel_stop4));
			//printf("\nConst Elapsed Time: %.3fms",delta_time4);

			/**********/

			printf("\n%s (Const= %.5fms . Literal= %.5fms . Gmem= %.5fms . NewConst= %.5fms)",device_prefix,delta_time1,delta_time2,delta_time3,delta_time4);
			/*
			if(delta_time1>delta_time2)
			{
				printf("\n%sConstant Version is faster by: %.2fms (Const= %.2fms vs. Literal= %.2fms)",device_prefix,delta_time1-delta_time2,delta_time1,delta_time2);
			}
			else
			{
				printf("\n%sLiteral Version is faster by: %.2fms (Const= %.2fms vs. Literal= %.2fms)",device_prefix,delta_time2-delta_time1,delta_time1,delta_time2);
			}
			*/

			CUDA_CALL(cudaEventDestroy(kernel_start1));
			CUDA_CALL(cudaEventDestroy(kernel_start2));
			CUDA_CALL(cudaEventDestroy(kernel_start3));
			CUDA_CALL(cudaEventDestroy(kernel_start4));
			CUDA_CALL(cudaEventDestroy(kernel_stop1));
			CUDA_CALL(cudaEventDestroy(kernel_stop2));
			CUDA_CALL(cudaEventDestroy(kernel_stop3));
			CUDA_CALL(cudaEventDestroy(kernel_stop4));
		}
		CUDA_CALL(cudaDeviceReset());
		printf("\n");
	}
}

int main()
{
	gpu_kernel();
	return 0;
}
