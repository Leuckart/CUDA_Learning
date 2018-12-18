/**************************************************
	> File Name:  invert.cpp
	> Author:     Leuckart
	> Time:       2018-12-09 19:13
**************************************************/

#include "inverse.h"

__global__ void Row_Kernel_Function(double *ori, double *inv, int now)
{
	const unsigned int _idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int _idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * _idy) + _idx;

	const unsigned int idx = thread_idx / SIZE;
	const unsigned int idy = thread_idx % SIZE;

	if ((idx < SIZE) && (idy < SIZE))
	{
		double ii = Element(ori, now, now);
		double temp = 0.0;

		if (idx != now)
		{
			temp = Element(ori, idx, now) / ii;
			Element(ori, idx, idy) -= Element(ori, now, idy) * temp;
			Element(inv, idx, idy) -= Element(inv, now, idy) * temp;
		}
	}
	__syncthreads();
}

__global__ void Row_Kernel_Normalize(double *ori, double *inv)
{
	const unsigned int _idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int _idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * _idy) + _idx;

	const unsigned int idx = thread_idx / SIZE;
	const unsigned int idy = thread_idx % SIZE;

	if ((idx < SIZE) && (idy < SIZE))
	{
		double temp = 1. / Element(ori, idx, idx);
		Element(ori, idx, idy) *= temp;
		Element(inv, idx, idy) *= temp;
	}
	__syncthreads();
}

void Inverse_Matrix_Kernel_Handle(double *ori, double *inv, dim3 Blocks_Per_Grid, dim3 Threads_Per_Block)
{
	for (int i = 0; i < SIZE; i++)
	{
		Row_Kernel_Function<<<Blocks_Per_Grid, Threads_Per_Block>>>(ori, inv, i);
		cudaThreadSynchronize();
	}
	Row_Kernel_Normalize<<<Blocks_Per_Grid, Threads_Per_Block>>>(ori, inv);
	cudaThreadSynchronize();
}

int main()
{
	unsigned int Byte_Size = SIZE * SIZE * sizeof(double);
	double *Matrix_Ori = (double *)malloc(Byte_Size);
	Initialize_Matrix(Matrix_Ori);
	double *Matrix_Inv = (double *)malloc(Byte_Size);

	/* Initial Threads Blocks Begin */
	int thread_xdim = 32;
	int thread_ydim = 32;
	const dim3 Threads_Per_Block(thread_xdim, thread_ydim);
	const dim3 Blocks_Per_Grid(int((SIZE - 1) / Threads_Per_Block.x) + 1, int((SIZE - 1) / Threads_Per_Block.y) + 1);
	/* Initial Threads Blocks End */

	/* Initial Memory Begin */
	double *Matrix_GPU;
	double *Matrix_Inv_GPU;
	double *ident = (double *)malloc(Byte_Size);
	for (int i = 0; i < SIZE; i++)
	{
		Element(ident, i, i) = 1;
	}
	Cuda_Call(cudaMalloc((void **)&Matrix_GPU, Byte_Size));
	Cuda_Call(cudaMalloc((void **)&Matrix_Inv_GPU, Byte_Size));
	/* Initial Memory Begin */

	/* Test On Every Device Begin */
	int Device_All;
	Cuda_Call(cudaGetDeviceCount(&Device_All));
	/* Test On Every Device End */

	for (int device_number = 0; device_number < Device_All; ++device_number)
	{
		/* Set Device Parameters Begin */
		Cuda_Call(cudaSetDevice(device_number));
		struct cudaDeviceProp device_prop;
		char device_prefix[100];
		Cuda_Call(cudaGetDeviceProperties(&device_prop, device_number));
		sprintf(device_prefix, "ID: %d %s: ", device_number, device_prop.name);
		/* Set Device Parameters End */

		/* Initial Time Block Begin */
		cudaEvent_t kernel_start, kernel_stop;
		float delta_time = 0.;
		Cuda_Call(cudaEventCreate(&kernel_start));
		Cuda_Call(cudaEventCreateWithFlags(&kernel_stop, cudaEventBlockingSync));
		Cuda_Call(cudaEventRecord(kernel_start, 0));
		/* Initial Time Block End */

		/* Kernel Function Execute Begin */
		Cuda_Call(cudaMemcpy(Matrix_GPU, Matrix_Ori, Byte_Size, cudaMemcpyHostToDevice));
		Cuda_Call(cudaMemcpy(Matrix_Inv_GPU, ident, Byte_Size, cudaMemcpyHostToDevice));
		Inverse_Matrix_Kernel_Handle(Matrix_GPU, Matrix_Inv_GPU, Blocks_Per_Grid, Threads_Per_Block);
		Cuda_Call(cudaMemcpy(Matrix_Inv, Matrix_Inv_GPU, Byte_Size, cudaMemcpyDeviceToHost));

		/* Kernel Function Execute End */

		/* Time Clock Begin */
		Cuda_Call(cudaEventRecord(kernel_stop, 0));
		Cuda_Call(cudaEventSynchronize(kernel_stop));
		Cuda_Call(cudaEventElapsedTime(&delta_time, kernel_start, kernel_stop));
		printf("%s %.5fms\n", device_prefix, delta_time);
		Cuda_Call(cudaEventDestroy(kernel_start));
		Cuda_Call(cudaEventDestroy(kernel_stop));
		/* Time Clock End */
	}

	/* Copy GPU To CPU Begin */
	//Cuda_Call(cudaMemcpy(Matrix_Ori, Matrix_GPU, Byte_Size, cudaMemcpyDeviceToHost));
	//Show_Matrix(Matrix_Ori,"...");
	/* Copy GPU To CPU End */

	//double *Matrix_Res = (double *)malloc(Byte_Size);
	//Matrix_Mult(Matrix_Ori, Matrix_Inv, Matrix_Res);
	////Matrix_Mult(Matrix_Ori,ident,Matrix_Res);
	//Show_Matrix(Matrix_Res, "Mult Matrix :");

	/* Free Memory Begin */
	Cuda_Call(cudaFree(Matrix_GPU));
	free(Matrix_Ori);
	//free(Matrix_Res);
	/* Free Memory End */
	return 0;
}
