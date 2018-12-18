/**************************************************
	> File Name:  invert.cpp
	> Author:     Leuckart
	> Time:       2018-12-09 19:13
**************************************************/

#include "inverse.h"

double Get_Det(double *mat, int n)
{
	if (n == 1)
	{
		return mat[0];
	}
	double ans = 0;
	double *cof = (double *)malloc((n - 1) * (n - 1) * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			for (int k = 0; k < n - 1; k++)
			{
				Point(cof, j, k, n - 1) = Point(mat, j + 1, k < i ? k : k + 1, n);
			}
		}
		double t = Get_Det(cof, n - 1);
		ans += mat[i] * t * (i % 2 == 0 ? 1 : -1);
	}
	free(cof);
	return ans;
}

__device__ double Loop(double *mat,int n)
{
	if (n == 1)
	{
		return mat[0];
	}
	double ans = 0;
	double *cof = (double *)malloc((n - 1) * (n - 1) * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			for (int k = 0; k < n - 1; k++)
			{
				Point(cof, j, k, n - 1) = Point(mat, j + 1, k < i ? k : k + 1, n);
			}
		}
		double t = Loop(cof, n - 1);
		ans += mat[i] * t * (i % 2 == 0 ? 1 : -1);
	}
	free(cof);
	return ans;
}

__global__ void Kernel_Function(double *ori,double *inv,double det)
{
	const unsigned int _idx=(blockIdx.x*blockDim.x)+threadIdx.x;
	const unsigned int _idy=(blockIdx.y*blockDim.y)+threadIdx.y;

	double *cof;
	cof=(double *)malloc((SIZE - 1) * (SIZE - 1) * sizeof(double));
	int i=_idy;
	int j=_idx;

	for (int k = 0; k < SIZE - 1; k++)
	{
		for (int t = 0; t < SIZE - 1; t++)
		{
			Point(cof, k, t, SIZE - 1) = Point(ori, k < i ? k : k + 1, t < j ? t : t + 1, SIZE);
		}
	}
	Point(inv, j, i, SIZE) = Loop(cof,SIZE-1)* ((i + j) % 2 == 0 ? 1 : -1)/det;
	free(cof);
}

void Inverse_Matrix_Handle(double *ori, double *inv,dim3 Blocks_Per_Grid,dim3 Threads_Per_Block,double det)
{
	if (0 == det)
	{
		cout << "Warning : Singular Matrix !" << endl;
		exit(1);
	}
	Kernel_Function<<<Blocks_Per_Grid,Threads_Per_Block>>>(ori,inv,det);
}

int main()
{
	unsigned int Byte_Size = SIZE * SIZE * sizeof(double);
	double *Matrix_Ori = (double *)malloc(Byte_Size);
	Initialize_Matrix(Matrix_Ori);
	Show_Matrix(Matrix_Ori, "Original Matrix :");

	double *Matrix_Inv = (double *)malloc(Byte_Size);

	/* Initial Threads Blocks Begin */
	int thread_xdim = SIZE;
	int thread_ydim = SIZE;
	const dim3 Threads_Per_Block(thread_xdim, thread_ydim);
	const dim3 Blocks_Per_Grid(1, 1);
	/* Initial Threads Blocks End */

	/* Initial Memory Begin */
	double *Matrix_GPU;
	double *Matrix_Inv_GPU;
	Cuda_Call(cudaMalloc((void **)&Matrix_GPU, Byte_Size));
	Cuda_Call(cudaMalloc((void **)&Matrix_Inv_GPU, Byte_Size));
	Cuda_Call(cudaMemcpy(Matrix_GPU, Matrix_Ori, Byte_Size, cudaMemcpyHostToDevice));
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
		float det=Get_Det(Matrix_Ori,SIZE);
		Inverse_Matrix_Handle(Matrix_GPU,Matrix_Inv_GPU,Blocks_Per_Grid,Threads_Per_Block,det);
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

	//double *Matrix_Res = (double *)malloc(Byte_Size);
	//Matrix_Mult(Matrix_Ori, Matrix_Inv, Matrix_Res);
	//Show_Matrix(Matrix_Res, "Mult Matrix :");

	/* Free Memory Begin */
	Cuda_Call(cudaFree(Matrix_GPU));
	free(Matrix_Ori);
	free(Matrix_Inv);
	/* Free Memory End */
	return 0;
}
