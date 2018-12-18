/**************************************************
	> File Name:  invert.cpp
	> Author:     Leuckart
	> Time:       2018-12-09 19:13
**************************************************/

#include "inverse.h"

void Row_Function(double *ori, double *inv, int now)
{
	double ii = Element(ori, now, now);
	double temp = 0.0;
	for (int i = 0; i < SIZE; i++)
	{
		if (i == now)
		{
			continue;
		}
		temp = Element(ori, i, now) / ii;
		for (int j = 0; j < SIZE; j++)
		{
			Element(ori, i, j) -= Element(ori, now, j) * temp;
			Element(inv, i, j) -= Element(inv, now, j) * temp;
		}
	}
}

void Row_Normalize(double *ori, double *inv)
{
	for (int i = 0; i < SIZE; i++)
	{
		double temp = 1. / Element(ori, i, i);
		for (int j = 0; j < SIZE; j++)
		{
			//Element(ori,i,j)*=temp;
			Element(inv, i, j) *= temp;
		}
	}
}

void Inverse_Matrix_Handle(double *ori, double *inv)
{
	for (int i = 0; i < SIZE; i++)
	{
		Row_Function(ori, inv, i);
	}
	Row_Normalize(ori, inv);
}

int main()
{
	unsigned int Byte_Size = SIZE * SIZE * sizeof(double);
	double *Matrix_Ori = (double *)malloc(Byte_Size);
	Initialize_Matrix(Matrix_Ori);
	double *Matrix_Ori_Copy = (double *)malloc(Byte_Size);
	memcpy(Matrix_Ori_Copy, Matrix_Ori, Byte_Size);

	double *Matrix_Inv = (double *)malloc(Byte_Size);
	for (int i = 0; i < SIZE; i++)
	{
		Element(Matrix_Inv, i, i) = 1;
	}

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
		
		Inverse_Matrix_Handle(Matrix_Ori_Copy, Matrix_Inv);

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
	free(Matrix_Ori);
	free(Matrix_Ori_Copy);
	free(Matrix_Inv);
	//free(Matrix_Res);
	/* Free Memory End */
	return 0;
}
