/**************************************************
	> File Name:  cublas_2019.09.08_2.cpp
	> Author:     Leuckart
	> Time:       2019-09-08 14:57
**************************************************/

/* nvcc *.cu -lcublas */

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

#define M 2
#define N 2
#define BYTE 128

template <typename T>
void Display(T *array, int row, int col)
{
	for (int i = 0; i < row * col; i++)
	{
		cout << static_cast<int32_t>(array[i]) << " ";
		if ((i + 1) % col == 0)
			cout << endl;
	}
	cout << endl;
}

int main()
{
	int8_t *h_A = (int8_t *)malloc(N * M * sizeof(int8_t));
	int8_t *h_B = (int8_t *)malloc(N * M * sizeof(int8_t));
	int32_t *h_C = (int32_t *)malloc(M * M * sizeof(int32_t));

	for (int i = 0; i < N * M; i++)
	{
		h_A[i] = (int8_t)(rand() % 5);
		h_B[i] = (int8_t)(rand() % 5);
	}
	//cout << h_A[0] << endl;
	Display(h_A, M, N);
	Display(h_B, N, M);

	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED)
		{
			cout << "Initialization Error." << endl;
		}
		return EXIT_FAILURE;
	}

	int8_t *d_A, *d_B;
	int32_t *d_C;
	cudaMalloc((void **)&d_A, N * M * sizeof(int8_t));
	cudaMalloc((void **)&d_B, N * M * sizeof(int8_t));
	cudaMalloc((void **)&d_C, M * M * sizeof(int32_t));

	cublasSetVector(N * M, sizeof(int8_t), h_A, 1, d_A, 1);
	cublasSetVector(N * M, sizeof(int8_t), h_B, 1, d_B, 1);
	//cudaMemcpy(d_A, h_A, sizeof(char) * N * M, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_B, h_B, sizeof(char) * N * M, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	int8_t a = 1, b = 0;
	cout << "OK" << endl;
	cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T,
										  M, M, N,
										  &a,
										  d_A, CUDA_R_8I, M,
										  d_B, CUDA_R_8I, N,
										  &b,
										  d_C, CUDA_R_32I, M,
										  CUDA_R_32I,
										  CUBLAS_GEMM_ALGO0);
	cudaDeviceSynchronize();
	cublasGetVector(M * M, sizeof(int32_t), d_C, 1, h_C, 1);
	//cudaMemcpy(h_C, d_C, sizeof(int) * M * M, cudaMemcpyDeviceToHost);

	cout << "C:" << endl;
	Display<int32_t>(h_C, M, M);

	free(h_C);
	free(h_B);
	free(h_A);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);
	return 0;
}
