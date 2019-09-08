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

#define M 8
#define L 4
#define N 12

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
	int8_t *h_A = (int8_t *)malloc(M * L * sizeof(int8_t));
	int8_t *h_B = (int8_t *)malloc(L * N * sizeof(int8_t));
	int32_t *h_C = (int32_t *)malloc(M * N * sizeof(int32_t));

	for (int i = 0; i < M * L; i++)
		h_A[i] = (int8_t)(rand() % 3);
	for (int i = 0; i < L * N; i++)
		h_B[i] = (int8_t)(rand() % 3);

	Display(h_A, M, L);
	Display(h_B, L, N);

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
	cudaMalloc((void **)&d_A, M * L * sizeof(int8_t));
	cudaMalloc((void **)&d_B, L * N * sizeof(int8_t));
	cudaMalloc((void **)&d_C, M * N * sizeof(int32_t));

	cublasSetVector(M * L, sizeof(int8_t), h_A, 1, d_A, 1);
	cublasSetVector(L * N, sizeof(int8_t), h_B, 1, d_B, 1);
	//cudaMemcpy(d_A, h_A, sizeof(char) * N * M, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_B, h_B, sizeof(char) * N * M, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	int32_t alpha = 1, beta = 0;
	cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T,
				 M, N, L,
				 &alpha,
				 d_A, CUDA_R_8I, L,
				 d_B, CUDA_R_8I, N,
				 &beta,
				 d_C, CUDA_R_32I, M,
				 CUDA_R_32I,
				 CUBLAS_GEMM_DEFAULT);
	cudaDeviceSynchronize();
	cublasGetVector(M * N, sizeof(int32_t), d_C, 1, h_C, 1);
	//cudaMemcpy(h_C, d_C, sizeof(int) * M * M, cudaMemcpyDeviceToHost);

	cout << "C:" << endl;
	Display<int32_t>(h_C, N, M);

	free(h_C);
	free(h_B);
	free(h_A);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);
	return 0;
}
