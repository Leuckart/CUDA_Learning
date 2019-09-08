/**************************************************
	> File Name:  cublas_2019.09.08_1.cu
	> Author:     Leuckart
	> Time:       2019-09-08 13:07
**************************************************/

/* nvcc *.cu -lcublas */

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;

#define M 3
#define L 2
#define N 4

/* CuBLAS: Col Major. */

/*
* C = alpha * A * B + beta * C
* CUBLAS_OP_N: Not Trans, CUBLAS_OP_T: Trans.
* cublasStatus_t cublasSgemm(
* 	cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
* 	int m, int n, int k,
* 	const float *alpha,		 // Host or device pointer.
* 	const float *A, int lda, // Row of Real A: m * k.
* 	const float *B, int ldb, // Row of Real B: k * n.
* 	const float *beta,		 // Host or device pointer
* 	float *C, int ldc);		 // Line of C. C: m * n.
*/

void Display(float *array, int row, int col)
{
	for (int i = 0; i < row * col; i++)
	{
		cout << array[i] << " ";
		if ((i + 1) % col == 0)
			cout << endl;
	}
	cout << endl;
}

int main()
{
	float *h_A = (float *)malloc(M * L * sizeof(float));
	float *h_B = (float *)malloc(L * N * sizeof(float));
	float *h_C = (float *)malloc(M * N * sizeof(float));

	for (int i = 0; i < M * L; i++)
		h_A[i] = (float)(rand() % 5);
	for (int i = 0; i < L * N; i++)
		h_B[i] = (float)(rand() % 5);

	cout << "A:" << endl;
	Display(h_A, M, L);
	cout << "B:" << endl;
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

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, M * L * sizeof(float));
	cudaMalloc((void **)&d_B, L * N * sizeof(float));
	cudaMalloc((void **)&d_C, M * N * sizeof(float));

	cublasSetVector(M * L, sizeof(float), h_A, 1, d_A, 1);
	cublasSetVector(L * N, sizeof(float), h_B, 1, d_B, 1);
	//cudaMemcpy(d_A, h_A, sizeof(float) * N * M, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_B, h_B, sizeof(float) * N * M, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	float a = 1., b = 0.;
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
				M, N, L,
				&a,
				d_A, L, d_B, N,
				&b,
				d_C, M);
	cudaDeviceSynchronize();
	cublasGetVector(M * N, sizeof(float), d_C, 1, h_C, 1);

	cout << "C:" << endl;
	Display(h_C, N, M);

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);

	return 0;
}
