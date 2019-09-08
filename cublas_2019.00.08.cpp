/**************************************************
	> File Name:  cublas_2019.00.08.cu
	> Author:     Leuckart
	> Time:       2019-09-08 13:07
**************************************************/

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;

#define M 5
#define N 3

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
	cublasStatus_t status;
	float *h_A = (float *)malloc(N * M * sizeof(float));
	float *h_B = (float *)malloc(N * M * sizeof(float));
	float *h_C = (float *)malloc(M * M * sizeof(float));
	for (int i = 0; i < N * M; i++)
	{
		h_A[i] = (float)(rand() % 10 + 1);
		h_B[i] = (float)(rand() % 10 + 1);
	}

	Display(h_A, M, N);
	Display(h_B, M, N);

	cout << "OK" << endl;
	return 0;
}
