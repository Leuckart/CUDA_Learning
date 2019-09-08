/**************************************************
	> File Name:  cublas_2019.09.08_2.cpp
	> Author:     Leuckart
	> Time:       2019-09-08 14:57
**************************************************/

/* nvcc *.cu -lcublas */

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <iostream>
#include <math.h>

using namespace std;

#define M 5
#define N 3
#define BYTE 128

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
	int8_t *h_A = (int8_t *)malloc(N * M * sizeof(int8_t));
	int8_t *h_B = (int8_t *)malloc(N * M * sizeof(int8_t));
	int32_t *h_C = (int32_t *)malloc(M * M * sizeof(int32_t));

	float *f_A = (float *)malloc(N * M * sizeof(float));
	float *f_B = (float *)malloc(N * M * sizeof(float));
	for (int i = 0; i < N * M; i++)
	{
		f_A[i] = (float)(rand() % 5);
		f_B[i] = (float)(rand() % 5);
	}

	for (int i = 0; i < N * M; i++)
	{
		h_A[i] = (char)(round(f_A[i] * BYTE));
		h_B[i] = (char)(round(f_B[i] * BYTE));
	}

	// 创建并初始化 CUBLAS 库对象
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	/*
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED)
		{
			cout << "CUBLAS 对象实例化出错" << endl;
		}
		getchar();
		return EXIT_FAILURE;
	}

	char *d_A, *d_B;
	int *d_C;
	// 在 显存 中为将要计算的矩阵开辟空间
	cudaMalloc(
		(void **)&d_A,		 // 指向开辟的空间的指针
		N * M * sizeof(char) //　需要开辟空间的字节数
	);
	cudaMalloc(
		(void **)&d_B,
		N * M * sizeof(char));

	// 在 显存 中为将要存放运算结果的矩阵开辟空间
	cudaMalloc(
		(void **)&d_C,
		M * M * sizeof(int));

	// 将矩阵数据传递进 显存 中已经开辟好了的空间
	cublasSetVector(
		N * M,		  // 要存入显存的元素个数
		sizeof(char), // 每个元素大小
		h_A,		  // 主机端起始地址
		1,			  // 连续元素之间的存储间隔
		d_A,		  // GPU 端起始地址
		1			  // 连续元素之间的存储间隔
	);
	//注意：当矩阵过大时，使用cudaMemcpy是更好地选择：
	//cudaMemcpy(d_A, h_A, sizeof(char)*N*M, cudaMemcpyHostToDevice);

	cublasSetVector(
		N * M,
		sizeof(char),
		h_B,
		1,
		d_B,
		1);
	//cudaMemcpy(d_B, h_B, sizeof(char)*N*M, cudaMemcpyHostToDevice);
	// 同步函数
	cudaThreadSynchronize();

	// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
	float a = 1.0;
	float b = 0;
	// 矩阵相乘。该函数必然将数组解析成列优先数组
	cublasSgemm(
		handle,		 // blas 库对象
		CUBLAS_OP_T, // 矩阵 A 属性参数
		CUBLAS_OP_T, // 矩阵 B 属性参数
		M,			 // A, C 的行数
		M,			 // B, C 的列数
		N,			 // A 的列数和 B 的行数
		&a,			 // 运算式的 α 值
		d_A,		 // A 在显存中的地址
		N,			 // lda
		d_B,		 // B 在显存中的地址
		M,			 // ldb
		&b,			 // 运算式的 β 值
		d_C,		 // C 在显存中的地址(结果矩阵)
		M			 // ldc
	);
	cublasGemmEx(handle,		   //句柄
				 CUBLAS_OP_T,	  //矩阵 A 属性参数
				 CUBLAS_OP_T,	  //矩阵 B 属性参数
				 M,				   //A, C 的行数
				 M,				   //B, C 的列数
				 N,				   //A 的列数和 B 的行数
				 &a,			   //运算式的 α 值
				 d_A,			   //A矩阵
				 CUDA_R_8I,		   //A矩阵计算模式，int8型
				 N,				   //A矩阵的列数
				 d_B,			   //B矩阵
				 CUDA_R_8I,		   //B矩阵计算模式，int8型
				 M,				   //B矩阵的行数
				 &b,			   //乘法因子beta
				 d_C,			   //C结果矩阵
				 CUDA_R_32I,	   //C矩阵计算模式，int32型
				 M,				   //C矩阵的行数
				 CUDA_R_32I,	   //计算模式，int32模式
				 CUBLAS_GEMM_ALGO0 //算法参数
				 )

		// 同步函数
		cudaDeviceSynchronize();

	// 从 显存 中取出运算结果至 内存中去
	cublasGetVector(
		M * M,		 //  要取出元素的个数
		sizeof(int), // 每个元素大小
		d_C,		 // GPU 端起始地址
		1,			 // 连续元素之间的存储间隔
		h_C,		 // 主机端起始地址
		1			 // 连续元素之间的存储间隔
	);

	//或使用cudaMemcpy(h_C, d_C, sizeof(int)*M*M, cudaMemcpyDeviceToHost);
	// 打印运算结果
	cout << "计算结果的转置 ( (A*B)的转置 )：" << endl;

	for (int i = 0; i < M * M; i++)
	{
		cout << h_C[i] << " ";
		if ((i + 1) % M == 0)
			cout << endl;
	}

	//注意，这里需要进行归一化操作，乘出来的矩阵需要除以128*128，以还原原来的大小。在此就省略这一步。
	// 清理掉使用过的内存
	free(h_C);
	free(h_B);
	free(h_A);
	free(f_B);
	free(f_A);

	try
	{
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}
	catch (...)
	{
		cout << "cudaFree Error!" << endl;
		// 释放 CUBLAS 库对象
	}

	cublasDestroy(handle);
	*/
	return 0;
}
