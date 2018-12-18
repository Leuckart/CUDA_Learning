/**************************************************
	> File Name:  invert.h
	> Author:     Leuckart
	> Time:       2018-12-09 19:25
**************************************************/
#ifndef INVERSE_H_
#define INVERSE_H_

#include <iostream>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#define Cuda_Call(x)                                                              \
	{                                                                             \
		const cudaError_t a = (x);                                                \
		if (a != cudaSuccess)                                                     \
		{                                                                         \
			printf("\nCUDA ERROR: %s (err_num= %d)\n", cudaGetErrorString(a), a); \
			cudaDeviceReset();                                                    \
			exit(1);                                                              \
		}                                                                         \
	}
// if a thread block has too many thread, result will be wrong, this limit is 1024.
// if block num , sared memory should be broadcast
#define SIZE 20
#define Point(_arr, _i, _j, _size) ((_arr)[(_i) * (_size) + (_j)])
#define Element(_arr, _i, _j) ((_arr)[(_i) * (SIZE) + (_j)])
using namespace std;
double Get_Det(double *mat, int n);

void Show_Matrix(double *mat, const char *mesg);
void Matrix_Mult(double *a, double *b, double *res);
void Initialize_Matrix(double *mat);

#endif
