/**************************************************
	> File Name:  invert.h
	> Author:     Leuckart
	> Time:       2018-12-09 19:25
**************************************************/
#ifndef INVERT_H_
#define INVERT_H_

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
// if block num 1, shared memory should be broadcast
#define SIZE 161
#define Point(_arr, _i, _j, _size) ((_arr)[(_i) * (_size) + (_j)])
using namespace std;
double Get_Det(double *mat, int n);

#endif
