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

#define SIZE 8
#define Element(_arr, _i, _j) ((_arr)[(_i)*8 + (_j)])

using namespace std;

#endif
