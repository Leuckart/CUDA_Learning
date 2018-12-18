/**************************************************
	> File Name:  utils.cu
	> Author:     Leuckart
	> Time:       2018-12-18 21:13
**************************************************/

#include "inverse.h"

void Show_Matrix(double *mat, const char *mesg)
{
	cout << mesg << endl;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			if (Element(mat, i, j) < 0.00001 && Element(mat, i, j) > -0.00001)
			{
				cout << "0"
					 << " ";
			}
			else
			{
				cout << Element(mat, i, j) << " ";
			}
		}
		cout << endl;
	}
	cout << endl;
}

void Matrix_Mult(double *a, double *b, double *res)
{
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			double temp = 0.0;
			for (int k = 0; k < SIZE; k++)
			{
				temp += Element(a, i, k) * Element(b, k, j);
			}
			Element(res, i, j) = temp;
		}
	}
}

void Initialize_Matrix(double *mat)
{
	srand((unsigned)time(0));
	unsigned int mat_size = SIZE * SIZE;

	for (int i = 0; i < mat_size; i++)
	{
		mat[i] = rand() % 100 * 0.01;
	}
}