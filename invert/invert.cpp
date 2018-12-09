/**************************************************
	> File Name:  invert.cpp
	> Author:     Leuckart
	> Time:       2018-12-09 19:13
**************************************************/

#include "invert.h"

float Get_Det(float mat[SIZE][SIZE], int n)
{
	if (n == 1)
	{
		return mat[0][0];
	}
	float ans = 0;
	float temp[SIZE][SIZE];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			for (int k = 0; k < n - 1; k++)
			{
				if (k >= i)
				{
					temp[j][k] = mat[j + 1][k + 1];
				}
				else
				{
					temp[j][k] = mat[j + 1][k];
				}
			}
		}
		float t = Get_Det(temp, n - 1);
		if (i % 2 == 0)
		{
			ans += mat[0][i] * t;
		}
		else
		{
			ans -= mat[0][i] * t;
		}
	}
	return ans;
}

void Get_Adj(float arcs[SIZE][SIZE], int n, float ans[SIZE][SIZE])
{
	if (n == 1)
	{
		ans[0][0] = 1;
		return;
	}
	float temp[SIZE][SIZE];
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n - 1; k++)
			{
				for (int t = 0; t < n - 1; t++)
				{
					if (t >= j)
					{
						temp[k][t] = arcs[k >= i ? k + 1 : k][t + 1];
					}
					else
					{
						temp[k][t] = arcs[k >= i ? k + 1 : k][t];
					}
				}
			}

			ans[j][i] = Get_Det(temp, n - 1);
			if ((i + j) % 2 == 1)
			{
				ans[j][i] = -ans[j][i];
			}
		}
	}
}

bool Inverse_Matrix(float src[SIZE][SIZE], int n, float des[SIZE][SIZE])
{
	float flag = Get_Det(src, n);
	float t[SIZE][SIZE];
	if (0 == flag)
	{
		cout << "Warning : Singular Matrix !" << endl;
		return true;
	}
	else
	{
		Get_Adj(src, n, t);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				des[i][j] = t[i][j] / flag;
			}
		}
	}
	return false;
}

void Show_Matrix(float mat[SIZE][SIZE], const char *mesg)
{
	cout << mesg << endl;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			cout << mat[i][j] << " ";
		}
		cout << endl;
	}
	cout<<endl;
}

int main()
{
	bool is_singular;
	int row = SIZE;
	int col = SIZE;
	float matrix_before[SIZE][SIZE]{}; //{1,2,3,4,5,6,7,8,9};

	/* should replace by urandom. Leuckart. */
	srand((unsigned)time(0));
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			matrix_before[i][j] = rand() % 100 * 0.01;
		}
	}

	Show_Matrix(matrix_before, "Original Matrix :");

	float matrix_after[SIZE][SIZE]{};
	is_singular = Inverse_Matrix(matrix_before, SIZE, matrix_after);
	if (true == is_singular)
		return 0;

	Show_Matrix(matrix_after, "Inverse Matrix :");

	float inverse_inverse_matrix[SIZE][SIZE]{};
	Inverse_Matrix(matrix_after, SIZE, inverse_inverse_matrix);

	Show_Matrix(inverse_inverse_matrix, "Inverse Inverse Matrix :");

	return 0;
}
