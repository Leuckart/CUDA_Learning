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

			ans[j][i] = Get_Det(temp, n - 1); //此处顺便进行了转置
			if ((i + j) % 2 == 1)
			{
				ans[j][i] = -ans[j][i];
			}
		}
	}
}

//得到给定矩阵src的逆矩阵保存到des中。
bool GetMatrixInverse(float src[SIZE][SIZE], int n, float des[SIZE][SIZE])
{
	float flag = Get_Det(src, n);
	float t[SIZE][SIZE];
	if (0 == flag)
	{
		cout << "原矩阵行列式为0，无法求逆。请重新运行" << endl;
		return false; //如果算出矩阵的行列式为0，则不往下进行
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

	return true;
}

int main()
{
	bool flag; //标志位，如果行列式为0，则结束程序
	int row = SIZE;
	int col = SIZE;
	float matrix_before[SIZE][SIZE]{}; //{1,2,3,4,5,6,7,8,9};

	//随机数据，可替换
	srand((unsigned)time(0));
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			matrix_before[i][j] = rand() % 100 * 0.01;
		}
	}

	cout << "原矩阵：" << endl;

	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			//cout << matrix_before[i][j] <<" ";
			cout << *(*(matrix_before + i) + j) << " ";
		}
		cout << endl;
	}

	float matrix_after[SIZE][SIZE]{};
	flag = GetMatrixInverse(matrix_before, SIZE, matrix_after);
	if (false == flag)
		return 0;

	cout << "逆矩阵：" << endl;

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << matrix_after[i][j] << " ";
			//cout << *(*(matrix_after+i)+j)<<" ";
		}
		cout << endl;
	}

	GetMatrixInverse(matrix_after, SIZE, matrix_before);

	cout << "反算的原矩阵：" << endl; //为了验证程序的精度

	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			//cout << matrix_before[i][j] <<" ";
			cout << *(*(matrix_before + i) + j) << " ";
		}
		cout << endl;
	}

	return 0;
}
