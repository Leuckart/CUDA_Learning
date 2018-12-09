/**************************************************
	> File Name:  invert.cpp
	> Author:     Leuckart
	> Time:       2018-12-09 19:13
**************************************************/

#include "invert.h"

float Get_Det(float *mat, int n)
{
	if (n == 1)
	{
		return mat[0];
	}
	float ans = 0;
	float *temp = (float *)malloc(SIZE * SIZE * sizeof(float));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			for (int k = 0; k < n - 1; k++)
			{
				if (k >= i)
				{
					Element(temp, j, k) = Element(mat, j + 1, k + 1);
				}
				else
				{
					Element(temp, j, k) = Element(mat, j + 1, k);
				}
			}
		}
		float t = Get_Det(temp, n - 1);
		if (i % 2 == 0)
		{
			ans += mat[i] * t;
		}
		else
		{
			ans -= mat[i] * t;
		}
	}
	return ans;
}

void Get_Adj(float *arcs, int n, float *ans)
{
	if (n == 1)
	{
		ans[0] = 1;
		return;
	}
	float *temp = (float *)malloc(SIZE * SIZE * sizeof(float));
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
						temp[k * SIZE + t] = arcs[(k >= i ? k + 1 : k) * SIZE + t + 1];
					}
					else
					{
						temp[k * SIZE + t] = arcs[(k >= i ? k + 1 : k) * SIZE + t];
					}
				}
			}

			ans[j * SIZE + i] = Get_Det(temp, n - 1);
			if ((i + j) % 2 == 1)
			{
				ans[j * SIZE + i] = -ans[j * SIZE + i];
			}
		}
	}
}

void Inverse_Matrix(float *src, float *des)
{
	float flag = Get_Det(src, SIZE);
	float *t = (float *)malloc(SIZE * SIZE * sizeof(float));
	if (0 == flag)
	{
		cout << "Warning : Singular Matrix !" << endl;
		exit(1);
	}
	else
	{
		Get_Adj(src, SIZE, t);
		for (int i = 0; i < SIZE; i++)
		{
			for (int j = 0; j < SIZE; j++)
			{
				des[i * SIZE + j] = t[i * SIZE + j] / flag;
			}
		}
	}
}

void Show_Matrix(float *mat, const char *mesg)
{
	cout << mesg << endl;

	unsigned int mat_size = SIZE * SIZE;
	int flag = 0;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			cout << Element(mat, i, j) << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void Initialize_Matrix(float *mat)
{
	/* should replace by urandom. Leuckart. */
	srand((unsigned)time(0));
	unsigned int mat_size = SIZE * SIZE;

	for (int i = 0; i < mat_size; i++)
	{
		mat[i] = rand() % 100 * 0.01;
	}
}

__global__ void Matrix_Mult(float mata[SIZE][SIZE], float matb[SIZE][SIZE])
{
	//print()
	//cout<<mata[0][0]<<" "<<matb[0][0]<<endl;
}

int main()
{
	unsigned int Byte_Size = SIZE * SIZE * sizeof(float);
	float *Matrix_Ori = (float *)malloc(Byte_Size);

	Initialize_Matrix(Matrix_Ori);
	Show_Matrix(Matrix_Ori, "Original Matrix :");

	float *Matrix_Inv = (float *)malloc(Byte_Size);
	Inverse_Matrix(Matrix_Ori, Matrix_Inv);
	Show_Matrix(Matrix_Inv, "Inverse Matrix :");

	float *Matrix_Inv_Inv = (float *)malloc(Byte_Size);
	Inverse_Matrix(Matrix_Inv, Matrix_Inv_Inv);
	Show_Matrix(Matrix_Inv_Inv, "Inverse Inverse Matrix :");

	/* Initial Threads Blocks Begin */
	int thread_xdim = SIZE;
	int thread_ydim = SIZE;
	const dim3 Threads_Per_Block(thread_xdim, thread_ydim);
	const dim3 Blocks_Per_Grid(1, 1);
	/* Initial Threads Blocks End */

	/* Initial Memory Begin */
	//float *Matrix_CPU=original_matrix;//(float *)malloc(Byte_Size);
	//float *Matrix_GPU;
	//Cuda_Call(cudaMalloc((void **)&Matrix_GPU,Byte_Size));
	/* Initial Memory Begin */

	return 0;
}
