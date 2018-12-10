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
	float *cof = (float *)malloc((n - 1) * (n - 1) * sizeof(float));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			for (int k = 0; k < n - 1; k++)
			{
				Point(cof, j, k, n - 1) = Point(mat, j + 1, k < i ? k : k + 1, n);
			}
		}
		float t = Get_Det(cof, n - 1);
		ans += mat[i] * t * (i % 2 == 0 ? 1 : -1);
	}
	free(cof);
	return ans;
}

void Get_Adj(float *ori, float *adj)
{
	if (SIZE == 1)
	{
		adj[0] = 1;
		return;
	}
	float *cof = (float *)malloc((SIZE - 1) * (SIZE - 1) * sizeof(float));
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			for (int k = 0; k < SIZE - 1; k++)
			{
				for (int t = 0; t < SIZE - 1; t++)
				{
					Point(cof, k, t, SIZE - 1) = Point(ori, k < i ? k : k + 1, t < j ? t : t + 1, SIZE);
				}
			}
			Point(adj, j, i, SIZE) = Get_Det(cof, SIZE - 1) * ((i + j) % 2 == 0 ? 1 : -1);
		}
	}
	free(cof);
}

void Inverse_Matrix(float *ori, float *inv)
{
	float det = Get_Det(ori, SIZE);
	float *adj = (float *)malloc(SIZE * SIZE * sizeof(float));
	if (0 == det)
	{
		cout << "Warning : Singular Matrix !" << endl;
		exit(1);
	}
	else
	{
		Get_Adj(ori, adj);
		for (int i = 0; i < SIZE; i++)
		{
			for (int j = 0; j < SIZE; j++)
			{
				Point(inv, i, j, SIZE) = Point(adj, i, j, SIZE) / det;
			}
		}
	}
	free(adj);
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
			cout << Point(mat, i, j, SIZE) << " ";
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

__global__ void kernel_convolution(float *image, int size)
{
	for(int i=0;i<size*size;i++)
	{
		image[i]*=2;
	}
}

int main()
{
	unsigned int Byte_Size = SIZE * SIZE * sizeof(float);
	float *Matrix_Ori = (float *)malloc(Byte_Size);

	Initialize_Matrix(Matrix_Ori);
	Show_Matrix(Matrix_Ori, "Original Matrix :");

	cout << Get_Det(Matrix_Ori, SIZE) << endl;

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
	float *Matrix_GPU;
	Cuda_Call(cudaMalloc((void **)&Matrix_GPU, Byte_Size));
	Cuda_Call(cudaMemcpy(Matrix_GPU, Matrix_Ori, Byte_Size, cudaMemcpyHostToDevice));
	/* Initial Memory Begin */

	/* Test On Every Device Begin */
	int Device_All;
	Cuda_Call(cudaGetDeviceCount(&Device_All));
	/* Test On Every Device End */

	for (int device_number = 0; device_number < Device_All; ++device_number)
	{
		/* Set Device Parameters Begin */
		Cuda_Call(cudaSetDevice(device_number));
		struct cudaDeviceProp device_prop;
		char device_prefix[100];
		Cuda_Call(cudaGetDeviceProperties(&device_prop, device_number));
		sprintf(device_prefix, "ID: %d %s: ", device_number, device_prop.name);
		/* Set Device Parameters End */

		/* Initial Time Block Begin */
		cudaEvent_t kernel_start, kernel_stop;
		float delta_time = 0.;
		Cuda_Call(cudaEventCreate(&kernel_start));
		Cuda_Call(cudaEventCreateWithFlags(&kernel_stop, cudaEventBlockingSync));
		Cuda_Call(cudaEventRecord(kernel_start, 0));
		/* Initial Time Block End */

		/* Kernel Function Execute Begin */
		kernel_convolution<<<Blocks_Per_Grid,Threads_Per_Block>>>(Matrix_GPU,SIZE);
		/* Kernel Function Execute End */

		/* Time Clock Begin */
		Cuda_Call(cudaEventRecord(kernel_stop, 0));
		Cuda_Call(cudaEventSynchronize(kernel_stop));
		Cuda_Call(cudaEventElapsedTime(&delta_time, kernel_start, kernel_stop));
		printf("%s %.5fms\n", device_prefix, delta_time);
		Cuda_Call(cudaEventDestroy(kernel_start));
		Cuda_Call(cudaEventDestroy(kernel_stop));
		/* Time Clock End */
	}

	/* Copy GPU To CPU Begin */
	Cuda_Call(cudaMemcpy(Matrix_Ori, Matrix_GPU, Byte_Size, cudaMemcpyDeviceToHost));
	/* Copy GPU To CPU End */

	Show_Matrix(Matrix_Ori, "Original Matrix :");
	
	/* Free Memory Begin */
	Cuda_Call(cudaFree(Matrix_GPU));
	free(Matrix_Ori);
	/* Free Memory End */
	return 0;
}