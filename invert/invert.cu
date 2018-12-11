/**************************************************
	> File Name:  invert.cpp
	> Author:     Leuckart
	> Time:       2018-12-09 19:13
**************************************************/

#include "invert.h"

double Get_Det(double *mat, int n)
{
	if (n == 1)
	{
		return mat[0];
	}
	double ans = 0;
	double *cof = (double *)malloc((n - 1) * (n - 1) * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			for (int k = 0; k < n - 1; k++)
			{
				Point(cof, j, k, n - 1) = Point(mat, j + 1, k < i ? k : k + 1, n);
			}
		}
		double t = Get_Det(cof, n - 1);
		ans += mat[i] * t * (i % 2 == 0 ? 1 : -1);
	}
	free(cof);
	return ans;
}

void Inverse_Matrix(double *ori, double *inv)
{
	double det = Get_Det(ori, SIZE);
	if (0 == det)
	{
		cout << "Warning : Singular Matrix !" << endl;
		exit(1);
	}

	double *cof = (double *)malloc((SIZE - 1) * (SIZE - 1) * sizeof(double));
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
			Point(inv, j, i, SIZE) = Get_Det(cof, SIZE - 1) * ((i + j) % 2 == 0 ? 1 : -1)/det;
		}
	}
	free(cof);
}

__global__ void Kernel_Function(double *ori,double *inv,int now)
{
	const unsigned int _idx=(blockIdx.x*blockDim.x)+threadIdx.x;
	const unsigned int _idy=(blockIdx.y*blockDim.y)+threadIdx.y;
	if(_idy==0)
	{
		return;
	}
	
	const unsigned int index=((gridDim.x*blockDim.x)*_idx)+_idy;

	__shared__ double memory[SIZE];
	if(_idy!=0)
	{
		memory[_idx]=Point(ori,_idx,_idx,SIZE);
	}
	__syncthreads();

	//inv[index]=index;
	//inv[_idx*SIZE+_idy]+=index;
	//inv[index]=ori[index];

	//__syncthreads();
	//__shared__ double ii;
	//double ii;
	//ii=Point(ori,now,now,SIZE);
	double ii=memory[now];
	double temp=0.0;

	/*__syncthreads();
	if(_idx==now)
	{
		temp=1./ii;//1./Point(ori,_idx,now,SIZE);
		for(int i=0;i<SIZE;i++)
		{
			Point(ori,now,i,SIZE)*=temp;
			Point(inv,now,i,SIZE)*=temp;
		}
	}
	__syncthreads();*/

	if(_idx!=now)
	{
		temp=Point(ori,_idx,now,SIZE)/ii;
		//temp=ii/Point(ori,_idx,now,SIZE);
		for(int i=0;i<SIZE;i++)
		{
			Point(ori,_idx,i,SIZE)-=Point(ori,now,i,SIZE)*temp;
			Point(inv,_idx,i,SIZE)-=Point(inv,now,i,SIZE)*temp;
		}
	}
	else
	{
		return;
	}
}

__global__ void Kernel_Normalize(double *ori,double *inv)
{
	const unsigned int _idx=(blockIdx.x*blockDim.x)+threadIdx.x;
	const unsigned int _idy=(blockIdx.y*blockDim.y)+threadIdx.y;
	if(_idy==0)
	{
		return;
	}
	
	const unsigned int index=((gridDim.x*blockDim.x)*_idx)+_idy;

	//__shared__ double ii;
	double ii;
	ii=Point(ori,_idx,_idx,SIZE);
	double temp=0.0;

	temp=1./ii;//1./Point(ori,_idx,now,SIZE);
	for(int i=0;i<SIZE;i++)
	{
		Point(ori,_idx,i,SIZE)*=temp;
		Point(inv,_idx,i,SIZE)*=temp;
	}
	//__syncthreads();
}

void Inverse_Matrix_Handle(double *ori, double *inv,dim3 Blocks_Per_Grid,dim3 Threads_Per_Block)
{
	for(int i=0;i<SIZE;i++)
	{
		Kernel_Function<<<Blocks_Per_Grid,Threads_Per_Block>>>(ori,inv,i);
		cudaThreadSynchronize();
		//Kernel_Normalize<<<Blocks_Per_Grid,Threads_Per_Block>>>(ori,inv,i);
		//cudaThreadSynchronize();
	}
	Kernel_Normalize<<<Blocks_Per_Grid,Threads_Per_Block>>>(ori,inv);
	cudaThreadSynchronize();
}

void Show_Matrix(double *mat, const char *mesg)
{
	cout << mesg << endl;

	unsigned int mat_size = SIZE * SIZE;
	int flag = 0;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			if(Point(mat,i,j,SIZE)<0.00001&&Point(mat,i,j,SIZE)>-0.00001)
			{
				cout<<"0"<<" ";
				//cout << Point(mat, i, j, SIZE) << " ";
			}
			else
			{
				cout << Point(mat, i, j, SIZE) << " ";
			}
		}
		cout << endl;
	}
	cout << endl;
}

void Initialize_Matrix(double *mat)
{
	/* should replace by urandom. Leuckart. */
	srand((unsigned)time(0));
	unsigned int mat_size = SIZE * SIZE;

	for (int i = 0; i < mat_size; i++)
	{
		mat[i] = rand() % 100 * 0.01;
	}
}

int main()
{
	unsigned int Byte_Size = SIZE * SIZE * sizeof(double);
	double *Matrix_Ori = (double *)malloc(Byte_Size);

	Initialize_Matrix(Matrix_Ori);
	Show_Matrix(Matrix_Ori, "Original Matrix :");

	//cout << Get_Det(Matrix_Ori, SIZE) << endl;

	double *Matrix_Inv = (double *)malloc(Byte_Size);
	Inverse_Matrix(Matrix_Ori, Matrix_Inv);
	Show_Matrix(Matrix_Inv, "Inverse Matrix :");

	double *Matrix_Inv_Inv = (double *)malloc(Byte_Size);
	//Inverse_Matrix(Matrix_Inv, Matrix_Inv_Inv);
	//Show_Matrix(Matrix_Inv_Inv, "Inverse Inverse Matrix :");

	/* Initial Threads Blocks Begin */
	int thread_xdim = SIZE;
	int thread_ydim = SIZE;
	const dim3 Threads_Per_Block(thread_xdim, thread_ydim);
	const dim3 Blocks_Per_Grid(1, 1);
	/* Initial Threads Blocks End */

	/* Initial Memory Begin */
	double *Matrix_GPU;
	double *Matrix_Inv_GPU;
	double *Matrix_Inv_Inv_GPU;
	double *ident=(double *)malloc(Byte_Size);
	for(int i=0;i<SIZE;i++)
	{
		Point(ident,i,i,SIZE)=1;
	}
	Cuda_Call(cudaMalloc((void **)&Matrix_GPU, Byte_Size));
	Cuda_Call(cudaMalloc((void **)&Matrix_Inv_GPU, Byte_Size));
	Cuda_Call(cudaMalloc((void **)&Matrix_Inv_Inv_GPU, Byte_Size));
	Cuda_Call(cudaMemcpy(Matrix_GPU, Matrix_Ori, Byte_Size, cudaMemcpyHostToDevice));
	Cuda_Call(cudaMemcpy(Matrix_Inv_GPU, ident, Byte_Size, cudaMemcpyHostToDevice));
	Cuda_Call(cudaMemcpy(Matrix_Inv_Inv_GPU, ident, Byte_Size, cudaMemcpyHostToDevice));
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
		Inverse_Matrix_Handle(Matrix_GPU,Matrix_Inv_GPU,Blocks_Per_Grid,Threads_Per_Block);
		//Inverse_Matrix_Handle(Matrix_Inv_GPU,Matrix_Inv_Inv_GPU,Blocks_Per_Grid,Threads_Per_Block);
		//Inverse_Matrix(Matrix_Ori,Matrix_Inv);
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
	Cuda_Call(cudaMemcpy(Matrix_Inv, Matrix_Inv_GPU, Byte_Size, cudaMemcpyDeviceToHost));
	//Cuda_Call(cudaMemcpy(Matrix_Ori, Matrix_GPU, Byte_Size, cudaMemcpyDeviceToHost));
	Cuda_Call(cudaMemcpy(Matrix_Inv_Inv, Matrix_Inv_Inv_GPU, Byte_Size, cudaMemcpyDeviceToHost));
	/* Copy GPU To CPU End */

	Show_Matrix(Matrix_Ori, "Original Matrix (Should Be I):");
	Show_Matrix(Matrix_Inv, "Inv Matrix :");
	//Show_Matrix(Matrix_Inv_Inv, "Inv Inv Matrix :");

	double *Matrix_Mult = (double *)malloc(Byte_Size);
	for(int i=0;i<SIZE;i++)
	{
		for(int j=0;j<SIZE;j++)
		{
			double temp=0.0;
			for(int k=0;k<SIZE;k++)
			{
				temp+=Point(Matrix_Ori,i,k,SIZE)*Point(Matrix_Inv,k,j,SIZE);
			}
			Point(Matrix_Mult,i,j,SIZE)=temp;
		}
	}
	Show_Matrix(Matrix_Mult, "Mult Matrix :");

	/* Free Memory Begin */
	Cuda_Call(cudaFree(Matrix_GPU));
	free(Matrix_Ori);
	/* Free Memory End */
	return 0;
}
