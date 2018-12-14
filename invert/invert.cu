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

__global__ void Row_Kernel_Function(double *ori,double *inv,int now)
{
	const unsigned int _idx=(blockIdx.x*blockDim.x)+threadIdx.x;
	const unsigned int _idy=(blockIdx.y*blockDim.y)+threadIdx.y;
	const unsigned int thread_idx=((gridDim.x*blockDim.x)*_idy)+_idx;
	//const long thread_idx=((gridDim.x*blockDim.x)*_idy)+_idx;

	const unsigned int idx=thread_idx/SIZE;
	const unsigned int idy=thread_idx%SIZE;
	//if(idx>=SIZE||idy>=SIZE)
	if(thread_idx>=SIZE*SIZE)
	{
		return;
	}

	double ii=Point(ori,now,now,SIZE);
	double temp=0.0;

	if(idx!=now)
	{
		temp=Point(ori,idx,now,SIZE)/ii;
		Point(ori,idx,idy,SIZE)-=Point(ori,now,idy,SIZE)*temp;
		Point(inv,idx,idy,SIZE)-=Point(inv,now,idy,SIZE)*temp;
	}
	else
	{}
	__syncthreads();

	/*
	__shared__ double memory[SIZE];
	if(idy==0)
	{
		memory[idx]=Point(ori,idx,idx,SIZE);
	}
	__syncthreads();

	double ii=memory[now];
	double temp=0.0;

	if(idx!=now)
	{
		temp=Point(ori,idx,now,SIZE)/ii;
		Point(ori,idx,idy,SIZE)-=Point(ori,now,idy,SIZE)*temp;
		Point(inv,idx,idy,SIZE)-=Point(inv,now,idy,SIZE)*temp;
	}
	*/
}

__global__ void Row_Kernel_Normalize(double *ori,double *inv)
{
	const unsigned int _idx=(blockIdx.x*blockDim.x)+threadIdx.x;
	const unsigned int _idy=(blockIdx.y*blockDim.y)+threadIdx.y;
	const unsigned int thread_idx=((gridDim.x*blockDim.x)*_idy)+_idx;

	const unsigned int idx=thread_idx/SIZE;
	const unsigned int idy=thread_idx%SIZE;
	//if(idx>=SIZE||idy>=SIZE)
	if(thread_idx>=SIZE*SIZE)
	{
		return;
	}

	double temp=1./Point(ori,idx,idx,SIZE);
	Point(ori,idx,idy,SIZE)*=temp;
	Point(inv,idx,idy,SIZE)*=temp;
	__syncthreads();

	//Point(inv,idx,idy,SIZE)=idx*1000+idy;
	/*
	__shared__ double head[SIZE];
	if(idy==0)
	{
		head[idx]=1./Point(ori,idx,idx,SIZE);
	}
	__syncthreads();
	

	Point(ori,idx,idy,SIZE)*=head[idx];
	Point(inv,idx,idy,SIZE)*=head[idx];
	*/
	//__syncthreads();
}

void Row_Function(double *ori,double *inv,int now)
{
	double ii=Point(ori,now,now,SIZE);
	double temp=0.0;
	for(int i=0;i<SIZE;i++)
	{
		if(i==now)
		{
			continue;
		}
		temp=Point(ori,i,now,SIZE)/ii;
		for(int j=0;j<SIZE;j++)
		{
			Point(ori,i,j,SIZE)-=Point(ori,now,j,SIZE)*temp;
			Point(inv,i,j,SIZE)-=Point(inv,now,j,SIZE)*temp;
		}
	}
}

void Row_Normalize(double *ori,double *inv)
{
	for(int i=0;i<SIZE;i++)
	{
		double temp=1./Point(ori,i,i,SIZE);
		for(int j=0;j<SIZE;j++)
		{
			//Point(ori,i,j,SIZE)*=temp;
			Point(inv,i,j,SIZE)*=temp;
		}
	}
}

void Inverse_Matrix_Handle(double *ori, double *inv)
{
	for(int i=0;i<SIZE;i++)
	{
		Row_Function(ori,inv,i);
	}
	Row_Normalize(ori,inv);
}

void Inverse_Matrix_Kernel_Handle(double *ori, double *inv,dim3 Blocks_Per_Grid,dim3 Threads_Per_Block)
{
	for(int i=0;i<SIZE;i++)
	{
		Row_Kernel_Function<<<Blocks_Per_Grid,Threads_Per_Block>>>(ori,inv,i);
		cudaThreadSynchronize();
	}
	Row_Kernel_Normalize<<<Blocks_Per_Grid,Threads_Per_Block>>>(ori,inv);
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
				cout<<Point(mat, i, j, SIZE) << " ";
			}
			//if(int(Point(mat,i,j,SIZE))!=int((i==j)))
			//	cout<<i<<" "<<j<<" "<<float(Point(mat,i,j,SIZE))<<endl;
			if(j>10)break;//Leuckart
		}
		cout << endl;
		break;//Leuckart.
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
		mat[i] = rand() % 1000 * 0.01;
	}
}


void Matrix_Mult(double *a,double *b,double *res)
{
	for(int i=0;i<SIZE;i++)
	{
		for(int j=0;j<SIZE;j++)
		{
			double temp=0.0;
			for(int k=0;k<SIZE;k++)
			{
				temp+=Point(a,i,k,SIZE)*Point(b,k,j,SIZE);
			}
			Point(res,i,j,SIZE)=temp;
		}
	}
}

int main()
{
	unsigned int Byte_Size = SIZE * SIZE * sizeof(double);
	double *Matrix_Ori = (double *)malloc(Byte_Size);

	Initialize_Matrix(Matrix_Ori);
	//Show_Matrix(Matrix_Ori, "Original Matrix :");

	//cout << Get_Det(Matrix_Ori, SIZE) << endl;

	double *Matrix_Inv = (double *)malloc(Byte_Size);
	//Inverse_Matrix(Matrix_Ori, Matrix_Inv);
	//Show_Matrix(Matrix_Inv, "Inverse Matrix :");

	/* Initial Threads Blocks Begin */
	int thread_xdim = 32;
	int thread_ydim = 32;
	const dim3 Threads_Per_Block(thread_xdim, thread_ydim);
	const dim3 Blocks_Per_Grid(int((SIZE-1)/thread_xdim)+1, int((SIZE-1)/thread_ydim)+1);
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
		Cuda_Call(cudaMemcpy(Matrix_GPU, Matrix_Ori, Byte_Size, cudaMemcpyHostToDevice));
		Cuda_Call(cudaMemcpy(Matrix_Inv_GPU, ident, Byte_Size, cudaMemcpyHostToDevice));
		Inverse_Matrix_Kernel_Handle(Matrix_GPU,Matrix_Inv_GPU,Blocks_Per_Grid,Threads_Per_Block);
		Cuda_Call(cudaMemcpy(Matrix_Inv, Matrix_Inv_GPU, Byte_Size, cudaMemcpyDeviceToHost));
		Show_Matrix(Matrix_Inv,"");

		double *Matrix_Ori_Copy=(double *)malloc(Byte_Size);
		memcpy(Matrix_Ori_Copy,Matrix_Ori,Byte_Size);
		Inverse_Matrix_Handle(Matrix_Ori_Copy,ident);
		Show_Matrix(ident,"");

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
	//Cuda_Call(cudaMemcpy(Matrix_Ori, Matrix_GPU, Byte_Size, cudaMemcpyDeviceToHost));
	//Show_Matrix(Matrix_Ori,"...");
	/* Copy GPU To CPU End */

	//double *Matrix_Res = (double *)malloc(Byte_Size);
	//Matrix_Mult(Matrix_Ori,Matrix_Inv,Matrix_Res);
	//Matrix_Mult(Matrix_Ori,ident,Matrix_Res);
	//Show_Matrix(Matrix_Res, "Mult Matrix :");

	/*int flag;
	for(int i=0;i<SIZE;i++)
	{
		flag=0;
		for(int j=0;j<SIZE;j++)
		{
			if((Point(Matrix_Ori,i,j,SIZE)>0.99)&&(Point(Matrix_Ori,i,j,SIZE)<1.01))
			{
				flag++;
				if(i!=j)
				{
					cout<<"Error : "<<i<<" , "<<j<<endl;
				}
			}
		}
		if(flag>1)
		{
			cout<<"Error : "<<i<<endl;
		}
		cout<<flag<<endl;
	}*/

	/* Free Memory Begin */
	Cuda_Call(cudaFree(Matrix_GPU));
	free(Matrix_Ori);
	//free(Matrix_Res);
	/* Free Memory End */
	return 0;
}
