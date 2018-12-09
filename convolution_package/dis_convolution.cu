/**************************************************
	> File Name:  Convolution_CUDA.cu
	> Author:     Leuckart
	> Time:       2018-09-25 15:25
**************************************************/

/* Config.txt :: Input_Size_X \n Input_Size_Y \n  Input_Channel \n Output_Channel \n Kernel_Size \n Strides */
/* Input.txt :: No Description */

/* Input Format :: Input_Size * Input_Size * Input_Channel */
/* Kernel Format :: Kernel_Size * Kernel_Size * Input_Channel * Output_Channel */
/* Output Format :: Output_Size * Output_Size * Output_Channel */

#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>

#define Cuda_Call(x) {const cudaError_t a=(x);if(a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num= %d)\n",cudaGetErrorString(a),a);cudaDeviceReset();exit(1);}}

__global__ void kernel_convolution(float *image,float *kernel,float *result,int input_size_x,int input_size_y,int kernel_size,int input_channel,int output_channel,int strides)
{
	int output_size_x=int((input_size_x-kernel_size)/strides)+1;
	int output_size_y=int((input_size_y-kernel_size)/strides)+1;
	const unsigned int _idx=(blockIdx.x*blockDim.x)+threadIdx.x;
	const unsigned int _idy=(blockIdx.y*blockDim.y)+threadIdx.y;
	const unsigned int index=((gridDim.x*blockDim.x)*_idy)+_idx;

	const int Idx=(index/(output_size_y*output_channel))%output_size_x;//?
	const int Idy=(index/output_channel)%output_size_y;
	const int Idc=index%output_channel;

	if(index>=output_size_x*output_size_y*output_channel)
	{
		return;
	}

	for(int dx=0;dx<kernel_size;++dx)
	{
		for(int dy=0;dy<kernel_size;++dy)
		{
			for(int dint=0;dint<input_channel;++dint)
			{
				int kernel_pos1=dx*kernel_size*input_channel*output_channel;
				int kernel_pos2=dy*input_channel*output_channel+dint*output_channel;

				result[index]+=kernel[kernel_pos1+kernel_pos2+Idc]*image[((Idx*strides+dx)*input_size_y+Idy*strides+dy)*input_channel+dint];
			}
		}
	}
}

int main()
{
	int Input_Size_X,Input_Size_Y,Input_Channel,Output_Channel;
	int Kernel_Size,Strides;

	/* Initial Config Begin */
	FILE *f_config=fopen("./Config.txt","r");
	fscanf(f_config,"%d\n",&Input_Size_X);
	fscanf(f_config,"%d\n",&Input_Size_Y);
	fscanf(f_config,"%d\n",&Input_Channel);
	fscanf(f_config,"%d\n",&Output_Channel);
	fscanf(f_config,"%d\n",&Kernel_Size);
	fscanf(f_config,"%d\n",&Strides);
	fclose(f_config);
	/* Initial Config End */

	/* Compute Other Parameters Begin */
	int Output_Size_X=int((Input_Size_X-Kernel_Size)/Strides)+1;
	int Output_Size_Y=int((Input_Size_Y-Kernel_Size)/Strides)+1;
	int Input_Byte=Input_Size_X*Input_Size_Y*Input_Channel*sizeof(float);
	int Kernel_Byte=Kernel_Size*Kernel_Size*Input_Channel*Output_Channel*sizeof(float);
	int Output_Byte=Output_Size_X*Output_Size_Y*Output_Channel*sizeof(float);
	/* Compute Other Parameters End */

	/* Initial Threads Blocks Begin */
	int thread_xdim=32;
	int thread_ydim=32;
	const dim3 Threads_Per_Block(thread_xdim,thread_ydim);
	const dim3 Blocks_Per_Grid(int((Output_Channel*Output_Size_X-1)/Threads_Per_Block.x)+1,int((Output_Size_Y-1)/Threads_Per_Block.y)+1);// Need To Dec & Add 1 ?
	/* Initial Threads Blocks End */

	/* Initial Memory Begin */
	float *CPU_Input=(float *)malloc(Input_Byte);
	float *CPU_Kernel=(float *)malloc(Kernel_Byte);
	float *CPU_Output=(float *)malloc(Output_Byte);
	float *GPU_Input;
	float *GPU_Kernel;
	float *GPU_Output;
	Cuda_Call(cudaMalloc((void **)&GPU_Input,Input_Byte));
	Cuda_Call(cudaMalloc((void **)&GPU_Kernel,Kernel_Byte));
	Cuda_Call(cudaMalloc((void **)&GPU_Output,Output_Byte));
	/* Initial Memory Begin */

	/* Initial Input Begin */
	FILE *f_input=fopen("./Input.txt","r");
	float _temp=0;
	for(int i=0;i<Input_Size_X;++i)
	{
		for(int j=0;j<Input_Size_Y;++j)
		{
			for(int k=0;k<Input_Channel;++k)
			{
				fscanf(f_input,"%f\n",&_temp);
				CPU_Input[i*Input_Size_Y*Input_Channel+j*Input_Channel+k]=_temp;
			}
		}
	}
	fclose(f_input);
	/* Initial Input End */

	/* Initial Kernel Begin */
	FILE *f_kernel=fopen("./Kernel.txt","r");
	for(int i=0;i<Kernel_Size;++i)
	{
		for(int j=0;j<Kernel_Size;++j)
		{
			for(int k=0;k<Input_Channel;++k)
			{
				for(int l=0;l<Output_Channel;++l)
				{
					fscanf(f_kernel,"%f\n",&_temp);
					CPU_Kernel[i*(Kernel_Size*Input_Channel*Output_Channel)+j*(Input_Channel*Output_Channel)+k*(Output_Channel)+l]=_temp;
				}
			}
		}
	}
	fclose(f_kernel);
	/* Initial Kernel End */

	/* Copy CPU To GPU Begin */
	Cuda_Call(cudaMemcpy(GPU_Input,CPU_Input,Input_Byte,cudaMemcpyHostToDevice));
	Cuda_Call(cudaMemcpy(GPU_Kernel,CPU_Kernel,Kernel_Byte,cudaMemcpyHostToDevice));
	Cuda_Call(cudaMemcpy(GPU_Output,CPU_Output,Output_Byte,cudaMemcpyHostToDevice));
	/* Copy CPU To GPU End */

	/* Test On Every Device Begin */
	int Device_All;
	Cuda_Call(cudaGetDeviceCount(&Device_All));
	/* Test On Every Device End */

	for(int device_number=0;device_number<Device_All;++device_number)
	{
		/* Set Device Parameters Begin */
		Cuda_Call(cudaSetDevice(device_number));
		struct cudaDeviceProp device_prop;
		char device_prefix[100];
		Cuda_Call(cudaGetDeviceProperties(&device_prop,device_number));
		sprintf(device_prefix,"ID: %d %s: ",device_number,device_prop.name);
		/* Set Device Parameters End */

		/* Initial Time Block Begin */
		cudaEvent_t kernel_start,kernel_stop;
		float delta_time=0.;
		Cuda_Call(cudaEventCreate(&kernel_start));
		Cuda_Call(cudaEventCreateWithFlags(&kernel_stop,cudaEventBlockingSync));
		Cuda_Call(cudaEventRecord(kernel_start,0));
		/* Initial Time Block End */

		/* Kernel Function Execute Begin */
		kernel_convolution<<<Blocks_Per_Grid,Threads_Per_Block>>>(GPU_Input,GPU_Kernel,GPU_Output,Input_Size_X,Input_Size_Y,Kernel_Size,Input_Channel,Output_Channel,Strides);
		/* Kernel Function Execute End */

		/* Time Clock Begin */
		Cuda_Call(cudaEventRecord(kernel_stop,0));
		Cuda_Call(cudaEventSynchronize(kernel_stop));
		Cuda_Call(cudaEventElapsedTime(&delta_time,kernel_start,kernel_stop));
		printf("%s %.5fms\n",device_prefix,delta_time);
		Cuda_Call(cudaEventDestroy(kernel_start));
		Cuda_Call(cudaEventDestroy(kernel_stop));
		/* Time Clock End */
	}

	/* Copy GPU To CPU Begin */
	Cuda_Call(cudaMemcpy(CPU_Input,GPU_Input,Input_Byte,cudaMemcpyDeviceToHost));
	Cuda_Call(cudaMemcpy(CPU_Kernel,GPU_Kernel,Kernel_Byte,cudaMemcpyDeviceToHost));
	Cuda_Call(cudaMemcpy(CPU_Output,GPU_Output,Output_Byte,cudaMemcpyDeviceToHost));
	/* Copy GPU To CPU End */

	/* Print Result Begin */
	FILE *f_output=fopen("./output_cuda.txt","w+");
	if(f_output==NULL)
	{
		printf("Output File Write Error.\n");
		exit(1);
	}
	for(int i=0;i<Output_Size_X;++i)
	{
		for(int j=0;j<Output_Size_Y;++j)
		{
			//printf("%d , %d  ::  ",i,j);
			for(int k=0;k<Output_Channel;++k)
			{
				//printf("%d  ",CPU_Output[i*Output_Size_Y*Output_Channel+j*Output_Channel+k]);
				fprintf(f_output,"%f\n",CPU_Output[i*Output_Size_Y*Output_Channel+j*Output_Channel+k]);
			}
			//printf("\n");
		}
	}
	fclose(f_output);
	/* Print Result End */

	/* Free Memory Begin */
	Cuda_Call(cudaFree(GPU_Input));
	Cuda_Call(cudaFree(GPU_Kernel));
	Cuda_Call(cudaFree(GPU_Output));
	free(CPU_Input);
	free(CPU_Kernel);
	free(CPU_Output);
	/* Free Memory End */

	return 0;
}
