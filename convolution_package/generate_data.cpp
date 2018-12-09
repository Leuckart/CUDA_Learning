/**************************************************
	> File Name:  generate_data.cpp
	> Author:     Leuckart
	> Time:       2018-10-08 15:29
**************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float randoming()
{
	int rand_num=rand();
	return 10*(float(rand_num)/RAND_MAX);
	//return (float(rand_num)/RAND_MAX)*2.-1.;
}

int main()
{
	int Input_Size_X,Input_Size_Y,Input_Channel,Output_Channel;
	int Kernel_Size,Strides;

	/* Initial Config Begin */
	FILE *f_config=fopen("./Config.txt","r");
	if(f_config==NULL)
	{
		printf("Config File Read Error.\n");
		exit(1);
	}
	fscanf(f_config,"%d\n",&Input_Size_X);
	fscanf(f_config,"%d\n",&Input_Size_Y);
	fscanf(f_config,"%d\n",&Input_Channel);
	fscanf(f_config,"%d\n",&Output_Channel);
	fscanf(f_config,"%d\n",&Kernel_Size);
	fscanf(f_config,"%d\n",&Strides);
	fclose(f_config);
	/* Initial Config End */

	/* Initial Random Seed Begin */
	srand((int)time(0));
	/* Initial Random Seed End */

	/* Initial Input Begin */
	FILE *f_input=fopen("./Input.txt","w+");
	if(f_input==NULL)
	{
		printf("Input File Write Error.\n");
		exit(1);
	}
	for(int i=0;i<Input_Size_X;++i)
	{
		for(int j=0;j<Input_Size_Y;++j)
		{
			for(int k=0;k<Input_Channel;++k)
			{
				fprintf(f_input,"%f\n",randoming());
			}
		}
	}
	fclose(f_input);
	/* Initial Input Begin */

	/* Initial Kernel Begin */
	FILE *f_kernel=fopen("./Kernel.txt","w+");
	if(f_kernel==NULL)
	{
		printf("Kernel File Write Error.\n");
		exit(1);
	}
	for(int i=0;i<Kernel_Size;++i)
	{
		for(int j=0;j<Kernel_Size;++j)
		{
			for(int k=0;k<Input_Channel;++k)
			{
				for(int l=0;l<Output_Channel;++l)
				{
					fprintf(f_kernel,"%f\n",randoming());
				}
			}
		}
	}
	fclose(f_kernel);
	/* Initial Kernel End */

	return 0;
}
