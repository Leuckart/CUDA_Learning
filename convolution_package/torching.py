import numpy as np
import torch.nn as nn

from torch.autograd import Variable
import torch

def main():
    ''' Initial Config Begin '''
    Config=[]
    with open("./Config.txt","r") as f:
        for line in f.readlines():
            line=line.strip("\n")
            Config.append(int(line))
    Input_Size_X,Input_Size_Y,Input_Channel,Output_Channel,Kernel_Size,Strides=Config
    ''' Initial Config End '''

    ''' Initial Input Begin '''
    _input=[]
    with open("./Input.txt") as f:
        for line in f.readlines():
            line=line.strip("\n")
            _input.append(float(line))
    Input=torch.zeros(1,Input_Channel,Input_Size_X,Input_Size_Y)
    _i=0
    for i in range(Input_Size_X):
        for j in range(Input_Size_Y):
            for k in range(Input_Channel):
                Input[0][k][i][j]=_input[_i]
                _i+=1
    ''' Initial Input End '''

    ''' Initial Layer Begin '''
    Layer=nn.Conv2d(in_channels=Input_Channel,out_channels=Output_Channel,kernel_size=Kernel_Size,stride=Strides,padding=0)
    ''' Initial Layer End '''

    ''' Initial Kernel Begin '''
    _kernel=[]
    with open("./Kernel.txt") as f:
        for line in f.readlines():
            line=line.strip("\n")
            _kernel.append(float(line))
    _i=0
    Kernel=np.zeros((Output_Channel,Input_Channel,Kernel_Size,Kernel_Size))
    for i in range(Kernel_Size):
        for j in range(Kernel_Size):
            for k in range(Input_Channel):
                for l in range(Output_Channel):
                    Kernel[l,k,i,j]=_kernel[_i]
                    Layer.weight[l,k,i,j]=_kernel[_i]
                    _i+=1
    ''' Initial Kernel End '''

    ''' Initial Bias Begin '''
    for i in range(Output_Channel):
        Layer.bias[i]=0.
    ''' Initial Bias End '''

    ''' PyTorch Convolution Layer Begin '''
    Output_Pytorch=Layer(Input)
    ''' PyTorch Convolution Layer End '''

    ''' Set Output Size Begin '''
    Output_Size_X=int((Input_Size_X-Kernel_Size)/Strides)+1
    Output_Size_Y=int((Input_Size_Y-Kernel_Size)/Strides)+1
    Pad_Size=int((Kernel_Size-1)/2)
    Output_Python=np.zeros((1,Output_Channel,Output_Size_X,Output_Size_Y))
    ''' Set Output Size End '''

    ''' Compute Convolution Begin '''
    Input_Numpy=Input.data.numpy()
    for i in range(Output_Channel):
        for j in range(Output_Size_X):
            for k in range(Output_Size_Y):
                value=np.sum(Input_Numpy[0,:,j*Strides:j*Strides+Kernel_Size,k*Strides:k*Strides+Kernel_Size]*Kernel[i,:,:,:])
                Output_Python[0,i,j,k]=value+Layer.bias[i].data.numpy()

                ''' Terrible Float Error '''
                if (Output_Python[0,i,j,k]-Output_Pytorch[0,i,j,k].data.numpy())/Output_Pytorch[0,i,j,k].data.numpy()>=0.0001:
                    print(i,j,k,Output_Python[0,i,j,k],Output_Pytorch[0,i,j,k].data.numpy(),"Error")
                #print(Output_Python[0,i,j,k],Output_Pytorch[0,i,j,k].data.numpy())
    ''' Compute Convolution End '''

    ''' Print Begin '''
    writer=open("./output_torch.txt","w")
    for i in range(Output_Size_X):
        for j in range(Output_Size_Y):
            #print(Output_Pytorch[0,:,i,j].data.numpy())
            for k in range(Output_Channel):
                writer.write(str(float(Output_Pytorch[0,k,i,j])))
                #writer.write(str(float(Output_Python[0,k,i,j])))
                writer.write("\n")
            #print(Output_Python[0,:,i,j])
    writer.close()
    ''' Print End '''

if __name__=="__main__":
    main()
