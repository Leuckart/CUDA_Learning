def main():
    Cuda=[]
    Torch=[]
    Config=[]

    with open("./output_cuda.txt","r") as f:
        for line in f.readlines():
            line=line.strip("\n")
            Cuda.append(float(line))

    with open("./output_torch.txt","r") as f:
        for line in f.readlines():
            line=line.strip("\n")
            Torch.append(float(line))

    with open("./Config.txt","r") as f:
        for line in f.readlines():
            line=line.strip("\n")
            Config.append(int(line))

    Input_Size_X,Input_Size_Y,Input_Channel,Output_Channel,Kernel_Size,Strides=Config
    Output_Size_X=int((Input_Size_X-Kernel_Size)/Strides)+1
    Output_Size_Y=int((Input_Size_Y-Kernel_Size)/Strides)+1

    _i=0
    for i in range(Output_Size_X):
        for j in range(Output_Size_Y):
            for k in range(Output_Channel):
                value_cuda=Cuda[_i]
                value_torch=Torch[_i]
                _i+=1
                if (value_cuda-value_torch)/value_torch>=0.0001:
                    print(i,j,k," : ",value_cuda,value_torch)


if __name__=="__main__":
    main()
