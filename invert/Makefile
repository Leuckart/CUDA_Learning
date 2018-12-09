ARCH=-gencode arch=compute_52,code=compute_52

VPATH=./
SLIB=libfantasynet.so
ALIB=libfantasynet.a
EXEC=fantasynet
OBJDIR=./

CC=gcc
CPP=g++
NVCC=nvcc
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS=-lm #-lcurand -lcudnn -lcublas#-pthread
COMMON=-Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

# OPENMP=0
# DEBUG=0

CFLAGS+=$(OPTS)

# OPENCV=0
# GPU=1
COMMON+=-DGPU -I/usr/local/cuda-9.1/include/
CFLAGS+=-DGPU
LDFLAGS+=-L/usr/local/cuda-9.1/lib64/ -lcuda -lcudart -lcublas -lcurand

# CUDNN=1
COMMON+=-DCUDNN
CFLAGS+=-DCUDNN
LDFLAGS+=-lcudnn

# TO BE CONTINUED
OBJ=cuda.o utils.o list.o blas.o blas_kernels.o
#EXECOBJA=

LDFLAGS+=-lstdc++
#OBJ+=

# EXECOBJ
# OBJS
# DEPS

#all:
#	$(NVCC) $(ARCH) -w -o output *.cu $(LDFLAGS)
all:
	g++ -o invert *.cpp
