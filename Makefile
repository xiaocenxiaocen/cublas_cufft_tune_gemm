# Location of the CUDA Toolkit
# CUDA_PATH = /d0/data/zxd/mojoview/build/Linux.x86_64/opt/cuda
CUDA_PATH = /u/geoeast/geoeast3.1/support/cuda
CC = icc -O3 -g -Wall -std=c99 -xHost
CXX = icpc -O3 -g -Wall -std=c++0x -Wno-deprecated -xHost

#NVCC = nvcc -ccbin gcc -Xcompiler -fopenmp

NVCC = nvcc -ccbin icc -Xcompiler -openmp -Xcompiler -mkl

CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_COMMON_INCLUDE = /s0/u/zhangx/samples/common/inc
INCLUDES = -I$(CUDA_COMMON_INCLUDE) -I$(CUDA_INCLUDE) \
	-I/d0/data/zx/madagascar-1.6/include \
	-I/s0/GEOEAST/geoeast3.1/support/intel/composer_xe_2015.3.187/mkl/include \
	-I/s0/GEOEAST/geoeast3.1/support/intel/composer_xe_2015.3.187/mkl/include/fftw

GENCODE_FLAGS = -m64 -gencode arch=compute_20,code=sm_20
CUDA_FLAGS = -fmad=true --ptxas-options=-v -maxrregcount=63
CFLAGS = $(CUDA_FLAGS)

CXXFLAGS = -mkl -openmp -D_OPENMP -DHAVE_MKL $(INCLUDES)

LIBRARIES = -L/d0/data/zx/madagascar-1.6/lib -L/s0/GEOEAST/geoeast3.1/support/intel/composer_xe_2015.3.187/mkl/lib/intel64 -L$(CUDA_PATH)/lib64

LDFLAGS = -lrsf -lm -lpthread -lrsf++ -lcudart -lcublas

all: target

target: cumatmul mysgemm

cumatmul.o: cumatmul.cu
	$(NVCC) $(INCLUDES) $(CFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mysgemm.o: mygemm.cu
	$(NVCC) $(INCLUDES) $(CFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

cumatmul: cumatmul.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

mysgemm: mysgemm.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $<

.c.o:
	$(CC) -c $(CXXFLAGS) $<


.PHONY: clean
clean:
	-rm *.o
	-rm cumatmul
	-rm mysgemm
