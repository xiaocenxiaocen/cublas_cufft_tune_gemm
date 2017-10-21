include config.h

all: target

target: cumatmul mysgemm

#cumatmul.o: cumatmul.cu
#	$(NVCC) $(INCLUDES) $(CUDA_FLAGS) -o $@ -c $<
#
#mysgemm.o: mygemm.cu
#	$(NVCC) $(INCLUDES) $(CUDA_FLAGS) -o $@ -c $<

cumatmul: cumatmul.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

mysgemm: mysgemm.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $<

.c.o:
	$(CC) -c $(CXXFLAGS) $<


%.o: %.cu
	$(NVCC) -c $(INCLUDES) $(CUDA_FLAGS) -o $@ -c $<

.PHONY: clean
clean:
	-rm *.o
	-rm cumatmul
	-rm mysgemm
