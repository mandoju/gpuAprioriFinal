CUB_HOME ?= /home/jorge/TCC/aprioriGitHub/cub/
CUDA = 1

NVCC_FLAGS =-arch=sm_35

ifeq (${CUDA},1)
NVCC_FLAGS +=-D__CUDA__=1
endif

ifeq (${VERBOSE},1)
NVCC_FLAGS += -Xptxas -v,-abi=no
endif

ifeq (${DEBUG},1)
NVCC_FLAGS += -D__DEBUG__=1 -g -G
endif

CCFLAGS = -Xcompiler -fopenmp

INCLUDES = -I$(CUB_HOME)

apriori: apriori.o
	nvcc $(NVCC_FLAGS) $(CCFLAGS) -o apriori apriori.o -lboost_system

%.o: %.cu
	nvcc -c $(NVCC_FLAGS) $(CCFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm *.o apriori
