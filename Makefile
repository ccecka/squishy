#
# A simple makefile for CUDA projects.
#

# Executable to build
EXEC            := FeCalc
# Cuda source files (compiled with nvcc)
CUFILES         := FeCalc.cu
# C/C++ source files
CCFILES         :=
CCOFILES        := $(CCFILES:%.cpp=%.o) $(CUFILES:%.cu=%.o)

OPTS := -O3 --gpu-architecture sm_30 --ptxas-options="-v"
#OPTS := -O0 -g -Xcompiler -D__builtin_stdarg_start=__builtin_va_start

############################################################
####################
# Rules and targets

CUDA_PATH := /usr/local/cuda

# Basic directory setup for SDK
# (override directories only if they are not already defined)
CUDA_INC     := $(CUDA_PATH)/include
CUDA_LIB     := $(CUDA_PATH)/lib64

# Compilers
NVCC       := nvcc -ccbin /usr/bin/g++-4.6
CXX        := g++
LINK       := nvcc

# Includes
CU_INCL    += -I$(CUDA_INC)
C_INCL     += -I$(CUDA_INC)
# Libs
LIB      := -L$(CUDA_LIB)
LIB_L    := -lm -lmetis -lcublas -lGL -lGLU -lglut -lGLEW


default: $(CCOFILES)
	$(LINK) -o $(EXEC) $(OPTS) $(CCOFILES) $(LIB) $(LIB_L)

run:
	./$(EXEC)

%.o: %.cpp
	$(CXX) $(OPTS) -c $(C_INCL) $<

%.o: %.cu
	$(NVCC) $(OPTS) -c $(CU_INCL) $<

clean:
	rm -f $(EXEC) $(CCOFILES) *~ ./*/*~

all:
	make clean;
	make;
	make run;
