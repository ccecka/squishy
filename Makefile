#
# A simple makefile for CUDA projects.
#

# Get the host-name if empty
ifeq ($(host-name),)
	host-name := $(shell hostname)
endif
# Get the kernel-name if empty
ifeq ($(kernel-name),)
	kernel-name := $(shell uname -s)
endif
# Get the arch-name if empty
ifeq ($(arch-name),)
	arch-name := $(shell uname -p)
endif

# Compilers
NVCC       := nvcc
CXX        := g++

# Executable to build
EXEC            := FeCalc

############################################################
####################
# Rules and targets

CUDA_PATH := /usr/local/cuda

# Define any directories containing header files
#   To include directories use -Ipath/to/files
INCLUDES += -I. -I$(CUDA_PATH)/include

# Define cxx compile flags
CXXFLAGS  = -fopenmp -funroll-loops -O3 -W -Wall -Wextra

# Define nvcc compile flags   TODO: Detect and generate appropriate sm_XX
NVCCFLAGS := -ccbin=g++-4.6 -arch=sm_20 -O3 --compiler-options "$(CXXFLAGS)" #-Xptxas="-v"

# Define any directories containing libraries
#   To include directories use -Lpath/to/files
LDFLAGS +=
# Use cuda lib64 if it exists, else cuda lib
ifneq ($(wildcard $(CUDA_PATH)/lib64/.*),)
	LDFLAGS += -L$(CUDA_PATH)/lib64
else
	LDFLAGS += -L$(CUDA_PATH)/lib
endif

# Define any libraries to link into executable
#   To link in libraries (libXXX.so or libXXX.a) use -lXXX
LDLIBS  += -lm -lmetis -lcudart -lcublas -lGLEW
ifeq ($(kernel-name),Darwin)
	LDLIBS += -framework GLUT -framework OpenGL -framework carbon
endif
ifeq ($(kernel-name),Linux)
	LDLIBS += -lcublas -lGL -lGLU -lglut -lGLEW
endif

# Dependency directory and flags
DEPSDIR := $(shell mkdir -p .deps; echo .deps)
# MD: Dependency as side-effect of compilation
# MF: File for output
# MP: Include phony targets
DEPSFILE = $(DEPSDIR)/$(notdir $*.d)
DEPSFLAGS = -MD -MF $(DEPSFILE) #-MP

####################
## Makefile Rules ##
####################

# Suffix replacement rules
#   $^: the name of the prereqs of the rule
#   $<: the name of the first prereq of the rule
#   $@: the name of the target of the rule
.SUFFIXES:                       # Delete the default suffixes
.SUFFIXES: .h .hpp .cpp .cu .o   # Define our suffix list

# 'make' - default rule
all: $(EXEC)

# Default rule for creating an exec of $(EXEC) from a .o file
$(EXEC): % : %.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Default rule for creating a .o file from a .cpp file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPSFLAGS) -c -o $@ $<

# Default rule for creating a .o file from a .cu file
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -M -o $(DEPSFILE) $<

# 'make clean' - deletes all .o and temp files, exec, and dependency file
clean:
	-$(RM) *.o
	-$(RM) $(EXEC)
	$(RM) -r $(DEPSDIR)

# Define rules that do not actually generate the corresponding file
.PHONY: clean all

# Include the dependency files
-include $(wildcard $(DEPSDIR)/*.d)
