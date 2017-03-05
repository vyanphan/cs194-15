OBJS = vvadd.o matmul.o sqr_sgemm.o clhelp.o incr.o
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Linux)
OCL_INC=/usr/local/cuda-6.5/include
OCL_LIB=/usr/local/cuda-6.5/lib64

%.o: %.cpp clhelp.h
	g++ -msse4 -O2 -c $< -I$(OCL_INC)

all: $(OBJS)
	g++ incr.o clhelp.o -o incr -L$(OCL_LIB) -lOpenCL
	g++ vvadd.o clhelp.o -o vvadd -L$(OCL_LIB) -lOpenCL
	g++ matmul.o clhelp.o sqr_sgemm.o -o matmul -L$(OCL_LIB) -lOpenCL
endif

ifeq ($(UNAME_S), Darwin)
%.o: %.cpp clhelp.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ incr.o clhelp.o -o incr -framework OpenCL
	g++ vvadd.o clhelp.o -o vvadd -framework OpenCL
	g++ matmul.o clhelp.o sqr_sgemm.o -o matmul -framework OpenCL
endif


clean:
	rm -rf $(OBJS) matmul vvadd incr
