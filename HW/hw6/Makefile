OBJS = radixsort.o clhelp.o

UNAME_S := $(shell uname -s)

#check if linux
ifeq ($(UNAME_S), Linux)
OCL_INC=/usr/local/cuda-6.5/include
OCL_LIB=/usr/local/cuda-6.5/lib64
%.o: %.cpp clhelp.h
	g++ -O2 -c $< -I$(OCL_INC)

all: $(OBJS)
	g++ radixsort.o clhelp.o -o radixsort -L$(OCL_LIB) -lOpenCL
endif

#check if os x
ifeq ($(UNAME_S), Darwin)
%.o: %.cpp clhelp.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ radixsort.o clhelp.o -o radixsort -framework OpenCL
endif

clean:
	rm -rf $(OBJS) radixsort
