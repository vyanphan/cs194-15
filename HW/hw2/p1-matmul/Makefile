OBJ = main.o matmul.o omp_matmul_for.o omp_matmul_task.o pthread_matmul.o

all: $(OBJ)
	g++ -pthread -fopenmp $(OBJ) -o matmul

%.o: %.cpp
	g++ -fopenmp -O3 -c $< 

clean:
	rm -rf matmul $(OBJ)

rebuild: clean all