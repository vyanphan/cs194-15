#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>


/* helper functions */
double timestamp();
void zero_matrix(double **m);
void rand_matrix(double **m);
double report_mflops(double t);
bool compare(double **a, double **b);
void print_csv(double *r, int n, std::string fname);

/* serial matrix multiply */
void matmuld(double **a,
	     double **b,
	     double **c);

/* omp task matrix multiply */
void omp_task_matmuld(double **a,
		      double **b,
		      double **c,
		      int nthr);

void omp_for_matmuld(double **a,
		     double **b,
		     double **c,
		     int nthr);

void pthread_matmuld(double **a,
		     double **b,
		     double **c,
		     int nthr);

int main(int argc, char *argv[])
{
  double **a, **b, **c, **s;
  a = new double*[1024];
  b = new double*[1024];
  c = new double*[1024];
  s = new double*[1024];

  int nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  printf("This machine has hardware %d threads\n", nprocs);

  double *omp_for_results = new double[nprocs];
  double *omp_task_results = new double[nprocs];
  double *pthread_results = new double[nprocs];


  for(int i = 0; i < 1024; i++) {
    a[i] = new double[1024];
    b[i] = new double[1024];
    c[i] = new double[1024];
    s[i] = new double[1024];
  }
  double t0=0.0;


  double serial_time=0.0;
  double omp_task_time = 0.0;
  double omp_for_time = 0.0;
  double pthread_time = 0.0;


  rand_matrix(a);
  rand_matrix(b);
  zero_matrix(c);
  zero_matrix(s);

  /* serial */
  serial_time = timestamp();
  matmuld(a,b,s);
  serial_time = (timestamp()-serial_time);
  printf("serial mflop/s = %f\n", report_mflops(serial_time));
  
  
  /* omp for */
  printf("Running OpenMP for:\n");
  for(int i = 1; i <= nprocs; i++) {
    zero_matrix(c);
    omp_for_time = timestamp();
    omp_for_matmuld(a,b,c,i);
    omp_for_time = (timestamp()-omp_for_time);
    if(compare(c,s)==false) {
      printf("wrong answer!\n");
	    break;
    }

    omp_for_results[i-1] = report_mflops(omp_for_time);
    printf("%d, %f\n", i, report_mflops(omp_for_time));
  }

  /* omp task */
  printf("Running OpenMP task:\n");
  for(int i = 1; i <= nprocs; i++) {
      zero_matrix(c);
      omp_task_time = timestamp();
      omp_task_matmuld(a,b,c,i);
      omp_task_time = (timestamp()-omp_task_time);
      if(compare(c,s)==false){
        printf("wrong answer!\n");
        break;
      }
      omp_task_results[i-1] =	report_mflops(omp_task_time);

      printf("%d, %f\n", i, report_mflops(omp_task_time));
  }


  /* pthreads */
  printf("Running pThreads:\n");
  for(int i = 1; i <= nprocs; i++) {
    zero_matrix(c);
    pthread_time = timestamp();
    pthread_matmuld(a,b,c,i);
    pthread_time = (timestamp()-pthread_time);
    if(compare(c,s)==false)	{
      printf("wrong answer!\n");
      break;
    }
    pthread_results[i-1] = report_mflops(pthread_time);
    printf("%d, %f\n", i, report_mflops(pthread_time));
  }

  for(int i = 0; i < 1024; i++) {
    delete [] a[i];
    delete [] b[i];
    delete [] c[i];
    delete [] s[i];
  }
  delete [] a;
  delete [] b;
  delete [] c;
  delete [] s;

  print_csv(omp_for_results,nprocs, "omp_for.csv");
  print_csv(omp_task_results,nprocs, "omp_task.csv");
  print_csv(pthread_results,nprocs, "pthread_results.csv");

  delete [] omp_for_results;
  delete [] omp_task_results;
  delete [] pthread_results;
  return 0;
}



double timestamp() {
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

void zero_matrix(double **m) {
  for(int i = 0; i < 1024; i++) {
    for(int j = 0; j < 1024; j++)
      m[i][j] = 0.0;
  }
}

void rand_matrix(double **m) {
  for(int i = 0; i < 1024; i++) {
    for(int j = 0; j < 1024; j++)
      m[i][j] = 1000.0 * drand48();
  }
}

double report_mflops(double t) {
  double n = 1024.0;
  double f = (2.0*(n*n*n)) / ((1024.0*1024.0)*t);
  return f;
}

bool compare(double **a, double **b) {
  for(int i=0;i<1024;i++) {
    for(int j=0;j<1024;j++) {
      double d = a[i][j]-b[i][j];
      d *= d;
      if(d > 0.00001) {
        return false;
      }
    }
  }
  return true;
}

void print_csv(double *r, int n, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "w");
  for(int i = 0; i < n; i++) {
    fprintf(fp,"%d,%f\n",i+1,r[i]);
  }
  fclose(fp);
}
