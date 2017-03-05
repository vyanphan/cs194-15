#include <omp.h>
#include <cstdio>

void do_mv(double **a, double **b, double **c, int i) {
  for(int j=0;j<1024;j++)
    for(int k=0;k<1024;k++)
      c[i][j] += a[i][k]*b[k][j];
      
}

void omp_task_matmuld(double **a, double **b, double **c, int nthr) {
  omp_set_num_threads(nthr);    //this call is needed to set the number of threads


  #pragma omp parallel
  {
    #pragma omp single
    {  
      for(int i=0;i<1024;i++) {     //CS194: add pragmas to this loop-nest to enable OpenMP task parallelism 
          #pragma omp task        
          {
            do_mv(a,b,c,i);
          }
      }
    }
  }


 
}
