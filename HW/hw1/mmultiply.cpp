#include <cstdio>
#include <cstring>
#include <cstdlib>

void opt_simd_sgemm(float *Y, float *A, float *B, int n);
void opt_scalar1_sgemm(float *Y, float *A, float *B, int n);
void opt_scalar0_sgemm(float *Y, float *A, float *B, int n);
void naive_sgemm(float *Y, float *A, float *B, int n);

int main(int argc, char *argv[])
{
  int n = (1<<10);
  float* A = new float[n*n];
  float* B = new float[n*n];
  float* Y = new float[n*n];
  naive_sgemm(Y, A, B, n);
  
  delete [] A;
  delete [] B;
  delete [] Y;
}
