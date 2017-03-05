//void matmulf(float a [N][N], float b [N][N], float c[N][N])
#include <smmintrin.h>

#define fBlkI 128
#define fBlkJ 128
#define fBlkK 128

#define fBlkII 8
#define fBlkJJ 8
#define fBlkKK 8


void sqr_sgemm(float *Y, float *A, float *B, int n)
{
  int i,j,k;
  int ii,jj,kk;
  int iii,jjj,kkk;

  float bt[fBlkK][fBlkK];
  
  float vC[4];
  
  __m128 sA0, sA1;
  __m128 sA2, sA3;
  
  __m128 sB0, sB1;
  __m128 sB2, sB3;
  __m128 sC;

  __m128 sT[16];

  for(i=0;i<n;i+=fBlkI)
    {
      for(j=0;j<n;j+=fBlkJ)
	{
	  for(k=0;k<n;k+=fBlkK)
	    {
	      /* transpose block */
	      for(jj=0;jj<fBlkJ;jj++)
		{
		  for(kk=0;kk<fBlkK;kk++)
		    {
		      bt[jj][kk] = B[(k + kk)*n + j + jj];
		    }
		}

	      for(ii=0;ii<fBlkI;ii+=4)
		{
		  for(jj=0;jj<fBlkJ;jj+=4)
		    {
		      for(kk=0;kk<16;kk++)
			{
			  sT[kk] = _mm_set_ps(0.0f,0.0f,0.0f,0.0f);
			}
		      		      
		      for(kk=0;kk<fBlkK;kk+=4)
			{
			  /* load b vectors */
			  sB0 = _mm_load_ps(&bt[jj+0][kk]);
			  sB1 = _mm_load_ps(&bt[jj+1][kk]);
			  sB2 = _mm_load_ps(&bt[jj+2][kk]);
			  sB3 = _mm_load_ps(&bt[jj+3][kk]);
			  

			  sA0 = _mm_load_ps(&A[(i+ii+0)*n+k+kk]);
			  sT[0] = _mm_add_ps(sT[0], _mm_mul_ps(sA0, sB0));
			  sT[1] = _mm_add_ps(sT[1], _mm_mul_ps(sA0, sB1));
			  sT[2] = _mm_add_ps(sT[2], _mm_mul_ps(sA0, sB2));
			  sT[3] = _mm_add_ps(sT[3], _mm_mul_ps(sA0, sB3));
			  
			  sA1 = _mm_load_ps(&A[(i+ii+1)*n+k+kk]);
			  sT[4] = _mm_add_ps(sT[4], _mm_mul_ps(sA1, sB0));
			  sT[5] = _mm_add_ps(sT[5], _mm_mul_ps(sA1, sB1));
			  sT[6] = _mm_add_ps(sT[6], _mm_mul_ps(sA1, sB2));
			  sT[7] = _mm_add_ps(sT[7], _mm_mul_ps(sA1, sB3));

			  sA2 = _mm_load_ps(&A[(i+ii+2)*n+k+kk]);
			  sT[8] = _mm_add_ps(sT[8], _mm_mul_ps(sA2, sB0));
			  sT[9] = _mm_add_ps(sT[9], _mm_mul_ps(sA2, sB1));
			  sT[10] = _mm_add_ps(sT[10], _mm_mul_ps(sA2, sB2));
			  sT[11] = _mm_add_ps(sT[11], _mm_mul_ps(sA2, sB3));

			  sA3 = _mm_load_ps(&A[(i+ii+3)*n+k+kk]);
			  sT[12] = _mm_add_ps(sT[12], _mm_mul_ps(sA3, sB0));
			  sT[13] = _mm_add_ps(sT[13], _mm_mul_ps(sA3, sB1));
			  sT[14] = _mm_add_ps(sT[14], _mm_mul_ps(sA3, sB2));
			  sT[15] = _mm_add_ps(sT[15], _mm_mul_ps(sA3, sB3));
			}

		      for(iii=0;iii<4;iii++)
			{
			  for(jjj=0;jjj<4;jjj++)
			    {
			      _mm_store_ps(vC, sT[iii*4+jjj]);
			      Y[(iii+ii+i)*n+jjj+jj+j] += (vC[0]+vC[1]+vC[2]+vC[3]);
			    }
			}
		      
		    }
		}
	    }
	}
    }
}

/*
void sqr_sgemm(float *Y, float *A, float *B, int n)
{
  int i,j,k;
  int ii,jj,kk;
  int iii,jjj,kkk;

  float bt[fBlkK][fBlkK];
  float vC[fBlkKK];
  
  for(i=0;i<n;i+=fBlkI)
    {
      for(j=0;j<n;j+=fBlkJ)
	{
	  for(k=0;k<n;k+=fBlkK)
	    {
	    for(jj=0;jj<fBlkJ;jj++)
		{
		  for(kk=0;kk<fBlkK;kk++)
		    {
		      bt[jj][kk] = B[(k + kk)*n + j + jj];
		    }
		}

	      for(ii=0;ii<fBlkI;ii+=fBlkII)
		{
		  for(jj=0;jj<fBlkJ;jj+=fBlkJJ)
		    {
		      for(kk=0;kk<fBlkK;kk+=fBlkKK)
			{
			  for(iii=0;iii<fBlkII;iii++)
			    {
			      for(jjj=0;jjj<fBlkJJ;jjj++)
				{
				  for(kkk=0;kkk<fBlkKK;kkk++)
				    {
				      Y[(i+ii+iii)*n + j+jj+jjj] += 
					A[(i+ii+iii)*n + k+kk+kkk]*bt[jj+jjj][kk+kkk];
				    }
				}
			    }
			}
		    }
		}
	    }
	}
    }
}
*/


/*
void sqr_sgemm(float *Y, float *A, float *B, int n)
{
  float Y_0, Y_1, Y_2, Y_3;
  for(int ii = 0; ii < n; ii+=4)
    {
      for(int jj = 0; jj < n; jj+=4) 
	{
	  for(int i=ii; i < (ii+4); i++)
	    {
	      Y_0 = Y[i*n+jj+0]; Y_1 = Y[i*n+jj+1];
	      Y_2 = Y[i*n+jj+2]; Y_3 = Y[i*n+jj+3];
	      for(int k = 0; k < n; k++)
		{
		  const float A_i_k = A[i*n+k];
		  Y_0 += A_i_k*B[k*n+jj+0];
		  Y_1 += A_i_k*B[k*n+jj+1];
		  Y_2 += A_i_k*B[k*n+jj+2];
		  Y_3 += A_i_k*B[k*n+jj+3];
		}
	      Y[i*n+jj+0] = Y_0; Y[i*n+jj+1] = Y_1;
	      Y[i*n+jj+2] = Y_2; Y[i*n+jj+3] = Y_3;
	    }
	}
    }
}
*/
