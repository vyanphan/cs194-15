
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>
#include <x86intrin.h>
using namespace std;

double timestamp() {
    struct timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

// Simple Blur
void simple_blur(float* out, int n, float* frame, int* radii){
    for(int r=0; r<n; r++) {
        for(int c=0; c<n; c++) {
            int rd = radii[r*n+c];
            int num = 0;
            float avg = 0;
            for(int r2=max(0,r-rd); r2<=min(n-1, r+rd); r2++) {
                for(int c2=max(0, c-rd); c2<=min(n-1, c+rd); c2++){
                    avg += frame[r2*n+c2];
                    num++;
                }
            }
            out[r*n+c] = avg/num;
        }
    }
}

// My Blur
void my_blur(float* out, int n, float* frame, int* radii, int nthr) {    
    omp_set_num_threads(nthr);

    #pragma omp parallel for schedule(dynamic, 8)
    for(int r=0; r<n; r++) {
        for(int c=0; c<n; c++) {
            int rd = radii[r*n+c];
            int rstart = max(0, r-rd);
            int cstart = max(0, c-rd);
            int rend = min(n-1, r+rd);
            int cend = min(n-1, c+rd);
            int cells = (cend - cstart + 1) * (rend - rstart + 1);

            __m128 vec = _mm_setzero_ps();
            for(; cend-cstart>=4; cstart+=4) {
                for(int i = rstart; i <= rend; i++) {
                    __m128 temp = _mm_loadu_ps(&frame[i*n + cstart]);
                    vec = _mm_add_ps(vec, temp);
                }
            }

            float sum = 0;
            for(int j=cstart; j<=cend; j++) {
                for(int i=rstart; i<=rend; i++) {
                    sum += frame[i*n + j];
                }
            }
            for(int i=0; i<4; i++) {
                sum += vec[i];
            }          

            out[r*n+c] = sum/cells;
        }        
    }            
}

int main(int argc, char *argv[])
{
    //Generate random radii
    srand(0);
    int n = 3000;
    int* radii = new int[n*n];
    for(int i=0; i<n*n; i++)
        radii[i] = 6*i/(n*n) + rand()%6;

    //Generate random frame
    float* frame = new float[n*n];
    for(int i=0; i<n*n; i++)
        frame[i] = rand()%256;

    //Blur using simple blur
    float* out = new float[n*n];
    double time = timestamp();
    simple_blur(out, n, frame, radii);
    time = timestamp() - time;
    printf("Time needed for naive blur = %.3f seconds.\n", time);


    for(int nthr=1; nthr<=16; nthr++) {
        //Blur using your blur
        float* out2 = new float[n*n];
        double time2 = timestamp();
        my_blur(out2, n, frame, radii, nthr);
        time2 = timestamp() - time2;

        //Check result
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++){
                float dif = out[i*n+j] - out2[i*n+j];
                if(dif*dif>1.0f){
                    printf("Your blur does not give the right result!\n");
                    printf("For element (row, column) = (%d, %d):\n", i, j);
                    printf("  Simple blur gives %.2f\n", out[i*n+j]);
                    printf("  Your blur gives %.2f\n", out2[i*n+j]);
                    exit(-1);
                }
            }
        }
        printf("%.3f\n", time2);
        delete[] out2;
    }
  

//Delete
delete[] radii;
delete[] frame;
delete[] out;
}
