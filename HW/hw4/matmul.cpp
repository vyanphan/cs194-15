#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>

#include "clhelp.h"

void sqr_sgemm(float *Y, float *A, float *B, int n);

int main(int argc, char *argv[]) {
    std::string matmul_kernel_str;
    std::string matmul_name_str = std::string("matmul");
    std::string matmul_kernel_file = std::string("matmul.cl");

    cl_vars_t cv; 
    cl_kernel matmul;
  
    readFile(matmul_kernel_file, matmul_kernel_str);
    initialize_ocl(cv);
    compile_ocl_program(matmul, cv, matmul_kernel_str.c_str(), matmul_name_str.c_str());
  
    float *h_A, *h_B, *h_Y, *h_YY;
    cl_mem g_A, g_B, g_Y;
    int n = (1<<10);
    h_A = new float[n*n];
    assert(h_A);
    h_B = new float[n*n];
    assert(h_B);
    h_Y = new float[n*n];
    assert(h_Y);
    h_YY = new float[n*n];
    assert(h_YY);
    bzero(h_Y, sizeof(float)*n*n);
    bzero(h_YY, sizeof(float)*n*n);
  
    for(int i = 0; i < (n*n); i++) {
        h_A[i] = (float)drand48();
        h_B[i] = (float)drand48();
    }


    cl_int err = CL_SUCCESS;
    // CS194: Allocate Buffers on the GPU. (We're already allocating the Y buffer on the GPU for you)
    g_Y = clCreateBuffer(cv.context,CL_MEM_READ_WRITE, sizeof(float)*n*n, NULL, &err);
    CHK_ERR(err);
    g_A = clCreateBuffer(cv.context,CL_MEM_READ_WRITE, sizeof(float)*n*n, NULL, &err);
    CHK_ERR(err);
    g_B = clCreateBuffer(cv.context,CL_MEM_READ_WRITE, sizeof(float)*n*n, NULL, &err);
    CHK_ERR(err);
  
    /* CS194: Copy data from host CPU to GPU */
    err = clEnqueueWriteBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n*n, h_Y, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, g_A, true, 0, sizeof(float)*n*n, h_A, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, g_B, true, 0, sizeof(float)*n*n, h_B, 0, NULL, NULL);
    CHK_ERR(err);

    /* CS194: Create appropriately sized workgroups */
    size_t global_work_size[2] = {n, n};
    size_t local_work_size[2] = {16, 16};
  
    /* CS194: Set kernel arguments */
    err = clSetKernelArg(matmul, 0, sizeof(cl_mem), &g_Y);
    CHK_ERR(err);
    err = clSetKernelArg(matmul, 1, sizeof(cl_mem), &g_A);
    CHK_ERR(err);
    err = clSetKernelArg(matmul, 2, sizeof(cl_mem), &g_B);
    CHK_ERR(err);
    err = clSetKernelArg(matmul, 3, sizeof(int), &n);
    CHK_ERR(err);   

    double t0 = timestamp();
    err = clEnqueueNDRangeKernel(cv.commands,
                   matmul,
                   2,                  //work_dim, the number of dimensions used in the work group to execute the kernel
                   NULL,               //global_work_offset
                   global_work_size,   //global_work_size, an array size work_dim describing the dimensions to execute the kernel
                   local_work_size,    //local_work_size, an array size work_dim holding number of work items making up a work group
                   0,                  //num_events_in_wait_list
                   NULL,               //event_wait_list
                   NULL);
    CHK_ERR(err);
    err = clFinish(cv.commands);
    CHK_ERR(err);
    t0 = timestamp()-t0;

    /* Read result of GPU on host CPU */
    err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n*n, h_Y, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueReadBuffer(cv.commands, g_A, true, 0, sizeof(float)*n*n, h_A, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueReadBuffer(cv.commands, g_B, true, 0, sizeof(float)*n*n, h_B, 0, NULL, NULL);
    CHK_ERR(err);
    err = clFinish(cv.commands);
    CHK_ERR(err);

    double t1 = timestamp();
    sqr_sgemm(h_YY, h_A, h_B, n);
    t1 = timestamp()-t1;

    for(int i = 0; i < (n*n); i++) {
        double d = h_YY[i] - h_Y[i];
        d *= d;
        if(d > 0.0001) {
            printf("Output: %f  Expected: %f\n", h_Y[i], h_YY[i]);
            break;
        }
    }
  
    uninitialize_ocl(cv);
  
    delete [] h_A; 
    delete [] h_B; 
    delete [] h_Y;
    delete [] h_YY;

    clReleaseMemObject(g_A); 
    clReleaseMemObject(g_B); 
    clReleaseMemObject(g_Y);
  
    double gpu_flops_s = (2.0 * pow((double)n, 3.0)) / t0;
    printf("GPU: %g gflops/sec\n", gpu_flops_s / (1e9));

    double cpu_flops_s = (2.0 * pow((double)n, 3.0)) / t1;
    printf("CPU: %g gflops/sec\n", cpu_flops_s / (1e9));
    return 0;
}
