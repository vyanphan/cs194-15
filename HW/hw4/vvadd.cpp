#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "clhelp.h"

int main(int argc, char *argv[]) {
    std::string vvadd_kernel_str;

    // Provide names of the OpenCL kernels and cl file that they're kept in
    std::string vvadd_name_str = std::string("vvadd");
    std::string vvadd_kernel_file = std::string("vvadd.cl");

    cl_vars_t cv; 
    cl_kernel vvadd;

    readFile(vvadd_kernel_file, vvadd_kernel_str);      // Read OpenCL file into STL string
    initialize_ocl(cv);                                 // Initialize the OpenCL runtime source in clhelp.cpp
    compile_ocl_program(vvadd, cv, vvadd_kernel_str.c_str(), vvadd_name_str.c_str()); // Compile all OpenCL kernels
    float *h_A, *h_B, *h_Y;                             // Arrays on the host (CPU) 
    cl_mem g_A, g_B, g_Y;                               // Arrays on the device (GPU)

    int n = (1<<20);                                    // Allocate arrays on the host and fill with random data */
    h_A = new float[n];
    h_B = new float[n];
    h_Y = new float[n];
    bzero(h_Y, sizeof(float)*n);
  
    for(int i = 0; i < n; i++) {
        h_A[i] = (float)drand48();
        h_B[i] = (float)drand48();
    }

    cl_int err = CL_SUCCESS;                            // CS194: Allocate memory for arrays on the GPU
    // CS194: Here's something to get you started 
    g_Y = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &err);
    CHK_ERR(err);
    g_A = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &err);
    CHK_ERR(err);
    g_B = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &err);
    CHK_ERR(err);
  
    /* CS194: Copy data from host CPU to GPU */
    err = clEnqueueWriteBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n, h_Y, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, g_A, true, 0, sizeof(float)*n, h_A, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, g_B, true, 0, sizeof(float)*n, h_B, 0, NULL, NULL);
    CHK_ERR(err);
     
    /* CS194: Define the global and local workgroup sizes */
    size_t global_work_size[1] = {n};
    size_t local_work_size[1] = {1};
      
    /* CS194: Set Kernel Arguments */
    err = clSetKernelArg(vvadd, 0, sizeof(cl_mem), &g_Y);
    CHK_ERR(err);
    err = clSetKernelArg(vvadd, 1, sizeof(cl_mem), &g_A);
    CHK_ERR(err);
    err = clSetKernelArg(vvadd, 2, sizeof(cl_mem), &g_B);
    CHK_ERR(err);
    err = clSetKernelArg(vvadd, 3, sizeof(int), &n);
    CHK_ERR(err);   

    /* CS194: Call kernel on the GPU */
    err = clEnqueueNDRangeKernel(cv.commands,
                   vvadd,
                   1,                  //work_dim, the number of dimensions used in the work group to execute the kernel
                   NULL,               //global_work_offset
                   global_work_size,   //global_work_size, an array size work_dim describing the dimensions to execute the kernel
                   local_work_size,    //local_work_size, an array size work_dim holding number of work items making up a work group
                   0,                  //num_events_in_wait_list
                   NULL,               //event_wait_list
                   NULL);
    CHK_ERR(err);


    // Read result of GPU on host CPU
    err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n, h_Y, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueReadBuffer(cv.commands, g_A, true, 0, sizeof(float)*n, h_A, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueReadBuffer(cv.commands, g_B, true, 0, sizeof(float)*n, h_B, 0, NULL, NULL);
    CHK_ERR(err);

    for(int i = 0; i < n; i++) {                        // Check answer
        float d = h_A[i] + h_B[i];
        if(h_Y[i] != d) {
            printf("Error at %d\n. Output: %f  Expected: %f\n", i, h_Y[i], d);
            break;
    	}
    }

    uninitialize_ocl(cv);                               // Shut down the OpenCL runtime
    delete [] h_A; 
    delete [] h_B; 
    delete [] h_Y;
    clReleaseMemObject(g_A); 
    clReleaseMemObject(g_B); 
    clReleaseMemObject(g_Y);
    return 0;
}
