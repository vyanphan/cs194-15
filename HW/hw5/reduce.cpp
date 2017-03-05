#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <unistd.h>
#include "clhelp.h"

int main(int argc, char *argv[]) {
    // Provide names of the OpenCL kernels and cl file that they're kept in
    std::string reduce_kernel_str;
    std::string reduce_name_str = std::string("reduce");
    std::string reduce_kernel_file = std::string("reduce.cl");

    cl_vars_t cv;                                           // Declare our variable list
    cl_kernel reduce;                                       // Declare our kernel (reduce)
    readFile(reduce_kernel_file, reduce_kernel_str);        // Read OpenCL file into STL string
    initialize_ocl(cv);                                     // Initialize the OpenCL runtime source in clhelp.cpp
    compile_ocl_program(reduce, cv, reduce_kernel_str.c_str(), reduce_name_str.c_str()); // Compile all OpenCl kernels

    int *h_A, *h_Y;                                         // Arrays on the host (CPU) 
    cl_mem g_Out, g_In;                                     // Arrays on the device (GPU)
    int n = (1<<24);                                        // Size of data

    int c;                                                  // how long do you want your arrays?
    while((c = getopt(argc, argv, "n:")) != -1) {
        switch(c) {
            case 'n':
            n = atoi(optarg);
            break;
        }
    }
    if(n==0) {
        return 0;
    }

    h_A = new int[n];                                       // Initialize input A
    h_Y = new int[n];                                       // Initialize destination Y
    for(int i = 0; i < n; i++) {                            // Fill with values
        h_A[i] = 1;
        h_Y[i] = 0;
    }

    cl_int err = CL_SUCCESS;                                // Allocate memory for arrays on GPU
    g_Out = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &err);
    CHK_ERR(err);  
    g_In  = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &err);
    CHK_ERR(err);

    // Write commands to a buffer object from host memory; copy data from host CPU to GPU
    err = clEnqueueWriteBuffer(cv.commands, g_Out, true, 0, sizeof(int)*n, h_Y, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, g_In,  true, 0, sizeof(int)*n, h_A, 0, NULL, NULL);
    CHK_ERR(err);

    // REDUCTION STAGE!!!!
    size_t global_work_size[1] = {n};
    size_t local_work_size[1] = {512};

    double t0 = timestamp();

    // LEVEL ONE!!!!!! ENTIRE WORK SPACE. Set kernel arguments (input, output, local buffer, size n)
    err = clSetKernelArg(reduce, 0, sizeof(cl_mem), &g_In);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 1, sizeof(cl_mem), &g_Out);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 2, sizeof(cl_int)* local_work_size[0], NULL);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 3, sizeof(int), &n);
    CHK_ERR(err);
  
    // Divvy up our reduce job into local workspaces; put kernel reduce on the queue to execute on GPU
    err = clEnqueueNDRangeKernel(cv.commands,
                                reduce,                     // our kernel, reduce (what we want to do)
                                1,                          // work_dim, #dims work group uses to execute kernel
                                NULL,                       // global_work_offset
                                global_work_size,           // global_work_size, matrix size work_dim holding dims of kernel
                                local_work_size,            // local_work_size, matrix size work_dim holding dims of work group
                                0,                          // num_events_in_wait_list
                                NULL,                       // event_wait_list
                                NULL);
    CHK_ERR(err);

    // LEVEL TWO!!!!!! BLOCKS SIZE 512.
    int m = n/512; 
    global_work_size[1] = {m};
   
    // Set kernel arguments (input, output, local buffer, size num blocks)
    err = clSetKernelArg(reduce, 0, sizeof(cl_mem), &g_Out);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 1, sizeof(cl_mem), &g_Out);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 2, sizeof(cl_int)* local_work_size[0], NULL);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 3, sizeof(int), &m);
    CHK_ERR(err);
  
    // Divvy up our reduce job into local workspaces; put kernel reduce on the queue to execute on GPU
    err = clEnqueueNDRangeKernel(cv.commands,               
                                reduce,                     // our kernel, reduce (what we want to do)
                                1,                          // work_dim, #dims work group uses to execute kernel
                                NULL,                       // global_work_offset
                                global_work_size,           // global_work_size, matrix size work_dim holding dims of kernel
                                local_work_size,            // local_work_size, matrix size work_dim holding dims of work group
                                0,                          // num_events_in_wait_list
                                NULL,                       // event_wait_list
                                NULL);
    CHK_ERR(err);

    // LEVEL THREE!!!!!! DIVIDE CHUNK 2 BLOCKS INTO MORE BLOCKS.
    int k = m/512; 
    global_work_size[1] = {k};    

    // Set kernel arguments (input, output, local buffer, size num smaller blocks)
    err = clSetKernelArg(reduce, 0, sizeof(cl_mem), &g_Out);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 1, sizeof(cl_mem), &g_Out);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 2, sizeof(cl_int)* local_work_size[0], NULL);
    CHK_ERR(err);
    err = clSetKernelArg(reduce, 3, sizeof(int), &m);
    CHK_ERR(err);
    
    // Divvy up our reduce job into local workspaces; put kernel reduce on the queue to execute on GPU
    err = clEnqueueNDRangeKernel(cv.commands,
                                reduce,                     // our kernel, reduce (what we want to do)
                                1,                          // work_dim, #dims work group uses to execute kernel
                                NULL,                       // global_work_offset
                                global_work_size,           // global_work_size, matrix size work_dim holding dims of kernel
                                local_work_size,            // local_work_size, matrix size work_dim holding dims of work group
                                0,                          // num_events_in_wait_list
                                NULL,                       // event_wait_list
                                NULL);
    CHK_ERR(err);
  
    t0 = timestamp() - t0;
  
    // Read result of GPU on host CPU
    err = clEnqueueReadBuffer(cv.commands, g_Out, true, 0, sizeof(int)*n, h_Y, 0, NULL, NULL);
    CHK_ERR(err);
  
    double t1 = timestamp();
    int sum=0.0f;
    for(int i = 0; i < n; i++) {
        sum += h_A[i];
    }
    t1 = timestamp() - t1;

    // Checking our OpenCL-reduced array against manual reduction; final timing
    if(sum!=h_Y[0]) {
        printf("WRONG: CPU sum = %d, GPU sum = %d\n", sum, h_Y[0]);
        printf("WRONG: difference = %d\n", sum-h_Y[0]);
    } else {
        printf("CORRECT: %d\n", h_Y[0]);
        printf("CPU: %g   GPU: %g\n", t1, t0);
        printf("Speedup: %g\n", t1/t0);
    }
 
    uninitialize_ocl(cv);   
    delete [] h_A; 
    delete [] h_Y;  
    clReleaseMemObject(g_Out); 
    clReleaseMemObject(g_In);   
    return 0;
}