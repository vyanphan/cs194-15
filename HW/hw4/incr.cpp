#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "clhelp.h"

int main(int argc, char *argv[]) {
    std::string incr_kernel_str;

    // Provide names of the OpenCL kernels and cl file that they're kept in 
    std::string incr_name_str = std::string("incr");
    std::string incr_kernel_file = std::string("incr.cl");

    cl_vars_t cv;                                   // define a cl_vars_t cv
    cl_kernel incr;                                 // define a cl_kernel incr (our instruction is the incr function)

    // Starting all of our OpenCl stuff
    readFile(incr_kernel_file, incr_kernel_str);    // Read OpenCL file into STL string
    initialize_ocl(cv);                             // Initialize the OpenCL runtime. Source in clhelp.cpp
    compile_ocl_program(incr, cv, incr_kernel_str.c_str(), incr_name_str.c_str());  // Compile all OpenCL kernels
    float *h_Y, *h_YY;                              // Arrays on the host (CPU)
    cl_mem g_Y;                                     // Arrays on the device (GPU)


    int n = (1<<20);                                // this will be our array size (we'll want to size the global workspace to this later)
    h_Y = new float[n];                             // initialize random identical float arrays h_Y and h_YY of size n
    h_YY = new float[n];                            // h_YY is our reference array; h_Y is the one we'll increment 
    for(int i = 0; i < n; i++) {
        h_YY[i] = h_Y[i] = (float)drand48();
    }

    cl_int err = CL_SUCCESS;
    // allocate memory for arrays on the GPU (with error checking)
    g_Y = clCreateBuffer(cv.context,CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &err);
    CHK_ERR(err);

    // write commands to a buffer object from host memory (with error checking)
    err = clEnqueueWriteBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n, h_Y, 0, NULL, NULL);
    CHK_ERR(err);

    // our global workspace should be the total size of our array; we picked 128 for the size of our local chunks
    size_t global_work_size[1] = {n};
    size_t local_work_size[1] = {128};
    
    // we're divvying up our incr job into our local workspaces in the next two chunks
    err = clSetKernelArg(incr, 0, sizeof(cl_mem), &g_Y);    // set argument 0 of kernel incr to be the commands
    CHK_ERR(err);
    err = clSetKernelArg(incr, 1, sizeof(int), &n);         // and argument 1 to be our global vector size
    CHK_ERR(err);
   
    // put kernel incr on the queue to execute, with error checking (i.e. preparing the incr function for execution)
    err = clEnqueueNDRangeKernel(cv.commands,
  			       incr,
  			       1,                  //work_dim, the number of dimensions used in the work group to execute the kernel
  			       NULL,               //global_work_offset
  			       global_work_size,   //global_work_size, an array size work_dim describing the dimensions to execute the kernel
  			       local_work_size,    //local_work_size, an array size work_dim holding the number of work items that make up a work group
  			       0,                  //num_events_in_wait_list
  			       NULL,               //event_wait_list
  			       NULL                //
  			       );
    CHK_ERR(err);

    // Read result of GPU on host CPU (this is where we execute all the actual incr stuff)
    err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n, h_Y, 0, NULL, NULL);
    CHK_ERR(err);

    // checking our OpenCL-incremented array (h_Y) against our manually incremented array (h_YY)
    bool er = false;
    for(int i = 0; i < n; i++) {
        float d = (h_YY[i] + 1.0f);
        if(h_Y[i] != d) {
  	        printf("error at %d\n", i);
  	        er = true;
            break;
        }
    }
    if(!er) {
        printf("CPU and GPU results match\n");
    }

    //clear memory and free arrays
    uninitialize_ocl(cv);    
    delete [] h_Y;
    delete [] h_YY;
    clReleaseMemObject(g_Y);
    
    return 0;
}
