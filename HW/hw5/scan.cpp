#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include "clhelp.h"

void cpu_scan(int *in, int *out, int n) {
    out[0] = in[0];
    for(int i = 1; i < n; i++) {
        out[i] = out[i-1] + in[i];
    }
}

// The recursive scan takes in chunks of the array that need to be scanned and keeps breaking it down until it gets to a 
// chunk that's small enough to fit in the local workspace (we are using size 128 here); that is, when there are no 
// leftovers remaining. When it gets to that point, it does a regular scan when. After all the scanning on the mini chunks
// are done, it calls the update kernel to consolidate all the little recursive scans.
void recursive_scan(cl_command_queue &queue, cl_context &context, cl_kernel &scan_kern, 
                    cl_kernel &update_kern, cl_mem &in, cl_mem &out, int len) {
    size_t global_work_size[1] = {len};
    size_t local_work_size[1] = {128};
    int left_over = 0;
    cl_int err;
  
    adjustWorkSize(global_work_size[0], local_work_size[0]);
    global_work_size[0] = std::max(local_work_size[0], global_work_size[0]);
    left_over = global_work_size[0] / local_work_size[0];
  
    cl_mem g_bscan = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*left_over, NULL, &err);
    CHK_ERR(err);

    // Set kernel arguments (input, output, output buffer, local buffer, array size)
    err = clSetKernelArg(scan_kern, 0, sizeof(cl_mem), &in);
    CHK_ERR(err);
    err = clSetKernelArg(scan_kern, 1, sizeof(cl_mem), &out);
    CHK_ERR(err);
    err = clSetKernelArg(scan_kern, 2, sizeof(cl_mem), &g_bscan);
    CHK_ERR(err);
    err = clSetKernelArg(scan_kern, 3, 2*local_work_size[0]*sizeof(cl_int), NULL);
    CHK_ERR(err);
    err = clSetKernelArg(scan_kern, 4, sizeof(int), &len);
    CHK_ERR(err);

    // Divvy up our scan job into local workspaces; put kernel scan on the queue to execute on GPU
    err = clEnqueueNDRangeKernel(queue,
                                scan_kern,                   // our kernel, reduce (what we want to do)
                                1,                           // work_dim, #dims work group uses to execute kernel
                                NULL,                        // global_work_offset
                                global_work_size,            // global_work_size, matrix size work_dim holding dims of kernel
                                local_work_size,             // local_work_size, matrix size work_dim holding dims of work group
                                0,                           // num_events_in_wait_list
                                NULL,                        // event_wait_list
                                NULL);
    CHK_ERR(err);

    // Checks if there are leftover chunks that still need scanning. The recursive part kicks in here.
    if(left_over > 1) {
        cl_mem g_bbscan = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*left_over, NULL, &err); 
        recursive_scan(queue, context, scan_kern, update_kern, g_bscan, g_bbscan, left_over);

        // Set kernel arguments (input, block, array size)
        err = clSetKernelArg(update_kern, 0, sizeof(cl_mem), &out);
        CHK_ERR(err);      
        err = clSetKernelArg(update_kern, 1, sizeof(cl_mem), &g_bbscan);
        CHK_ERR(err);
        err = clSetKernelArg(update_kern, 2, sizeof(int), &len);
        CHK_ERR(err);
      
        // Divvy up our scan consolidate job into local workspaces; put kernel for scan consolidating on the queue to execute on GPU
        err = clEnqueueNDRangeKernel(queue,
                                    update_kern,            // our kernel, reduce (what we want to do)
                                    1,                      // work_dim, #dims work group uses to execute kernel
                                    NULL,                   // global_work_offset
                                    global_work_size,       // global_work_size, matrix size work_dim holding dims of kernel
                                    local_work_size,        // local_work_size, matrix size work_dim holding dims of work group
                                    0,                      // num_events_in_wait_list
                                    NULL,                   // event_wait_list
                                    NULL);
        CHK_ERR(err);    
        clReleaseMemObject(g_bbscan);
    }
    clReleaseMemObject(g_bscan);
}


int main(int argc, char *argv[]) {
    // Provide names of the OpenCL kernels and cl file that they're kept in
    std::string kernel_source_str;  
    std::string arraycompact_kernel_file = std::string("scan.cl");
    std::list<std::string> kernel_names;
    std::string scan_name_str = std::string("scan");
    std::string update_name_str = std::string("update");
    kernel_names.push_back(scan_name_str);
    kernel_names.push_back(update_name_str);

    cl_vars_t cv;                                           // Declare our variable list
    std::map<std::string, cl_kernel> kernel_map;            // Declare our kernel (scan)

    int c;                                                  // How long do you want your arrays?
    int n = (1<<20);                                        // Array size/global workspace
    int *in, *out;
    int *cg_scan;
    while((c = getopt(argc, argv, "n:"))!=-1) {
        switch(c) {
            case 'n':
            n = atoi(optarg);
            break;
        }
    }

    in = new int[n];                                        // Initialize input array
    out = new int[n];                                       // Initialize output array
    cg_scan = new int[n];                                   // Initialize cgscan array
    bzero(cg_scan, sizeof(int)*n);                          // Zero out cgscan
    bzero(out, sizeof(int)*n);                              // Zero out our output
    srand(5);                                               // Generate random seed
    for(int i = 0; i < n; i++) {                            // Fill up our input array with random test data
        in[i] = rand() %2;
    }

    cpu_scan(in, out, n);                                   // CPU scan, to do correctness checking later

    readFile(arraycompact_kernel_file, kernel_source_str);  // Read OpenCL file into STL string   
    initialize_ocl(cv);                                     // Initialize the OpenCL runtime source in clhelp.cpp
    compile_ocl_program(kernel_map, cv, kernel_source_str.c_str(), kernel_names); // Compile all OpenCl kernels
  
    cl_mem g_in, g_scan;                                    // Arrays on the device (GPU)
    cl_int err = CL_SUCCESS;                                // Allocate memory for arrays on GPU
    g_in = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &err);
    CHK_ERR(err);  
    g_scan = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &err);
    CHK_ERR(err);
  
    // Write commands to a buffer object from host memory; copy data from host CPU to GPU
    err = clEnqueueWriteBuffer(cv.commands, g_in, true, 0, sizeof(int)*n, in, 0, NULL, NULL);
    CHK_ERR(err);

    // Set global and local work sizes  
    size_t global_work_size[1] = {n};
    size_t local_work_size[1] = {128};
    adjustWorkSize(global_work_size[0], local_work_size[0]);
    global_work_size[0] = std::max(local_work_size[0], global_work_size[0]);
 
    // Runs our version of a GPU scan to compare to the CPU scan
    recursive_scan(cv.commands, cv.context, kernel_map[scan_name_str], kernel_map[update_name_str], g_in, g_scan, n);
    
    // Issues all previously queued OpenCL commands in a command-queue to the device associated with the command-queue.
    err = clFlush(cv.commands);
    CHK_ERR(err);
  
    // Read result of GPU on host CPU
    err = clEnqueueReadBuffer(cv.commands, g_scan, true, 0, sizeof(int)*n, cg_scan, 0, NULL, NULL);
    CHK_ERR(err);
  
    // Check for correctness of our GPU scan against CPU scan and compares timing results
    for(int i =0; i < n; i++) {
        if(out[i] != cg_scan[i]) {
            printf("scan mismatch @ %d: cpu=%d, gpu=%d\n", i, out[i], cg_scan[i]);
            break;
        }
    }
    printf("CORRECT RESULT\n");

    // Free arrays and shut down kernels
    clReleaseMemObject(g_in); 
    clReleaseMemObject(g_scan);
    uninitialize_ocl(cv);
    delete [] in;
    delete [] out;
    delete [] cg_scan;
    return 0;
}