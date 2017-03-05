#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include "clhelp.h"

int int_compare(const void *x, const void *y) {
    int i_x = *((int*)x);
    int i_y = *((int*)y);
    return i_x > i_y;
}

void rsort_scan(cl_command_queue &q, cl_context &ctxt, cl_kernel &sk, cl_kernel &uk, cl_mem &in, cl_mem &out, int v, int k, int n);
void rsort_reassemble(cl_command_queue &q, cl_context &ctxt, cl_kernel &rk, cl_mem &in, cl_mem &ones, cl_mem &zeros, cl_mem &out, int k, int n);

void cpu_rscan(int *in, int *out, int v, int k, int n) {
    int t = (in[0] >> k) & 0x1;
    out[0] = (t==v);
    for(int i = 1; i < n; i++) {
        t = (in[i] >> k) & 0x1;
        out[i] = out[i-1]+(t==v);
    }
}

int main(int argc, char *argv[]) {
    // Provide names of the OpenCL kernels and cl file that they're kept in
    std::string kernel_source_str;
    std::string arraycompact_kernel_file = std::string("radixsort.cl");
    std::list<std::string> kernel_names;
    std::string scan_name_str = std::string("scan");
    std::string update_name_str = std::string("update");
    std::string reassemble_name_str = std::string("reassemble");

    kernel_names.push_back(scan_name_str);
    kernel_names.push_back(update_name_str);
    kernel_names.push_back(reassemble_name_str);

    cl_vars_t cv;                       // Declare our variable list
    std::map<std::string, cl_kernel>    // Declare our kernels
    kernel_map;

    int c;                              // how long do you want your arrays?
    int n = (1<<20);                    // Size of data
    int *in, *out;                      // Arrays on the host (CPU)
    int *c_scan;
    int n_out = -1;
    bool silent = false;

    while((c = getopt(argc, argv, "n:s:"))!=-1) {
        switch(c) {
            case 'n':
                n = 1 << atoi(optarg);
                break;
            case 's':
                silent = atoi(optarg) == 1;
                break;
        }
    }

    in = new int[n];                    // Initialize input in
    out = new int[n];                   // Initialize output out
    c_scan = new int[n];                // Initialize c_scan
    bzero(out, sizeof(int)*n);          // zero out output
    bzero(c_scan, sizeof(int)*n);       // zerou out c_scan
    srand(5);
    for(int i = 0; i < n; i++) {        // Fill input test array with random values
        in[i] = rand();
    }

    readFile(arraycompact_kernel_file, kernel_source_str);  // Read OpenCL file into STL string
    initialize_ocl(cv);                                     // Initialize the OpenCL runtime source in clhelp.cpp
    compile_ocl_program(kernel_map, cv, kernel_source_str.c_str(), kernel_names); // Compile all OpenCl kernels

    cl_mem g_in, g_zeros, g_ones, g_out;                    // Arrays on the device (GPU)
    cl_mem g_temp;

    cl_int err = CL_SUCCESS;                                // Allocate memory for arrays on GPU
    g_in = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &err);
    CHK_ERR(err);  
    g_ones = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &err);
    CHK_ERR(err);
    g_zeros = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &err);
    CHK_ERR(err);
    g_out = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &err);
    CHK_ERR(err);

    // Write commands to a buffer object from host memory; copy data from host CPU to GPU
    err = clEnqueueWriteBuffer(cv.commands, g_in, true, 0, sizeof(int)*n, in, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, g_out, true, 0, sizeof(int)*n, c_scan, 0, NULL, NULL);
    CHK_ERR(err);

    // Declare global and local work sizes
    size_t global_work_size[1] = {n};
    size_t local_work_size[1] = {128};
    adjustWorkSize(global_work_size[0], local_work_size[0]);
    global_work_size[0] = std::max(local_work_size[0], global_work_size[0]);
    int left_over = 0;

    double t0 = timestamp();
    /*  GPU Radix sort implementation.
        Radix sort operates by:
            for currDigit in range(least significant, most significant digit):
                1. Place array items into "bins" according to their current digit
                2. Take items out again, but in order of the bins
            By the time you get to the most significant digit, your values will be in order!
        Notes about GPU Radix Sort:
            1. We are processing the numbers in binary so we only have to deal with two bins, 0 and 1.
            2. Thus we have to do 32 iterations, because in this case intsize = 32 bs
            3. As each item in the array is independent, we can perform these bin operations in parallel.
            4. In each iteration we map out to our bins, reassemble in an intermediate ordering, and update digit k.
    */
    for (int b=0; b<32; b++) { // 0 to 31 for each bit in an int
        // Separately fill zeros array for v=0; ones array for v=1 where v is the least significant bit
        rsort_scan(cv.commands, cv.context, kernel_map[scan_name_str], kernel_map[update_name_str], g_in, g_zeros, 0, b, n);
        rsort_scan(cv.commands, cv.context, kernel_map[scan_name_str], kernel_map[update_name_str], g_in, g_ones, 1, b, n);

        // Set kernel arguments (input, output, ones array, zeros array, local buffer, b index, size)
        err = clSetKernelArg(kernel_map[reassemble_name_str], 0, sizeof(cl_mem), &g_in);
        CHK_ERR(err);
        err = clSetKernelArg(kernel_map[reassemble_name_str], 1, sizeof(cl_mem), &g_out);
        CHK_ERR(err);
        err = clSetKernelArg(kernel_map[reassemble_name_str], 2, sizeof(cl_mem), &g_ones);
        CHK_ERR(err);
        err = clSetKernelArg(kernel_map[reassemble_name_str], 3, sizeof(cl_mem), &g_zeros);
        CHK_ERR(err);
        err = clSetKernelArg(kernel_map[reassemble_name_str], 4, 2*local_work_size[0]*sizeof(cl_int), NULL);
        CHK_ERR(err);
        err = clSetKernelArg(kernel_map[reassemble_name_str], 5, sizeof(int), &b);
        CHK_ERR(err);
        err = clSetKernelArg(kernel_map[reassemble_name_str], 6, sizeof(int), &n);
        CHK_ERR(err);

        // Put kernel reassemble on the queue to execute on GPU, where we map 0 and 1 to the output array locations
        err = clEnqueueNDRangeKernel(cv.commands,
                                    kernel_map[reassemble_name_str],    // our kernel, reassemble (what we want to do)
                                    1,                                  // work_dim, #dims work group uses to execute kernel
                                    NULL,                               // global_work_offset
                                    global_work_size,                   // global_work_size, matrix size work_dim holding dims of kernel
                                    local_work_size,                    // local_work_size, matrix size work_dim holding dims of work group
                                    0,                                  // num_events_in_wait_list
                                    NULL,                               // event_wait_list
                                    NULL);
        CHK_ERR(err);

        // Output becomes next input; continue iteration on next bit up
        cl_mem tmp = g_in;
        g_in = g_out;
        g_out = tmp;
    }
    t0 = timestamp() - t0; 

    // Read result of GPU on host CPU
    err = clEnqueueReadBuffer(cv.commands, g_in, true, 0, sizeof(int)*n, out, 0, NULL, NULL);
    CHK_ERR(err);

    // Sort array on CPU
    double t1 = timestamp();
    qsort(in, n, sizeof(int), int_compare);
    t1 = timestamp() - t1;    

    // Check correctness of GPU result against CPU result
    for(int i = 0; i < n; i++) {
        if(in[i] != out[i]) {
            if(!silent) {
                printf("not sorted @ %d: %d vs %d!\n", i, in[i], out[i]);
            }
            goto done;
        }
    }
    if(!silent) {
        printf("array sorted\n");
    }

    // Timing comparisons
    if(silent) {
        printf("%d, %g, %g\n", n, t1, t0);
    } else {
        printf("GPU: array of length %d sorted in %g seconds\n", n, t0);
        printf("CPU: array of length %d sorted in %g seconds\n", n, t1);
    }
    done:

    clReleaseMemObject(g_in); 
    clReleaseMemObject(g_out);
    clReleaseMemObject(g_ones);
    clReleaseMemObject(g_zeros);
    uninitialize_ocl(cv);
    delete [] in;
    delete [] out;
    delete [] c_scan;
    return 0;
}

void rsort_scan(cl_command_queue &q, cl_context &ctxt, cl_kernel &sk, cl_kernel &uk, cl_mem &in, cl_mem &out, int v, int k, int len) {
    size_t global_work_size[1] = {len};
    size_t local_work_size[1] = {128};
    int left_over = 0;
    cl_int err;

    adjustWorkSize(global_work_size[0], local_work_size[0]);
    global_work_size[0] = std::max(local_work_size[0], global_work_size[0]);
    left_over = global_work_size[0] / local_work_size[0];

    cl_mem g_bscan = clCreateBuffer(ctxt, CL_MEM_READ_WRITE, sizeof(int)*left_over, NULL, &err);
    CHK_ERR(err);
    err = clSetKernelArg(sk, 0, sizeof(cl_mem), &in);
    CHK_ERR(err);
    err = clSetKernelArg(sk, 1, sizeof(cl_mem), &out);
    CHK_ERR(err);
    err = clSetKernelArg(sk, 2, sizeof(cl_mem), &g_bscan);                      // Per work-group partial scan output
    CHK_ERR(err);
    err = clSetKernelArg(sk, 3, 2*local_work_size[0]*sizeof(cl_int), NULL);     // #bytes for dynamically sized local (private memory) "buf"
    CHK_ERR(err);
    err = clSetKernelArg(sk, 4, sizeof(int), &v);                               // v = 0 or 1, to perform scan of bs set (or unset)
    CHK_ERR(err);
    err = clSetKernelArg(sk, 5, sizeof(int), &k);                               // The current b position (0 to 31) that we want to operate on
    CHK_ERR(err);
    err = clSetKernelArg(sk, 6, sizeof(int), &len);
    CHK_ERR(err);

    // Divvy up our scan job into local workspaces; put kernel scan on the queue to execute on GPU
    err = clEnqueueNDRangeKernel(q,
                                sk,                 // our kernel, scan (what we want to do)
                                1,                  // work_dim, #dims work group uses to execute kernel
                                NULL,               // global_work_offset
                                global_work_size,   // global_work_size, matrix size work_dim holding dims of kernel
                                local_work_size,    // local_work_size, matrix size work_dim holding dims of work group
                                0,                  // num_events_in_wait_list
                                NULL,               // event_wait_list
                                NULL);
    CHK_ERR(err);

    if(left_over > 1) { // Checks if there are leftover chunks that still need scanning. The recursive part kicks in here.
        cl_mem g_bbscan = clCreateBuffer(ctxt, CL_MEM_READ_WRITE, sizeof(int)*left_over, NULL, &err);        
        rsort_scan(q, ctxt, sk, uk, g_bscan, g_bbscan, -1, k, left_over);

        err = clSetKernelArg(uk, 0, sizeof(cl_mem), &out);
        CHK_ERR(err);      
        err = clSetKernelArg(uk, 1, sizeof(cl_mem), &g_bbscan);
        CHK_ERR(err);
        err = clSetKernelArg(uk, 2, sizeof(int), &len);
        CHK_ERR(err);

        // Divvy up our scan consolidate job into local workspaces; put kernel update on the queue to execute on GPU
        err = clEnqueueNDRangeKernel(q,                 // update partial scans on queue
                                    uk,                 // our kernel, update (consolidate scans)
                                    1,                  // work_dim, #dims work group uses to execute kernel
                                    NULL,               // global_work_offset
                                    global_work_size,   // global_work_size, matrix size work_dim holding dims of kernel
                                    local_work_size,    // local_work_size, matrix size work_dim holding dims of work group
                                    0,                  // num_events_in_wait_list
                                    NULL,               // event_wait_list
                                    NULL);
        CHK_ERR(err);
        clReleaseMemObject(g_bbscan);                   // clear memory
    }
    clReleaseMemObject(g_bscan);                        // clear memory
}