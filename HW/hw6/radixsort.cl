// The update kernel; consolidates the scan
__kernel void update(__global int *in, __global int *block, int n) {
    size_t glb_id = get_global_id(0);
    size_t loc_id = get_local_id(0);
    size_t loc_sz = get_local_size(0);
    size_t grp_id = get_group_id(0);
    if(glb_id < n && grp_id > 0) {
        in[glb_id] = in[glb_id] + block[grp_id-1];
    }
}

// The reassemble kernel; regroups the results of our separate radix sorts into our final array
__kernel void reassemble(__global int *in, __global int *out, __global int *ones, __global int *zeros, __local int *buf, int k, int n) {
    size_t glb_id = get_global_id(0);
    size_t loc_id = get_local_id(0);

    if(glb_id<n) {
        int temp = in[glb_id];
        buf[loc_id] = ((temp >> k) & 0x1);
    } else {
        buf[loc_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int index;
    if (glb_id<n) {
        if (buf[loc_id] == 1) {
            index = zeros[n - 1] + ones[glb_id] - 1;
        } else {
            index = zeros[glb_id] - 1;
    }
    if(index < n) 
        out[index] = in[glb_id];
    }
}

// The scan kernel; we perform the scanning operations here
__kernel void scan(__global int *in, __global int *out, __global int *bout, __local int *buf, int v, int k, int n) {
    size_t glb_id = get_global_id(0);
    size_t loc_id = get_local_id(0);
    size_t loc_sz = get_local_size(0);
    size_t grp_id = get_group_id(0);

    if(glb_id<n) {                                          // Copy data from input array into local buffer, according to bin 0 or 1
        int t = in[glb_id];
        t = (v==-1) ? t : (v==((t>>k) & 1));
        buf[loc_id] = t;
    } else {
        buf[loc_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);                           // Barrier to keep it in local memory

    // My scan code, optimized for the logarithmic method
    __local int buf2[128];                                  // 128 was our local work size as defined in the .cpp file
    for (int i = 1; i < loc_sz; i *= 2) {
        if (loc_id >= i) {
            buf2[loc_id - i] = buf[loc_id - i];             // Recopy the values over to buf2
        }
        barrier(CLK_LOCAL_MEM_FENCE);                       // Barrier to keep it in local memory
        if (loc_id >= i) {
            buf[loc_id] = buf2[loc_id - i] + buf[loc_id];   // Add every other value (according to step size) in parallel
        }
        barrier(CLK_LOCAL_MEM_FENCE);                       // Barrier to keep it in local memory
    }
    if (glb_id < n) {
        out[glb_id] = buf[loc_id];                          // Write the scanned data to output
    }
    if (loc_id == loc_sz - 1) {
        bout[grp_id] = out[glb_id];                         // Write the biggest data to bout; this will be used in the update
    }
}