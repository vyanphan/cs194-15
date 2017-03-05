// The update kernel; consolidates the scan
__kernel void update(__global int *in, __global int *block, int n) {
    // Some variables we're going to use later
    size_t glb_id = get_global_id(0);
    size_t loc_id = get_local_id(0);
    size_t loc_sz = get_local_size(0);
    size_t grp_id = get_group_id(0);

    if(glb_id < n && grp_id > 0) {                  
        in[glb_id] = in[glb_id] + block[grp_id-1];  // Adds biggest element of previous block to current block
    }
}

// The actual scanning kernel
__kernel void scan(__global int *in, __global int *out, __global int *bout, __local int *buf, int n) {
    // Some variables we're going to use later
    size_t glb_id = get_global_id(0);
    size_t loc_id = get_local_id(0);
    size_t loc_sz = get_local_size(0);
    size_t grp_id = get_group_id(0);
  
    out[glb_id] = 0;                                // Clear output so different runs don't mess with each other
    buf[loc_id] = in[glb_id];                       // Copy data from input array to local buffer
    barrier(CLK_LOCAL_MEM_FENCE);                   // Barrier to keep it in local memory
    
    // for(int i=1; i<=loc_id; i*=2) {
    //     for(int j=0; j<loc_id; j=j+i+1) {
    //          += buf[j] + buf[j+i];                      
    //     }
    // }
    for(int step=1; step<loc_id; step<<=2) {
        for(int i=0; i<loc_id; i=i+step+step) {
            buf[i] += buf[i+step];
        }
    }
    out[glb_id] = buf[0];
/*
i           step
0 = 0 + 1   1
2 = 2 + 3   1
4 = 4 + 5   1
6 = 6 + 7   1
0 = 0 + 2   2
4 = 4 + 6   2
0 = 0 + 4   4


for int i = 1; i < loc_id; i *= 2
    for int j = 0; j < loc_id; j += i
        buf[j] + buf[j+i]

    out[glb_id + i] = buf[i] + buf[i + 1]
*/

    // Write the scanned data to output
    if(loc_id == loc_sz-1) {
        bout[grp_id] = out[glb_id];                 // Write the biggest data to bout; this will be used in the update
    }
}



