__kernel void reduce(__global int *in, __global int *out, __local int *buf, int n) {
	// Some variables we're going to use later
	size_t glb_id = get_global_id(0);
	size_t loc_id = get_local_id(0); 
	size_t grp_id = get_group_id(0);
	size_t loc_sz = get_local_size(0);
	
	// Copy from input to local memory; barrier to keep it in local memory
	buf[loc_id] = in[glb_id]; 
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Reduce on each element
	for(int i = loc_sz / 2; i > 0; i /= 2) {
		if (loc_id < i) {
			buf[loc_id] += buf[loc_id + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// Write final reduced value to output
	if (loc_id == 0) {
		out[grp_id] = buf[0];
	}
}