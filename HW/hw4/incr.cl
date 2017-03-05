/*  This function takes in:
	  - a pointer to the vector of floats we want to increment (Y)
	  - the maximum id we want to reach (n)
    line 11:	idx = the first work-item ID in the group of global work-items executing the kernel.
   			 	(a work-item is a collection of parallel executions of a kernel)
   	line 13:	only if idx has not exceeded the maximum id (think of it as an index in our vector)
   				do we increment the value of our vector at that index (idx) by 1
*/

__kernel void incr(__global float *Y, int n) {	
	int idx = get_global_id(0);
  	if(idx < n) {								
      	Y[idx] = Y[idx] + 1.0f;
    }
}
