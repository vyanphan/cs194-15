/*  This function takes in:
	  - a pointer to the vector of floats we want to increment (Y)
	  - the maximum id we want to reach (n)
    line 10:	idx = the first work-item ID in the group of global work-items executing the kernel.
   			 	(a work-item is a collection of parallel executions of a kernel)
   	line 12:	perform the adding function
*/

__kernel void vvadd (__global float *Y, __global float *A, __global float *B, int n) {
	int idx = get_global_id(0);
  	if(idx < n) {								
      	Y[idx] = A[idx] + B[idx];
    }
}
