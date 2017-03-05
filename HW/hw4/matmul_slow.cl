/*  This function takes in:
      - a pointer to the vector of floats we want to increment (Y)
      - the maximum id we want to reach (n)
    We perform matrix multiplication on this particular work group only. This work group
    is split up into tiles so it doesn't have to do so many off-chip memory accesses.
*/

__kernel void matmul(__global float *Y, __global float *A, __global float *B, int n) {
    int s = 16;                             // tile size (see matmul.cpp:64, local work size)
    int r = get_local_id(0);                // local row for this work group
    int c = get_local_id(1);                // local col for this work group
    int globR = s*get_group_id(0) + r;      // global row for Y, output matrix
    int globC = s*get_group_id(1) + c;      // global col for Y, output matrix
    
    __local float Atile[16][16];            // Tiling A into smaller blocks size 16x16
    __local float Btile[16][16];            // Tiling B into smaller blocks size 16x16
    float sum = 0.0f;                       // this is our end value
    
    int numTiles = n/s;                     // number of tiles = total size/tile size
    for(int t=0; t<numTiles; t++) {         // Loop over all tiles
        int rTile = s*t + r;                // index tile row within local row 
        int cTile = s*t + c;                // index tile col within local col
        Atile[r][c] = A[globR*n + cTile];   // fill up tiles of matrix A
        Btile[r][c] = B[rTile*n + globC];   // fill up tiles of matrix B
        
        barrier(CLK_LOCAL_MEM_FENCE);       // barrier so tiles don't mess with each other
        for(int k=0; k<s; k++) {            // matmul the tiles
            sum += Atile[r][k] * Btile[k][c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } 
    Y[globR*n + globC] = sum;               // final result
}
