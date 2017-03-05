#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>
#include "counters.h"

int* randIntArray(int n) {
   
  //Create random array
  int *arr = new int[n];
  for(int i=0; i<n; i++) {
    arr[i] = i;
  }
  for(int i=n-1; i>=0; i--) {
    int j = rand() % (i+1);
    int temp = arr[j];
    arr[j] = arr[i];
    arr[i] = temp;
  }

  return arr;
  //Pointer chasing...I'm using p as the updated pointer instead of i here
  // int p = 0;
  // for(int i=0; i<(1<<20); i++) {
  //   p = arr[p];
  // }
  // return p;
}

int numDistinctValues(int n) {
  //Create random array of length n
  int arr[n];
  for(int i=0; i<n; i++) {
    arr[i] = i;
  }
  for(int i=n-1; i>=0; i--) {
    int j = rand() % (i+1);
    int temp = arr[j];
    arr[j] = arr[i];
    arr[i] = temp;
  }

  //Create another array to store visited values
  int visited[n];
  int numDistinct = 0;

  //Pointer chasing...I'm using p as the updated pointer instead of i here
  int p = 0;
  for(int i=0; i<(1<<20); i++) {
    if(visited[p] != 1) {
      visited[p] = 1;
      numDistinct++;
    }     
    p = arr[p];
  }
  return numDistinct;
}
