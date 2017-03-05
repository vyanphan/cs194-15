#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "counters.h"

void copyArray(int *dst, int *src, size_t nbytes) {
  	for(int i=0; i<nbytes/4; i++) {
      dst[i] = src[i];
    }
}