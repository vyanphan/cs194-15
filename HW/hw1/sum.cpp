#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>
#include "counters.h"

int sumA(int n) {  
    long long sum = 0;   
    for(long long i=0; i<n; i++) {
        sum += i;
    }
    return sum;
}

int sumB() {
    long long sum = 0;
    for(long long i=0; i<10000; i++) {
        sum += i;
    }
    return sum;
}
