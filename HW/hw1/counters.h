#ifndef __COUNTERS_H__
#define __COUNTERS_H__

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <errno.h>
#include <unistd.h>
#include <cassert>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <stdint.h>

typedef struct
{
  bool init;
  int fd;
  struct perf_event_attr attr;
  uint64_t buffer;
} hwCounter_t;

void initTicks(hwCounter_t &x);
void initInsns(hwCounter_t &x);

uint64_t getTicks(hwCounter_t &x);
uint64_t getInsns(hwCounter_t &x);

#endif
