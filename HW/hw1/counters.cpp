
#include "counters.h"

static void initCounter(hwCounter_t &x, uint64_t counterConfig)
{
  if(!x.init) 
    {
      memset(&(x.attr), 0, sizeof(x.attr));
      x.attr.disabled = 0;
      x.attr.type = PERF_TYPE_HARDWARE;
      x.attr.config =  counterConfig;
      pid_t pid = 0; 
      int cpu = -1;
      int group_fd = -1; 
      unsigned long flags = 0;
      x.fd = syscall(__NR_perf_event_open, &(x.attr), pid, 
		     cpu, group_fd, flags);
      assert(x.fd >= 0);
      x.init = true;
    }
}

void initTicks(hwCounter_t &x)
{
  initCounter(x,PERF_COUNT_HW_CPU_CYCLES);
}
void initInsns(hwCounter_t &x)
{
  initCounter(x,PERF_COUNT_HW_INSTRUCTIONS);
}

uint64_t getTicks(hwCounter_t &x)
{
  int rc;

  if(!x.init) 
    {
      initTicks(x);
    }
  
  rc = read(x.fd,&(x.buffer),sizeof(x.buffer));
  assert(rc == sizeof(x.buffer));
  return x.buffer;
}



uint64_t getInsns(hwCounter_t &x)
{
  int rc;

  if(!x.init) 
    {
      initInsns(x);
    }
  
  rc = read(x.fd,&(x.buffer),sizeof(x.buffer));
  assert(rc == sizeof(x.buffer));
  return x.buffer;
}
