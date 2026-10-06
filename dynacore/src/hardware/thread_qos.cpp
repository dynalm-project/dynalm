#include "dynacore/hardware/thread_qos.h"

#include "dynacore/base/platform.h"

#if ENGINE_OS_WINDOWS
#include <windows.h>
#elif ENGINE_OS_MACOS
#include <pthread.h>
#include <pthread/qos.h>
#elif ENGINE_OS_LINUX
#include <pthread.h>
#include <sched.h>
#endif

namespace dynacore {

void request_full_speed_process() {
#if ENGINE_OS_WINDOWS
  // ControlMask selects the policy; StateMask 0 = "do not throttle".
  PROCESS_POWER_THROTTLING_STATE s{};
  s.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  s.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  s.StateMask = 0;
  SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &s, sizeof(s));
#endif
}

void request_full_speed_thread() {
#if ENGINE_OS_WINDOWS
  THREAD_POWER_THROTTLING_STATE s{};
  s.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
  s.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
  s.StateMask = 0;
  SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &s, sizeof(s));
#elif ENGINE_OS_MACOS
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
}

bool pin_current_thread(int cpu) {
  if (cpu < 0) return false;
#if ENGINE_OS_WINDOWS
  if (cpu >= 64) return false;
  return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu) != 0;
#elif ENGINE_OS_LINUX
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
  (void)cpu;
  return false;
#endif
}

}  // namespace dynacore
