// pk910: native (non-emscripten) stubs for the copy-on-write fork's JS bridge,
// so the fork can be exercised under gdb. Compiled only for PK_FORK native
// debug builds; under emscripten these live in scripts/blink-lib.js instead.
#include "blink/builtin.h"
#if defined(PK_FORK) && !defined(__EMSCRIPTEN__)
#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

struct PkDead { int pid; int code; };
static struct PkDead g_dead[512];
static int g_ndead;
static int g_nextpid = 30001;

int js_kernel_pipe(int *out) {
  return pipe(out);
}

int js_vfork_dead(int code) {
  int p = g_nextpid++;
  if (g_ndead < 512) {
    g_dead[g_ndead].pid = p;
    g_dead[g_ndead].code = code & 255;
    ++g_ndead;
  }
  return p;
}

int js_vfork_wait(int pid, int nohang, int *code_out) {
  int i = 0, j;
  (void)nohang;
  if (g_ndead > 0) {
    if (pid > 0) {
      for (i = 0; i < g_ndead; ++i)
        if (g_dead[i].pid == pid) break;
      if (i >= g_ndead) return -ECHILD;
    }
    *code_out = g_dead[i].code;
    pid = g_dead[i].pid;
    for (j = i; j < g_ndead - 1; ++j) g_dead[j] = g_dead[j + 1];
    --g_ndead;
    return pid;
  }
  return -ECHILD;
}

int js_vfork_exec(const char *prog, char **argv, char **envp, int f0, int f1,
                  int f2) {
  (void)prog, (void)argv, (void)envp, (void)f0, (void)f1, (void)f2;
  return -ENOSYS;  // native debug focuses on fork+exit (subshells); no exec
}
#endif
