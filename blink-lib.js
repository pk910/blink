// Emscripten JS library for the blink wasm build (pk910.de terminal).
//
// emscripten_sleep: blink's port calls it in blocking-read poll loops and
// (patched) nanosleep, which normally requires ASYNCIFY. We always run in
// a worker, where synchronous blocking is legal - and when the pk910.de
// x86 runtime is present, ksys() parks the sleep in the kernel instead:
// interruptible by signals and billed as blocked time, not CPU.
//
// js_vfork_* / js_kernel_pipe: the sequential-vfork bridge (see the
// __EMSCRIPTEN__ block in blink/syscall.c). They call into the
// worker-global __pkx object defined by x86-runtime.js; without it
// (plain node harness) they report ENOSYS-style failures.
addToLibrary({
  emscripten_sleep: function (ms) {
    if (typeof ksys === 'function') {
      try {
        ksys('sleep', [ms]);
        return;
      } catch (e) {
        if (e && e.__exit) throw e; // fatal signal during the sleep
      }
    }
    if (typeof SharedArrayBuffer !== 'undefined') {
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
    } else {
      var end = Date.now() + ms;
      while (Date.now() < end) { /* last resort busy wait */ }
    }
  },

  js_kernel_pipe__deps: ['$FS'],
  js_kernel_pipe: function (out) {
    if (typeof __pkx === 'undefined') return -1;
    try {
      var fds = __pkx.kpipe(FS);
      HEAP32[out >> 2] = fds[0];
      HEAP32[(out >> 2) + 1] = fds[1];
      return 0;
    } catch (e) {
      if (e && e.__exit) throw e;
      return -1;
    }
  },

  js_vfork_exec__deps: ['$FS'],
  js_vfork_exec: function (prog, argv, envp, f0, f1, f2) {
    if (typeof __pkx === 'undefined') return -52; // ENOSYS (WASI numbering)
    var args = [];
    var env = {};
    var i;
    var p;
    for (i = 0; (p = HEAP32[(argv >> 2) + i]); i++) args.push(UTF8ToString(p));
    for (i = 0; (p = HEAP32[(envp >> 2) + i]); i++) {
      var s = UTF8ToString(p);
      var eq = s.indexOf('=');
      if (eq > 0) env[s.slice(0, eq)] = s.slice(eq + 1);
    }
    return __pkx.vforkExec(FS, UTF8ToString(prog), args, env, f0, f1, f2);
  },

  js_vfork_dead: function (code) {
    if (typeof __pkx === 'undefined') return 32767;
    return __pkx.deadChild(code);
  },

  js_vfork_wait: function (pid, nohang, code_out) {
    if (typeof __pkx === 'undefined') return -12; // ECHILD (WASI numbering)
    var r = __pkx.vwait(pid, !!nohang);
    if (r.err) return -r.err;
    HEAP32[code_out >> 2] = r.code | 0;
    return r.pid | 0;
  },
});
