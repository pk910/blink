// Emscripten JS library for the blink wasm build (pk910.de terminal).
//
// The pk910 direct-ksys syscall layer lives HERE (compiled into the glue) so
// the SAME functions run on EVERY thread - thread 0 and pthreads - locally,
// with no proxyToMainThread. Each thread exposes its own global `ksys` (a
// SharedArrayBuffer channel to the pk910 kernel; set up in x86-runtime.js).
// blink's ~46 host syscalls are overridden to talk straight to the kernel VFS:
// host fd == kernel fd, no emscripten overlay FS. Struct/dirent layouts mirror
// emscripten's own library_syscall marshaling.
//
// HEAP views (HEAPU8/HEAPU32/HEAP32/HEAP16/HEAP64) are the glue's globals; the
// wasm memory is a shared growable SAB so they stay valid. TextDecoder rejects
// views over shared memory, so C-strings/reads copy via .slice first.
//
// js_vfork_* / js_kernel_pipe reach the process-level fork/pipe bridge in
// x86-runtime.js via globalThis.__pkx. NONE of these proxy any more (each thread
// calls the kernel directly through its own ksys).

addToLibrary({
  // ── process exit ──
  // blink's guest exit_group -> _Exit -> proc_exit. Under PROXY_TO_PTHREAD
  // emscripten's own proc_exit never releases (keepRuntimeAlive stays true on the
  // pump thread), so onExit never fires and the process would linger. Tell the
  // kernel to reap us directly; ksys blocks until the kernel terminates the
  // worker(s), which never returns.
  proc_exit__proxy: 'none',
  proc_exit: function (code) {
    // The guest's exit_group reaches proc_exit on the main-runner PTHREAD, where
    // it must tell the kernel to reap (under PROXY_TO_PTHREAD onExit never fires -
    // keepRuntimeAlive stays true on the pump). Thread 0 also runs proc_exit
    // during teardown; reaping there would kill the process before main() runs, so
    // only the pthread path reaps.
    if (typeof ENVIRONMENT_IS_PTHREAD !== 'undefined' && ENVIRONMENT_IS_PTHREAD && typeof ksys === 'function') {
      try { ksys('procexit', [code | 0]); } catch (e) { /* being torn down */ }
    }
  },

  // ── blocking sleep: park in the kernel (interruptible, billed as blocked) ──
  emscripten_sleep__proxy: 'none',
  emscripten_sleep: function (ms) {
    if (typeof ksys === 'function') {
      try { ksys('sleep', [ms]); return; } catch (e) { if (e && e.__exit) throw e; }
    }
    if (typeof SharedArrayBuffer !== 'undefined') {
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
    } else {
      var end = Date.now() + ms;
      while (Date.now() < end) { /* last resort busy wait */ }
    }
  },

  // ── wasm JIT loader (pk910) ──
  // Compile a generated block module and install its exported function ("b")
  // into the shared wasm function table; return the table index, which is a
  // valid nexgen32e_f function pointer for blink's dispatch (a call_indirect on
  // the wasm ABI). Runs locally on every thread; the table is the module's own
  // __indirect_function_table so the index is directly callable by blink.
  // jldIdx is JitlessDispatch's table index, imported by the stub as env.jld.
  // See vendor/blink/blink/wasmjit.c and ai_plans/wasm-jit.md.
  pk_jit_install__proxy: 'none',
  pk_jit_install: function (ptr, len, firstCompile) {
    try {
      // the block imports the shared memory (reads/writes the Machine struct) and
      // the shared function table (call_indirect to blink's static Op handlers).
      var mod = new WebAssembly.Module(HEAPU8.slice(ptr, ptr + len));
      var inst = new WebAssembly.Instance(mod, { env: { mem: wasmMemory, tbl: wasmTable } });
      var idx = wasmTable.grow(1); // returns the previous length = the new slot
      wasmTable.set(idx, inst.exports.b);
      if (typeof ksys === 'function') {
        // firstCompile: the one thread that emitted this block's shared bytes.
        // (bytes are emitted once per ip; every thread instantiates its own slot)
        if (firstCompile) {
          // firstCompile: emit sequence number; negative marks a self-loop block.
          var sl = firstCompile < 0, fc = Math.abs(firstCompile);
          if (sl || fc === 1 || fc === 25 || fc % 250 === 0) { try { ksys('klog', ['wasmjit: emitted ' + (sl ? 'SELF-LOOP' : 'block') + ' #' + fc]); } catch (x) {} }
        }
        var n = (globalThis.__pkJitN = (globalThis.__pkJitN || 0) + 1);
        if (n === 1 || n % 500 === 0) { try { ksys('klog', ['wasmjit: instantiated ' + n + ' (this thread)']); } catch (x) {} }
      }
      return idx;
    } catch (e) {
      try { ksys('klog', ['wasmjit install failed: ' + (e && (e.stack || e.message) ? String(e.stack || e.message).slice(0, 200) : e)]); } catch (x) {}
      return 0;
    }
  },

  // ── direct-ksys support object (one per thread; holds per-thread state) ──
  // the runtime's ksys hands kernel-raised guest signals (alarm) to blink's
  // per-thread Machine through this hook once the wasm is up
  $PKSYS__postset: "globalThis.__pkEnqueueSignal = function (sig) { if (typeof _pk_enqueue_signal === 'function') _pk_enqueue_signal(sig); };",
  $PKSYS: {
    _dec: null,
    _enc: null,
    dirPos: {},
    // fd -> { u8, pos, len }: read-only files the kernel copied into a shared
    // SAB at open time, so read/seek/pread run locally with no kernel round-trip.
    cache: {},
    _sabU8: {},
    sabU8: function (i) {
      if (!PKSYS._sabU8[i]) PKSYS._sabU8[i] = new Uint8Array(globalThis.__pkFileSabs[i]);
      return PKSYS._sabU8[i];
    },
    umaskVal: 18, // 0o022
    rawActive: false,
    termios: {
      c_iflag: 0o25156 & 0xffff, c_oflag: 5, c_cflag: 191, c_lflag: 35387,
      c_cc: [3, 28, 127, 21, 4, 0, 1, 0, 17, 19, 26, 0, 18, 15, 23, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    dec: function () { return PKSYS._dec || (PKSYS._dec = new TextDecoder()); },
    enc: function () { return PKSYS._enc || (PKSYS._enc = new TextEncoder()); },
    // emscripten (WASI) errno numbering; a recognized name is an expected result
    wasiErrno: function (err) {
      var name = String((err && err.message) || err).split(':')[0];
      var map = {
        ENOENT: 44, EACCES: 2, EISDIR: 31, ENOTDIR: 54, EEXIST: 20, EPERM: 63, ENOSPC: 51, EBADF: 8, EPIPE: 64, EINVAL: 28, ESPIPE: 70, ENOTTY: 59,
        // sockets (WASI numbering; blink XlatErrno's them to Linux)
        EAGAIN: 6, EINPROGRESS: 26, EALREADY: 7, ECONNREFUSED: 14, ECONNRESET: 15, ECONNABORTED: 13, EADDRINUSE: 3, EADDRNOTAVAIL: 4,
        ENOTSOCK: 57, EAFNOSUPPORT: 5, EPROTONOSUPPORT: 66, ESOCKTNOSUPPORT: 66, EOPNOTSUPP: 138, ENOTSUP: 58, EHOSTUNREACH: 23, ENETUNREACH: 40, ENETDOWN: 39,
        ETIMEDOUT: 73, EISCONN: 30, ENOTCONN: 53, EMSGSIZE: 35, ENOBUFS: 42, EDESTADDRREQ: 17, ENOPROTOOPT: 50, EINTR: 27, ENODEV: 43, ENXIO: 60,
        ESRCH: 71, ENOSYS: 52, ESHUTDOWN: 15, EFAULT: 21, EIO: 29,
      };
      return map[name] || 29; // default EIO
    },
    // a kernel SysError (.sys) is an expected errno; anything else is a bug in
    // this layer - surface it to the kernel log (dmesg), never swallow it
    klog: function (e) {
      if (e && e.sys) return;
      try { ksys('klog', ['x86 syscall fault: ' + (e && (e.stack || e.message) ? String(e.stack || e.message).slice(0, 300) : e)]); } catch (x) { /* */ }
    },
    errW: function (e) { if (e && e.__exit) throw e; PKSYS.klog(e); return PKSYS.wasiErrno(e); }, // fd_* : positive
    errS: function (e) { if (e && e.__exit) throw e; PKSYS.klog(e); return -PKSYS.wasiErrno(e); }, // __syscall_* : negative
    cstr: function (ptr) {
      var end = ptr;
      while (HEAPU8[end]) end++;
      return PKSYS.dec().decode(HEAPU8.slice(ptr, end));
    },
    // AT_FDCWD (-100) or absolute -> kernel resolves vs cwd; real dirfd -> prepend
    atPath: function (dirfd, ptr) {
      var p = PKSYS.cstr(ptr);
      if (p[0] === '/' || (dirfd | 0) === -100) return p;
      var base = ksys('fdpath', [dirfd]);
      return base === '/' ? '/' + p : base + '/' + p;
    },
    writeStat: function (buf, st) {
      var typeBits = st.type === 'dir' ? 0o40000 : st.type === 'char' ? 0o20000 : st.type === 'fifo' ? 0o10000 : st.type === 'link' ? 0o120000 : st.type === 'sock' ? 0o140000 : 0o100000;
      var mode = typeBits | (st.mode & 0o7777);
      var secs = BigInt(Math.floor((st.mtime || 0) / 1000));
      HEAPU32[buf >> 2] = 910; // dev
      HEAPU32[(buf + 4) >> 2] = mode;
      HEAPU32[(buf + 8) >> 2] = 1; // nlink
      HEAPU32[(buf + 12) >> 2] = 0; // uid (guest runs as root)
      HEAPU32[(buf + 16) >> 2] = 0; // gid
      HEAPU32[(buf + 20) >> 2] = 0; // rdev
      HEAP64[(buf + 24) >> 3] = BigInt(st.size); // size
      HEAP32[(buf + 32) >> 2] = 4096; // blksize
      HEAP32[(buf + 36) >> 2] = Math.ceil(st.size / 512); // blocks
      HEAP64[(buf + 40) >> 3] = secs; HEAPU32[(buf + 48) >> 2] = 0; // atime
      HEAP64[(buf + 56) >> 3] = secs; HEAPU32[(buf + 64) >> 2] = 0; // mtime
      HEAP64[(buf + 72) >> 3] = secs; HEAPU32[(buf + 80) >> 2] = 0; // ctime
      HEAP64[(buf + 88) >> 3] = 1n; // ino
      return 0;
    },
    statfs: function (buf) {
      // blink fills the HOST (wasm32 musl) struct statfs then copies to the guest:
      // unsigned long is 4 bytes, fsblkcnt_t/fsfilcnt_t are 8 bytes (8-aligned).
      //   f_type u32@0  f_bsize u32@4  f_blocks u64@8  f_bfree u64@16  f_bavail u64@24
      //   f_files u64@32  f_ffree u64@40  f_fsid@48  f_namelen u32@56  f_frsize u32@60
      // The VFS is virtual, so report a plausible fixed-size fs so `df` shows real numbers.
      HEAPU8.fill(0, buf, buf + 88);
      var U32 = function (off, v) { HEAPU32[(buf + off) >> 2] = v >>> 0; };
      var U64 = function (off, v) { HEAPU32[(buf + off) >> 2] = v >>> 0; HEAPU32[(buf + off + 4) >> 2] = 0; };
      U32(0, 0x01021994);  // f_type = TMPFS_MAGIC
      U32(4, 4096);        // f_bsize
      U64(8, 262144);      // f_blocks = 1 GiB / 4096
      U64(16, 236000);     // f_bfree
      U64(24, 236000);     // f_bavail
      U64(32, 262144);     // f_files
      U64(40, 250000);     // f_ffree
      U32(56, 255);        // f_namelen
      U32(60, 4096);       // f_frsize
      return 0;
    },
    statPath: function (path, buf) {
      try { return PKSYS.writeStat(buf, ksys('stat', [path])); } catch (e) { return PKSYS.errS(e); }
    },
    read: function (fd, iov, iovcnt, pnum) {
      var total = 0;
      var c = PKSYS.cache[fd];
      if (c) {
        // shared-SAB file: copy straight out of the SAB, no kernel round-trip
        for (var j = 0; j < iovcnt; j++) {
          var p = HEAPU32[iov >> 2];
          var l = HEAPU32[(iov + 4) >> 2];
          iov += 8;
          if (l === 0) continue;
          var n = Math.min(l, c.len - c.pos);
          if (n <= 0) break;
          HEAPU8.set(c.u8.subarray(c.pos, c.pos + n), p);
          c.pos += n;
          total += n;
          if (n < l) break;
        }
        HEAPU32[pnum >> 2] = total;
        return 0;
      }
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAPU32[iov >> 2];
        var len = HEAPU32[(iov + 4) >> 2];
        iov += 8;
        if (len === 0) continue;
        var chunk = ksys('read', [fd, len]);
        if (chunk === null) break; // EOF
        var bytes = chunk instanceof Uint8Array ? chunk : PKSYS.enc().encode(chunk);
        if (bytes.length === 0) break; // timeout / no data
        HEAPU8.set(bytes.subarray(0, len), ptr);
        total += Math.min(bytes.length, len);
        if (bytes.length < len) break; // short read - stop here
      }
      HEAPU32[pnum >> 2] = total;
      return 0;
    },
    writev: function (fd, iov, iovcnt, pnum) {
      var total = 0;
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAPU32[iov >> 2];
        var len = HEAPU32[(iov + 4) >> 2];
        iov += 8;
        if (len === 0) continue;
        total += ksys('write', [fd, HEAPU8.slice(ptr, ptr + len)]);
      }
      HEAPU32[pnum >> 2] = total;
      return 0;
    },
    getdents: function (fd, dirp, count) {
      var entries = PKSYS.dirPos[fd];
      if (!entries) {
        entries = { list: [{ name: '.', type: 'dir' }, { name: '..', type: 'dir' }].concat(ksys('readdirfd', [fd])), idx: 0 };
        PKSYS.dirPos[fd] = entries;
      }
      var pos = 0;
      while (entries.idx < entries.list.length) {
        var e = entries.list[entries.idx];
        var nameBytes = PKSYS.enc().encode(e.name);
        var aligned = (19 + nameBytes.length + 1 + 7) & ~7;
        if (pos + aligned > count) break;
        var type = e.type === 'dir' ? 4 : e.type === 'link' ? 10 : 8;
        HEAP64[(dirp + pos) >> 3] = BigInt(entries.idx + 1); // d_ino
        HEAP64[(dirp + pos + 8) >> 3] = BigInt(entries.idx + 1); // d_off
        HEAP16[(dirp + pos + 16) >> 1] = aligned; // d_reclen
        HEAPU8[dirp + pos + 18] = type; // d_type
        HEAPU8.set(nameBytes, dirp + pos + 19);
        HEAPU8[dirp + pos + 19 + nameBytes.length] = 0;
        pos += aligned;
        entries.idx++;
      }
      return pos;
    },
    // pollfd (wasm32): i32 fd, i16 events, i16 revents. The kernel parks the
    // thread until something is ready or the timeout passes (-1 = forever).
    poll: function (fds, nfds, timeout) {
      var reqs = [];
      var i;
      for (i = 0; i < nfds; i++) {
        var base = fds + i * 8;
        reqs.push({ fd: HEAP32[base >> 2], events: HEAP16[(base + 4) >> 1] & 0xffff });
      }
      var revents = ksys('poll', [reqs, timeout === undefined ? -1 : timeout]);
      var ready = 0;
      for (i = 0; i < nfds; i++) {
        HEAP16[(fds + i * 8 + 6) >> 1] = revents[i] | 0;
        if (revents[i]) ready++;
      }
      return ready;
    },
    // ── sockets: host (wasm32 musl) sockaddr layouts <-> the kernel's SockAddr JSON ──
    readSockaddr: function (ptr, len) {
      if (!ptr || len < 2) return { fam: 'unspec' };
      var fam = HEAPU16[ptr >> 1];
      if (fam === 2) {
        if (len < 16) throw Object.assign(new Error('EINVAL: short sockaddr_in'), { sys: true });
        return { fam: 'inet', addr: HEAPU8[ptr + 4] + '.' + HEAPU8[ptr + 5] + '.' + HEAPU8[ptr + 6] + '.' + HEAPU8[ptr + 7], port: (HEAPU8[ptr + 2] << 8) | HEAPU8[ptr + 3] };
      }
      if (fam === 1) {
        var end = ptr + Math.min(len, 110);
        var start = ptr + 2;
        var abstract = HEAPU8[start] === 0 && end > start + 1;
        var p = abstract ? start + 1 : start;
        var q = p;
        while (q < end && HEAPU8[q] !== 0) q++;
        var path = PKSYS.dec().decode(HEAPU8.slice(p, q));
        return { fam: 'unix', path: abstract ? '@' + path : path };
      }
      if (fam === 16) return { fam: 'netlink', pid: HEAPU32[(ptr + 4) >> 2], groups: HEAPU32[(ptr + 8) >> 2] };
      if (fam === 0) return { fam: 'unspec' };
      throw Object.assign(new Error('EAFNOSUPPORT: address family ' + fam), { sys: true });
    },
    // writes into the caller's buffer (capacity *lenPtr), stores the full size in *lenPtr
    writeSockaddr: function (ptr, lenPtr, a) {
      if (!ptr || !lenPtr) return;
      var cap = HEAPU32[lenPtr >> 2];
      var buf;
      if (!a || a.fam === 'unspec') { buf = new Uint8Array(16); }
      else if (a.fam === 'inet') {
        buf = new Uint8Array(16);
        buf[0] = 2; buf[2] = (a.port >> 8) & 0xff; buf[3] = a.port & 0xff;
        var parts = String(a.addr).split('.');
        for (var i = 0; i < 4; i++) buf[4 + i] = parseInt(parts[i], 10) & 0xff;
      } else if (a.fam === 'unix') {
        var abstract = a.path.charAt(0) === '@';
        var pathBytes = PKSYS.enc().encode(abstract ? a.path.slice(1) : a.path);
        buf = new Uint8Array(2 + (abstract ? 1 : 0) + pathBytes.length + 1);
        buf[0] = 1;
        buf.set(pathBytes, abstract ? 3 : 2);
      } else if (a.fam === 'netlink') {
        buf = new Uint8Array(12);
        buf[0] = 16;
        buf[4] = a.pid & 0xff; buf[5] = (a.pid >> 8) & 0xff; buf[6] = (a.pid >> 16) & 0xff; buf[7] = (a.pid >>> 24) & 0xff;
        buf[8] = a.groups & 0xff; buf[9] = (a.groups >> 8) & 0xff; buf[10] = (a.groups >> 16) & 0xff; buf[11] = (a.groups >>> 24) & 0xff;
      } else { buf = new Uint8Array(16); }
      HEAPU8.set(buf.subarray(0, Math.min(cap, buf.length)), ptr);
      HEAPU32[lenPtr >> 2] = buf.length;
    },
    // msghdr (wasm32): name @0, namelen @4, iov @8, iovlen @12, control @16, controllen @20, flags @24
    gather: function (iov, iovcnt) {
      var parts = [];
      var total = 0;
      for (var i = 0; i < iovcnt; i++) {
        var p = HEAPU32[(iov + i * 8) >> 2];
        var l = HEAPU32[(iov + i * 8 + 4) >> 2];
        if (l) { parts.push(HEAPU8.slice(p, p + l)); total += l; }
      }
      var out = new Uint8Array(total);
      var o = 0;
      for (var j = 0; j < parts.length; j++) { out.set(parts[j], o); o += parts[j].length; }
      return out;
    },
    scatter: function (iov, iovcnt, data) {
      var off = 0;
      for (var i = 0; i < iovcnt && off < data.length; i++) {
        var p = HEAPU32[(iov + i * 8) >> 2];
        var l = HEAPU32[(iov + i * 8 + 4) >> 2];
        var n = Math.min(l, data.length - off);
        HEAPU8.set(data.subarray(off, off + n), p);
        off += n;
      }
      return off;
    },
    iovTotal: function (iov, iovcnt) {
      var t = 0;
      for (var i = 0; i < iovcnt; i++) t += HEAPU32[(iov + i * 8 + 4) >> 2];
      return t;
    },
    // getsockopt/setsockopt value marshaling: ints, timeval (16 bytes, Linux
    // amd64 layout passed through by blink), linger (8 bytes), raw bytes
    readOptVal: function (level, name, ptr, len) {
      if (level === 1 && name === 13 && len >= 8) return { on: HEAP32[ptr >> 2], linger: HEAP32[(ptr + 4) >> 2] };
      if (level === 1 && (name === 20 || name === 21) && len >= 8) return { sec: HEAP32[ptr >> 2], usec: len >= 16 ? HEAP32[(ptr + 8) >> 2] : HEAP32[(ptr + 4) >> 2] };
      if (len === 4) return HEAP32[ptr >> 2];
      if (len === 1) return HEAPU8[ptr];
      if (len === 2) return HEAP16[ptr >> 1];
      if (len === 8) return HEAP32[ptr >> 2];
      return Array.prototype.slice.call(HEAPU8.slice(ptr, ptr + len));
    },
    writeOptVal: function (v, ptr, lenPtr) {
      var cap = HEAPU32[lenPtr >> 2];
      if (v !== null && typeof v === 'object' && 'on' in v) {
        if (cap >= 8) { HEAP32[ptr >> 2] = v.on; HEAP32[(ptr + 4) >> 2] = v.linger; HEAPU32[lenPtr >> 2] = 8; }
        return;
      }
      if (v !== null && typeof v === 'object' && 'sec' in v) {
        if (cap >= 16) { HEAP32[ptr >> 2] = v.sec; HEAP32[(ptr + 4) >> 2] = 0; HEAP32[(ptr + 8) >> 2] = v.usec; HEAP32[(ptr + 12) >> 2] = 0; HEAPU32[lenPtr >> 2] = 16; }
        else if (cap >= 8) { HEAP32[ptr >> 2] = v.sec; HEAP32[(ptr + 4) >> 2] = v.usec; HEAPU32[lenPtr >> 2] = 8; }
        return;
      }
      if (Array.isArray(v) || (v && v.buffer)) {
        var bytes = v instanceof Uint8Array ? v : Uint8Array.from(v);
        HEAPU8.set(bytes.subarray(0, cap), ptr);
        HEAPU32[lenPtr >> 2] = Math.min(cap, bytes.length);
        return;
      }
      if (typeof v === 'string') v = PKSYS.wasiErrno(v); // SO_ERROR: an errno name
      if (cap >= 4) { HEAP32[ptr >> 2] = v | 0; HEAPU32[lenPtr >> 2] = 4; }
      else if (cap >= 1) { HEAPU8[ptr] = v & 0xff; HEAPU32[lenPtr >> 2] = 1; }
    },
    // SIOC* request numbers -> names the kernel knows (see net/abi/ioctl.ts)
    siocName: function (req) {
      var names = {
        0x890b: 'SIOCADDRT', 0x890c: 'SIOCDELRT', 0x8910: 'SIOCGIFNAME', 0x8912: 'SIOCGIFCONF', 0x8913: 'SIOCGIFFLAGS', 0x8914: 'SIOCSIFFLAGS',
        0x8915: 'SIOCGIFADDR', 0x8916: 'SIOCSIFADDR', 0x8917: 'SIOCGIFDSTADDR', 0x8918: 'SIOCSIFDSTADDR', 0x8919: 'SIOCGIFBRDADDR', 0x891a: 'SIOCSIFBRDADDR',
        0x891b: 'SIOCGIFNETMASK', 0x891c: 'SIOCSIFNETMASK', 0x891d: 'SIOCGIFMETRIC', 0x891e: 'SIOCSIFMETRIC', 0x8921: 'SIOCGIFMTU', 0x8922: 'SIOCSIFMTU',
        0x8924: 'SIOCSIFHWADDR', 0x8927: 'SIOCGIFHWADDR', 0x8933: 'SIOCGIFINDEX', 0x8942: 'SIOCGIFTXQLEN', 0x8943: 'SIOCSIFTXQLEN', 0x8970: 'SIOCGIFMAP', 0x8971: 'SIOCSIFMAP',
      };
      return names[req] || ('SIOC_' + req.toString(16));
    },
    applyTermios: function (data) {
      PKSYS.termios = { c_iflag: data.c_iflag, c_oflag: data.c_oflag, c_cflag: data.c_cflag, c_lflag: data.c_lflag, c_cc: data.c_cc || PKSYS.termios.c_cc };
      var wantRaw = (data.c_lflag & 2) === 0; // ICANON cleared = raw
      if (wantRaw !== PKSYS.rawActive) {
        try { ksys('ioctl', [0, 'raw', wantRaw]); PKSYS.rawActive = wantRaw; } catch (e) { if (e && e.__exit) throw e; }
      }
      return 0;
    },
    winsize: function () {
      try { var s = ksys('ioctl', [0, 'size']); return [s.rows, s.cols]; } catch (e) { if (e && e.__exit) throw e; return [24, 80]; }
    },
  },

  // ── files ──
  __syscall_openat__deps: ['$PKSYS'],
  __syscall_openat__proxy: 'none',
  __syscall_openat: function (dirfd, path, flags) {
    try {
      var r = ksys('openat', [PKSYS.atPath(dirfd, path), flags >>> 0]);
      if (r !== null && typeof r === 'object') {
        // shared-SAB read-only file: cache the SAB view so reads stay local
        if (r.sab >= 0) PKSYS.cache[r.fd] = { u8: PKSYS.sabU8(r.sab), pos: 0, len: r.len };
        return r.fd;
      }
      return r;
    } catch (e) { return PKSYS.errS(e); }
  },
  fd_close__deps: ['$PKSYS'],
  fd_close__proxy: 'none',
  fd_close: function (fd) {
    try { delete PKSYS.dirPos[fd]; delete PKSYS.cache[fd]; ksys('close', [fd]); return 0; } catch (e) { return PKSYS.errW(e); }
  },
  fd_read__deps: ['$PKSYS'],
  fd_read__proxy: 'none',
  fd_read: function (fd, iov, iovcnt, pnum) { try { return PKSYS.read(fd, iov, iovcnt, pnum); } catch (e) { return PKSYS.errW(e); } },
  fd_write__deps: ['$PKSYS'],
  fd_write__proxy: 'none',
  fd_write: function (fd, iov, iovcnt, pnum) { try { return PKSYS.writev(fd, iov, iovcnt, pnum); } catch (e) { return PKSYS.errW(e); } },
  fd_seek__deps: ['$PKSYS'],
  fd_seek__proxy: 'none',
  fd_seek: function (fd, offset, whence, newOffset) {
    var c = PKSYS.cache[fd];
    if (c) {
      var p = Number(offset);
      if (whence === 1) p += c.pos; else if (whence === 2) p += c.len;
      if (p < 0) p = 0;
      c.pos = p;
      HEAP64[newOffset >> 3] = BigInt(p);
      return 0;
    }
    try { var pos = ksys('seek', [fd, Number(offset), whence]); HEAP64[newOffset >> 3] = BigInt(pos); return 0; } catch (e) { return PKSYS.errW(e); }
  },
  fd_sync__proxy: 'none',
  fd_sync: function () { return 0; },
  __syscall_fdatasync__proxy: 'none',
  __syscall_fdatasync: function () { return 0; },
  fd_pread__deps: ['$PKSYS'],
  fd_pread__proxy: 'none',
  fd_pread: function (fd, iov, iovcnt, offset, pnum) {
    var c = PKSYS.cache[fd];
    if (c) { var saved = c.pos; c.pos = Number(offset); var r = PKSYS.read(fd, iov, iovcnt, pnum); c.pos = saved; return r; }
    try { ksys('seek', [fd, Number(offset), 0]); return PKSYS.read(fd, iov, iovcnt, pnum); } catch (e) { return PKSYS.errW(e); }
  },
  fd_pwrite__deps: ['$PKSYS'],
  fd_pwrite__proxy: 'none',
  fd_pwrite: function (fd, iov, iovcnt, offset, pnum) {
    try { ksys('seek', [fd, Number(offset), 0]); return PKSYS.writev(fd, iov, iovcnt, pnum); } catch (e) { return PKSYS.errW(e); }
  },
  fd_fdstat_get__deps: ['$PKSYS'],
  fd_fdstat_get__proxy: 'none',
  fd_fdstat_get: function (fd, pbuf) {
    try { HEAPU8[pbuf] = 4; HEAP16[(pbuf + 2) >> 1] = 0; HEAP64[(pbuf + 8) >> 3] = 0n; HEAP64[(pbuf + 16) >> 3] = 0n; return 0; } catch (e) { return PKSYS.errW(e); }
  },
  __syscall_ftruncate64__deps: ['$PKSYS'],
  __syscall_ftruncate64__proxy: 'none',
  __syscall_ftruncate64: function (fd, lo) { try { return ksys('ftruncate', [fd, lo >>> 0]); } catch (e) { return PKSYS.errS(e); } },

  // ── stat family ──
  __syscall_stat64__deps: ['$PKSYS'],
  __syscall_stat64__proxy: 'none',
  __syscall_stat64: function (path, buf) { return PKSYS.statPath(PKSYS.cstr(path), buf); },
  __syscall_lstat64__deps: ['$PKSYS'],
  __syscall_lstat64__proxy: 'none',
  __syscall_lstat64: function (path, buf) { return PKSYS.statPath(PKSYS.cstr(path), buf); },
  __syscall_newfstatat__deps: ['$PKSYS'],
  __syscall_newfstatat__proxy: 'none',
  __syscall_newfstatat: function (dirfd, path, buf) { return PKSYS.statPath(PKSYS.atPath(dirfd, path), buf); },
  __syscall_fstat64__deps: ['$PKSYS'],
  __syscall_fstat64__proxy: 'none',
  __syscall_fstat64: function (fd, buf) { try { return PKSYS.writeStat(buf, ksys('fstat', [fd])); } catch (e) { return PKSYS.errS(e); } },
  __syscall_statfs64__deps: ['$PKSYS'],
  __syscall_statfs64__proxy: 'none',
  __syscall_statfs64: function (_p, _sz, buf) { return PKSYS.statfs(buf); },
  __syscall_fstatfs64__deps: ['$PKSYS'],
  __syscall_fstatfs64__proxy: 'none',
  __syscall_fstatfs64: function (_fd, _sz, buf) { return PKSYS.statfs(buf); },

  // ── directories ──
  __syscall_getdents64__deps: ['$PKSYS'],
  __syscall_getdents64__proxy: 'none',
  __syscall_getdents64: function (fd, dirp, count) { try { return PKSYS.getdents(fd, dirp, count); } catch (e) { return PKSYS.errS(e); } },
  __syscall_mkdirat__deps: ['$PKSYS'],
  __syscall_mkdirat__proxy: 'none',
  __syscall_mkdirat: function (dirfd, path) { try { ksys('mkdir', [PKSYS.atPath(dirfd, path)]); return 0; } catch (e) { return PKSYS.errS(e); } },
  __syscall_unlinkat__deps: ['$PKSYS'],
  __syscall_unlinkat__proxy: 'none',
  __syscall_unlinkat: function (dirfd, path, flags) { try { ksys('unlink', [PKSYS.atPath(dirfd, path), (flags & 512) !== 0]); return 0; } catch (e) { return PKSYS.errS(e); } },
  __syscall_renameat__deps: ['$PKSYS'],
  __syscall_renameat__proxy: 'none',
  __syscall_renameat: function (od, op, nd, np) { try { ksys('rename', [PKSYS.atPath(od, op), PKSYS.atPath(nd, np)]); return 0; } catch (e) { return PKSYS.errS(e); } },
  __syscall_getcwd__deps: ['$PKSYS'],
  __syscall_getcwd__proxy: 'none',
  __syscall_getcwd: function (buf, size) {
    try { var cwd = PKSYS.enc().encode(ksys('cwd', [])); if (size < cwd.length + 1) return -68; HEAPU8.set(cwd, buf); HEAPU8[buf + cwd.length] = 0; return cwd.length + 1; } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_chdir__deps: ['$PKSYS'],
  __syscall_chdir__proxy: 'none',
  __syscall_chdir: function (path) { try { ksys('chdir', [PKSYS.cstr(path)]); return 0; } catch (e) { return PKSYS.errS(e); } },
  __syscall_fchdir__deps: ['$PKSYS'],
  __syscall_fchdir__proxy: 'none',
  __syscall_fchdir: function (fd) { try { ksys('chdir', [ksys('fdpath', [fd])]); return 0; } catch (e) { return PKSYS.errS(e); } },

  // ── metadata ──
  __syscall_chmod__deps: ['$PKSYS'],
  __syscall_chmod__proxy: 'none',
  __syscall_chmod: function (path, mode) { try { ksys('chmod', [PKSYS.cstr(path), mode & 0o7777]); return 0; } catch (e) { return PKSYS.errS(e); } },
  __syscall_fchmod__deps: ['$PKSYS'],
  __syscall_fchmod__proxy: 'none',
  __syscall_fchmod: function (fd, mode) { try { ksys('chmod', [ksys('fdpath', [fd]), mode & 0o7777]); return 0; } catch (e) { return PKSYS.errS(e); } },
  __syscall_fchmodat2__deps: ['$PKSYS'],
  __syscall_fchmodat2__proxy: 'none',
  __syscall_fchmodat2: function (dirfd, path, mode) { try { ksys('chmod', [PKSYS.atPath(dirfd, path), mode & 0o7777]); return 0; } catch (e) { return PKSYS.errS(e); } },
  __syscall_faccessat__deps: ['$PKSYS'],
  __syscall_faccessat__proxy: 'none',
  __syscall_faccessat: function (dirfd, path) { try { ksys('stat', [PKSYS.atPath(dirfd, path)]); return 0; } catch (e) { return PKSYS.errS(e); } },
  __syscall_readlinkat__proxy: 'none',
  __syscall_readlinkat: function () { return -22; }, // EINVAL: no symlinks in the VFS
  __syscall_symlinkat__proxy: 'none',
  __syscall_symlinkat: function () { return -1; }, // EPERM
  __syscall_linkat__proxy: 'none',
  __syscall_linkat: function () { return -1; }, // EPERM
  __syscall_utimensat__proxy: 'none',
  __syscall_utimensat: function () { return 0; }, // accepted, not stored
  __syscall_fchown32__proxy: 'none',
  __syscall_fchown32: function () { return 0; },
  __syscall_fchownat__proxy: 'none',
  __syscall_fchownat: function () { return 0; },
  __syscall_umask__deps: ['$PKSYS'],
  __syscall_umask__proxy: 'none',
  __syscall_umask: function (m) { var prev = PKSYS.umaskVal; PKSYS.umaskVal = m & 0o777; return prev; },

  // ── fds / ids / pipes ──
  __syscall_dup__deps: ['$PKSYS'],
  __syscall_dup__proxy: 'none',
  __syscall_dup: function (fd) { try { return ksys('dup', [fd]); } catch (e) { return PKSYS.errS(e); } },
  __syscall_dup3__deps: ['$PKSYS'],
  __syscall_dup3__proxy: 'none',
  __syscall_dup3: function (fd, newfd) { try { return ksys('dup', [fd, newfd]); } catch (e) { return PKSYS.errS(e); } },
  __syscall_fcntl64__deps: ['$PKSYS'],
  __syscall_fcntl64__proxy: 'none',
  __syscall_fcntl64: function (fd, cmd, varargs) {
    try {
      if (cmd === 0 || cmd === 1030) return ksys('dup', [fd]);
      var arg = varargs ? HEAP32[varargs >> 2] : 0;
      return ksys('fcntl', [fd, cmd, arg]);
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_getuid32__proxy: 'none',
  __syscall_getuid32: function () { return 0; },
  __syscall_geteuid32__proxy: 'none',
  __syscall_geteuid32: function () { return 0; },
  __syscall_getgid32__proxy: 'none',
  __syscall_getgid32: function () { return 0; },
  __syscall_getegid32__proxy: 'none',
  __syscall_getegid32: function () { return 0; },
  __syscall_pipe2__deps: ['$PKSYS'],
  __syscall_pipe2__proxy: 'none',
  __syscall_pipe2: function (fdptr) { try { var fds = ksys('pipe', []); HEAP32[fdptr >> 2] = fds[0]; HEAP32[(fdptr + 4) >> 2] = fds[1]; return 0; } catch (e) { return PKSYS.errS(e); } },

  // ── ioctl: termios + winsize (ENOTTY on non-tty so isatty is honest) ──
  __syscall_ioctl__deps: ['$PKSYS'],
  __syscall_ioctl__proxy: 'none',
  __syscall_ioctl: function (fd, op, varargs) {
    try {
      var argp = HEAPU32[varargs >> 2];
      if (op === 0x5401 || op === 0x5402 || op === 0x5403 || op === 0x5404 || op === 0x5413) {
        if (ksys('fstat', [fd]).type !== 'char') return -25; // ENOTTY
      }
      var tm = PKSYS.termios;
      if (op === 0x5401) {
        HEAPU32[argp >> 2] = tm.c_iflag; HEAPU32[(argp + 4) >> 2] = tm.c_oflag; HEAPU32[(argp + 8) >> 2] = tm.c_cflag; HEAPU32[(argp + 12) >> 2] = tm.c_lflag;
        for (var i = 0; i < 19; i++) HEAPU8[argp + 17 + i] = tm.c_cc[i] || 0;
        return 0;
      }
      if (op === 0x5402 || op === 0x5403 || op === 0x5404) {
        var cc = [];
        for (var j = 0; j < 19; j++) cc[j] = HEAPU8[argp + 17 + j];
        PKSYS.applyTermios({ c_iflag: HEAPU32[argp >> 2], c_oflag: HEAPU32[(argp + 4) >> 2], c_cflag: HEAPU32[(argp + 8) >> 2], c_lflag: HEAPU32[(argp + 12) >> 2], c_cc: cc });
        return 0;
      }
      if (op === 0x5413) {
        var sz = PKSYS.winsize();
        HEAP16[argp >> 1] = sz[0]; HEAP16[(argp + 2) >> 1] = sz[1]; HEAP16[(argp + 4) >> 1] = 0; HEAP16[(argp + 6) >> 1] = 0;
        return 0;
      }
      if (op === 0x5421) { ksys('ioctl', [fd, 'FIONBIO', HEAP32[argp >> 2] !== 0]); return 0; } // FIONBIO
      if (op === 0x541b) { HEAP32[argp >> 2] = ksys('ioctl', [fd, 'FIONREAD']) | 0; return 0; } // FIONREAD
      if (op === 0x8905) { HEAP32[argp >> 2] = ksys('ioctl', [fd, 'SIOCATMARK']) | 0; return 0; } // SIOCATMARK
      return -25; // ENOTTY
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_poll__deps: ['$PKSYS'],
  __syscall_poll__proxy: 'none',
  __syscall_poll: function (fds, nfds, timeout) { try { return PKSYS.poll(fds, nfds, timeout); } catch (e) { return PKSYS.errS(e); } },
  // emscripten's poll() routes a zero timeout through this probe import
  __syscall_poll_nonblocking__deps: ['$PKSYS'],
  __syscall_poll_nonblocking__proxy: 'none',
  __syscall_poll_nonblocking: function (fds, nfds) { try { return PKSYS.poll(fds, nfds, 0); } catch (e) { return PKSYS.errS(e); } },

  // ── vfork / pipe bridge: process-level fork+exec, wired in x86-runtime.js ──
  js_kernel_pipe__proxy: 'none',
  js_kernel_pipe: function (out) {
    if (typeof __pkx === 'undefined') return -1;
    try {
      var fds = __pkx.kpipe();
      HEAP32[out >> 2] = fds[0];
      HEAP32[(out >> 2) + 1] = fds[1];
      return 0;
    } catch (e) { if (e && e.__exit) throw e; return -1; }
  },
  js_vfork_exec__proxy: 'none',
  js_vfork_exec: function (prog, argv, envp, f0, f1, f2) {
    if (typeof __pkx === 'undefined') return -52; // ENOSYS (WASI)
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
    return __pkx.vforkExec(null, UTF8ToString(prog), args, env, f0, f1, f2);
  },
  js_vfork_dead__proxy: 'none',
  js_vfork_dead: function (code) {
    if (typeof __pkx === 'undefined') return 32767;
    return __pkx.deadChild(code);
  },
  js_vfork_wait__proxy: 'none',
  js_vfork_wait: function (pid, nohang, code_out) {
    if (typeof __pkx === 'undefined') return -12; // ECHILD (WASI)
    var r = __pkx.vwait(pid, !!nohang);
    if (r.err) return -r.err;
    HEAP32[code_out >> 2] = r.code | 0;
    return r.pid | 0;
  },

  // ── sockets: host fd == kernel fd, the kernel's network stack does the rest ──
  __syscall_socket__deps: ['$PKSYS'],
  __syscall_socket__proxy: 'none',
  __syscall_socket: function (domain, type, protocol) {
    try { return ksys('socket', [domain, type & 0xf, protocol, { nonblock: (type & 2048) !== 0 }]); } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_socketpair__deps: ['$PKSYS'],
  __syscall_socketpair__proxy: 'none',
  __syscall_socketpair: function (domain, type, protocol, fds) {
    try {
      var r = ksys('socketpair', [domain, type]);
      HEAP32[fds >> 2] = r[0];
      HEAP32[(fds + 4) >> 2] = r[1];
      return 0;
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_bind__deps: ['$PKSYS'],
  __syscall_bind__proxy: 'none',
  __syscall_bind: function (fd, addr, len) {
    try { ksys('bind', [fd, PKSYS.readSockaddr(addr, len)]); return 0; } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_connect__deps: ['$PKSYS'],
  __syscall_connect__proxy: 'none',
  __syscall_connect: function (fd, addr, len) {
    try { ksys('connect', [fd, PKSYS.readSockaddr(addr, len)]); return 0; } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_listen__deps: ['$PKSYS'],
  __syscall_listen__proxy: 'none',
  __syscall_listen: function (fd, backlog) {
    try { ksys('listen', [fd, backlog]); return 0; } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_accept4__deps: ['$PKSYS'],
  __syscall_accept4__proxy: 'none',
  __syscall_accept4: function (fd, addr, addrlen, flags) {
    try {
      var r = ksys('accept', [fd, { nonblock: (flags & 2048) !== 0 }]);
      if (addr && addrlen) PKSYS.writeSockaddr(addr, addrlen, r.peer);
      return r.fd;
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_getsockname__deps: ['$PKSYS'],
  __syscall_getsockname__proxy: 'none',
  __syscall_getsockname: function (fd, addr, addrlen) {
    try { PKSYS.writeSockaddr(addr, addrlen, ksys('getsockname', [fd])); return 0; } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_getpeername__deps: ['$PKSYS'],
  __syscall_getpeername__proxy: 'none',
  __syscall_getpeername: function (fd, addr, addrlen) {
    try { PKSYS.writeSockaddr(addr, addrlen, ksys('getpeername', [fd])); return 0; } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_shutdown__deps: ['$PKSYS'],
  __syscall_shutdown__proxy: 'none',
  __syscall_shutdown: function (fd, how) {
    try { ksys('shutdown', [fd, how]); return 0; } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_sendto__deps: ['$PKSYS'],
  __syscall_sendto__proxy: 'none',
  __syscall_sendto: function (fd, buf, len, flags, addr, alen) {
    try {
      var to = addr && alen ? PKSYS.readSockaddr(addr, alen) : undefined;
      return ksys('sendto', [fd, HEAPU8.slice(buf, buf + len), { to: to, flags: flags }]);
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_recvfrom__deps: ['$PKSYS'],
  __syscall_recvfrom__proxy: 'none',
  __syscall_recvfrom: function (fd, buf, len, flags, addr, alen) {
    try {
      var r = ksys('recvfrom', [fd, len, { flags: flags }]);
      if (r === null) return 0;
      HEAPU8.set(r.data.subarray(0, len), buf);
      if (addr && alen) PKSYS.writeSockaddr(addr, alen, r.from);
      return (flags & 32) && r.trunc ? r.trunc : r.data.length; // MSG_TRUNC reports the full length
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_sendmsg__deps: ['$PKSYS'],
  __syscall_sendmsg__proxy: 'none',
  __syscall_sendmsg: function (fd, msg, flags) {
    try {
      var name = HEAPU32[msg >> 2];
      var namelen = HEAPU32[(msg + 4) >> 2];
      var data = PKSYS.gather(HEAPU32[(msg + 8) >> 2], HEAP32[(msg + 12) >> 2]);
      var to = name && namelen ? PKSYS.readSockaddr(name, namelen) : undefined;
      return ksys('sendto', [fd, data, { to: to, flags: flags }]);
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_recvmsg__deps: ['$PKSYS'],
  __syscall_recvmsg__proxy: 'none',
  __syscall_recvmsg: function (fd, msg, flags) {
    try {
      var name = HEAPU32[msg >> 2];
      var iov = HEAPU32[(msg + 8) >> 2];
      var iovcnt = HEAP32[(msg + 12) >> 2];
      var total = PKSYS.iovTotal(iov, iovcnt);
      var r = ksys('recvfrom', [fd, total, { flags: flags }]);
      if (r === null) { HEAPU32[(msg + 20) >> 2] = 0; HEAP32[(msg + 24) >> 2] = 0; return 0; }
      var n = PKSYS.scatter(iov, iovcnt, r.data);
      if (name) PKSYS.writeSockaddr(name, msg + 4, r.from); // bounded by msg_namelen, which gets the full size
      HEAPU32[(msg + 20) >> 2] = 0; // no control data
      HEAP32[(msg + 24) >> 2] = r.trunc ? 32 : 0; // MSG_TRUNC
      return (flags & 32) && r.trunc ? r.trunc : n;
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_getsockopt__deps: ['$PKSYS'],
  __syscall_getsockopt__proxy: 'none',
  __syscall_getsockopt: function (fd, level, optname, optval, optlen) {
    try {
      var v = ksys('getsockopt', [fd, level, optname]);
      PKSYS.writeOptVal(v, optval, optlen);
      return 0;
    } catch (e) { return PKSYS.errS(e); }
  },
  __syscall_setsockopt__deps: ['$PKSYS'],
  __syscall_setsockopt__proxy: 'none',
  __syscall_setsockopt: function (fd, level, optname, optval, optlen) {
    try {
      ksys('setsockopt', [fd, level, optname, PKSYS.readOptVal(level, optname, optval, optlen)]);
      return 0;
    } catch (e) { return PKSYS.errS(e); }
  },
  // alarm(2)/setitimer(ITIMER_REAL) from the blink patch: the kernel keeps the
  // timer and raises SIGALRM in the guest (see raiseGuestSignal in kernel.ts)
  js_set_alarm__deps: ['$PKSYS'],
  js_set_alarm__proxy: 'none',
  js_set_alarm: function (ms, intervalMs) {
    try { return ksys('alarm', [ms, intervalMs]) | 0; } catch (e) { if (e && e.__exit) throw e; return 0; }
  },
  // interface ioctls from the blink patch (ioctl.c IoctlNetPassthrough): the
  // guest's Linux amd64 struct bytes go to the kernel's stack and come back
  js_net_ioctl__deps: ['$PKSYS'],
  js_net_ioctl__proxy: 'none',
  js_net_ioctl: function (fd, request, buf, len) {
    try {
      var r = ksys('ioctl', [fd, PKSYS.siocName(request >>> 0), HEAPU8.slice(buf, buf + len)]);
      if (r && r.length) HEAPU8.set(r.subarray(0, len), buf);
      return 0;
    } catch (e) { return PKSYS.errS(e); }
  },
});
