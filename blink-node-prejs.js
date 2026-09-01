// DEBUG-ONLY pre-js: a node-side `ksys` so the blink wasm build runs standalone
// under Node (scripts/blink-node-harness.cjs) with no kernel worker. Runs on
// every thread at module load; installs ONLY in Node when no real ksys exists,
// so the browser build (x86-runtime.js sets globalThis.ksys) is untouched.
//
// It models just enough kernel: a real-file-backed FS rooted at a host dir (the
// guest ELF + any data files live there), stdout/stderr, exit, sleep. The FS
// root + guest argv come from a JSON sidecar the harness writes next to
// blinkw.js (env vars do NOT propagate to emscripten pthread workers).
if (typeof globalThis.ksys !== 'function' &&
    typeof process !== 'undefined' && process.versions && process.versions.node) {
  var __pkFs; try { __pkFs = require('fs'); } catch (e) {}
  var __pkDbg = false;
  var __pkCfg = {};
  try {
    var __pkDir = (typeof __dirname !== 'undefined') ? __dirname : '.';
    __pkCfg = JSON.parse(__pkFs.readFileSync(__pkDir + '/blink-node-fs.json', 'utf8'));
    __pkDbg = !!__pkCfg.debug;
  } catch (e) {}
  var __pkRoot = __pkCfg.root || '.';
  var __pkFds = Object.create(null);   // fd -> {hfd, pos, path, discard, std}
  __pkFds[0] = { std: 0 }; __pkFds[1] = { std: 1 }; __pkFds[2] = { std: 2 };
  var __pkNext = 200;
  // Guest stdout/stderr go to files the harness tails (worker out()/std* are lost).
  var __pkOutFd = null, __pkErrFd = null;
  function __pkStdFd(which) {
    try {
      if (which === 1) { if (__pkOutFd == null && __pkCfg.stdout) __pkOutFd = __pkFs.openSync(__pkCfg.stdout, 'a'); return __pkOutFd; }
      if (__pkErrFd == null && __pkCfg.stderr) __pkErrFd = __pkFs.openSync(__pkCfg.stderr, 'a'); return __pkErrFd;
    } catch (e) { return null; }
  }

  function __pkLog(s) { if (__pkDbg) { try { process.stderr.write('[ksys] ' + s + '\n'); } catch (e) {} } }
  function __pkHostPath(p) {
    if (__pkCfg.files && __pkCfg.files[p]) return __pkCfg.files[p];
    return __pkRoot + (p[0] === '/' ? p : '/' + p);
  }
  function __pkStatObj(s) {
    return { type: s.isDirectory() ? 'dir' : s.isCharacterDevice() ? 'char' : s.isFIFO() ? 'fifo' : s.isSymbolicLink() ? 'link' : 'file',
             mode: s.mode & 0o7777, size: s.size, mtime: s.mtimeMs };
  }
  function __pkEnoent() { var e = new Error('ENOENT'); e.__errno = 2; throw e; }

  try { process.stderr.write('[prejs] out=' + (typeof out) + ' err=' + (typeof err) + ' Module=' + (typeof Module) + '\n'); } catch (e) {}
  globalThis.ksys = function (name, args) {
    if (name === 'write') __pkLog('write fd=' + args[0] + ' len=' + (args[1] && args[1].length));
    else if (name !== 'read') __pkLog(name + ' ' + JSON.stringify(args).slice(0, 80));
    switch (name) {
      case 'write': {
        var wfd = args[0] | 0, data = args[1];
        var buf = Buffer.from(data.buffer ? new Uint8Array(data) : data);
        var f = __pkFds[wfd];
        var stdw = (f && f.std != null) ? f.std : (wfd === 1 || wfd === 2 ? wfd : null);
        if (stdw === 1 || stdw === 2) {
          var sfd = __pkStdFd(stdw);
          if (sfd != null) { try { __pkFs.writeSync(sfd, buf); } catch (e) {} }
          else { try { process[stdw === 2 ? 'stderr' : 'stdout'].write(buf); } catch (e) {} }
          return buf.length;
        }
        if (f && f.hfd != null && !f.discard) { try { __pkFs.writeSync(f.hfd, buf, 0, buf.length, f.pos); f.pos += buf.length; } catch (e) {} }
        return buf.length;
      }
      case 'read': {
        var rfd = args[0] | 0, len = args[1] | 0, rf = __pkFds[rfd];
        if (!rf || rf.hfd == null) return new Uint8Array(0);
        var b = Buffer.alloc(len);
        var n = 0; try { n = __pkFs.readSync(rf.hfd, b, 0, len, rf.pos); } catch (e) { n = 0; }
        rf.pos += n;
        return n === 0 ? null : new Uint8Array(b.buffer, b.byteOffset, n);
      }
      case 'openat': case 'open': {
        var p = args[0], flags = (args[1] || 0) >>> 0;
        if (p && String(p).indexOf('blink.log') >= 0) { var lf = __pkNext++; __pkFds[lf] = { std: 2 }; return lf; }  // route blink diagnostics to stderr
        var host = __pkHostPath(p);
        var wr = (flags & 3) !== 0 || (flags & 0o100) !== 0;  // O_WRONLY/O_RDWR/O_CREAT
        var fd = __pkNext++;
        try {
          if (wr) {
            // writable file (e.g. blink.log): create under root, else discard
            var hfd = __pkFs.openSync(host, (flags & 0o1000) ? 'a' : 'w');
            __pkFds[fd] = { hfd: hfd, pos: 0, path: p, discard: false };
          } else {
            var rfd2 = __pkFs.openSync(host, 'r');
            __pkFds[fd] = { hfd: rfd2, pos: 0, path: p, discard: false };
          }
        } catch (e) {
          if (wr) { __pkFds[fd] = { hfd: null, pos: 0, path: p, discard: true }; }
          else { __pkEnoent(); }
        }
        return fd;
      }
      case 'close': { var cf = __pkFds[args[0]]; if (cf && cf.hfd != null && !cf.dup && cf.std == null) { try { __pkFs.closeSync(cf.hfd); } catch (e) {} } delete __pkFds[args[0]]; return 0; }
      case 'seek': {
        var sf = __pkFds[args[0]]; if (!sf) __pkEnoent();
        var off = args[1] | 0, whence = args[2] | 0;
        var size = 0; try { size = __pkFs.fstatSync(sf.hfd).size; } catch (e) {}
        sf.pos = whence === 1 ? sf.pos + off : whence === 2 ? size + off : off;
        return sf.pos;
      }
      case 'stat': { try { return __pkStatObj(__pkFs.statSync(__pkHostPath(args[0]))); } catch (e) { __pkEnoent(); } break; }
      case 'fstat': { var ff = __pkFds[args[0]]; if (!ff || ff.hfd == null) { return { type: 'char', mode: 0o666, size: 0, mtime: 0 }; } try { return __pkStatObj(__pkFs.fstatSync(ff.hfd)); } catch (e) { __pkEnoent(); } break; }
      case 'fdpath': { var pf = __pkFds[args[0]]; return pf ? (pf.path || '/') : '/'; }
      case 'dup': {
        var sfd2 = __pkFds[args[0] | 0];
        var nf = __pkNext++;
        __pkFds[nf] = sfd2 ? { std: sfd2.std, hfd: sfd2.hfd, pos: sfd2.pos || 0, path: sfd2.path, discard: sfd2.discard, dup: true } : { std: 1 };
        return nf;
      }
      case 'fcntl': {
        var cmd = args[1] | 0, ef = __pkFds[args[0] | 0];
        if (cmd === 3) return ef ? 2 : (function () { __pkEnoent(); })();  // F_GETFL -> O_RDWR
        if (cmd === 1) return ef ? 1 : 0;   // F_GETFD -> FD_CLOEXEC-ish
        return 0;                            // F_SETFD/F_SETFL etc.
      }
      case 'ioctl': return (args && args[1] === 'size') ? { rows: 24, cols: 80 } : null;
      case 'readdirfd': return [];
      case 'klog': {
        var kl = '[klog] ' + args[0] + '\n';
        var kfd = __pkStdFd(2);  // synchronous stderr file survives a JIT crash
        if (kfd != null) { try { __pkFs.writeSync(kfd, kl); __pkFs.fsyncSync(kfd); } catch (e) {} }
        else { try { process.stderr.write(kl); } catch (e) {} }
        return null;
      }
      case 'procexit': {
        var xc = args[0] | 0;
        try { if (__pkOutFd != null) __pkFs.fsyncSync(__pkOutFd); } catch (e) {}
        try { if (__pkErrFd != null) __pkFs.fsyncSync(__pkErrFd); } catch (e) {}
        try { if (__pkCfg.done) __pkFs.writeFileSync(__pkCfg.done, String(xc)); } catch (e) {}
        try { process.exit(xc); } catch (e) {}
        return null;
      }
      case 'sleep': if (typeof SharedArrayBuffer !== 'undefined') { try { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, args[0] | 0); } catch (e) {} } return null;
      default: return null;
    }
  };
}
