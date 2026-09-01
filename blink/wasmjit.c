// pk910: runtime x86 -> WebAssembly JIT for the wasm build of blink.
//
// blink's native JIT (jit.c) only targets x86_64/aarch64 hosts and is compiled
// out on the wasm host (HAVE_JIT undefined). This is the wasm equivalent: hot
// guest basic blocks are recompiled into WebAssembly functions the browser
// engine JITs to native. Purely additive - anything unsupported bails to the
// interpreter, so correctness holds regardless of coverage.
//
// TWO-TIER CACHE (see ai_plans/wasm-jit.md). emscripten does NOT share a
// pthread's grown wasm table across the process's other pthreads (a dynamically
// installed idx is out of bounds elsewhere), but linear MEMORY is shared and
// STATIC function indices are identical on every thread. So: a SHARED tier
// (process-global, shared memory, lock-free CAS) holds the emitted module BYTES
// (compiled once per ip), and a PER-THREAD tier maps ip -> this thread's table
// index (each thread instantiates from the shared bytes once).
//
// EMITTER. Walk the basic block; end at the first branch/precious op (ClassifyOp):
//   * INLINE register-direct 32/64-bit ALU (OpAluw Ev,Gv and OpAlui group1-imm,
//     identified by extern symbol): operands come from a block-local register
//     cache (16 wasm i64 locals), and the arithmetic + FLAGS are delegated to
//     blink's own kAlu[op][width] leaf (baked call_indirect) - exact flags for
//     free, and the cache survives across ALU ops (kAlu never touches guest regs).
//   * FALLBACK for everything else: spill the cache, store m->ip, baked
//     call_indirect to the instruction's real Op handler, invalidate the cache.
// Correct by construction; the win is skipped dispatch/operand-decode + the
// register cache keeping hot operands out of memory across runs of ALU ops.

#include "blink/builtin.h"

#ifdef HAVE_WASM_JIT
#include <stddef.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "blink/alu.h"
#include "blink/machine.h"
#include "blink/modrm.h"
#include "blink/flags.h"
#include "blink/rde.h"

extern int pk_jit_install(const void *bytes, int len, int first_compile);

// static in machine.c, un-static'd (pk910) so we can identify them by symbol
// and avoid 2-byte-opcode collisions from a pure-opcode decode.
void OpAluFlip(P);
void OpAluCmp(P);
void OpAluFlipCmp(P);
void OpAluTest(P);
void OpMovEvqpGvqp(P);  // mov Ev,Gv (#1 hottest; reg-reg when mod==3)
void OpMovGvqpEvqp(P);  // mov Gv,Ev
void OpMovZvqpIvqp(P);  // mov Zv,imm
void OpJcc(P);          // conditional jump (self-loop terminator detection)
void OpLeaGvqpM(P);     // lea Gv,M (gcc emits it as arithmetic; no flags)
void OpBsuwiImm(P);    // shift/rotate rm, imm (kBsu leaf)
void OpBsuwiCl(P);     // shift/rotate rm, cl
void OpMovzbGvqpEb(P);  // movzx Gv, byte rm  (memory-read inlining)
void OpMovzwGvqpEw(P);  // movzx Gv, word rm
void OpMovsbGvqpEb(P);  // movsx Gv, byte rm
void OpMovswGvqpEw(P);  // movsx Gv, word rm
void OpMovslGdqpEd(P);  // movsxd Gv, dword rm

// ── config ──────────────────────────────────────────────────────────────────
#define PKJIT_SHARED    (1u << 15)
#define PKJIT_LOCAL     (1u << 13)
#define PKJIT_THRESHOLD 50
#define PKJIT_MAXINSN   200
#define PKJIT_SCRATCH   131072

enum { kWarming = 0, kEmitting = 1, kReady = 2 };
struct SharedEntry {
  _Atomic(u64) virt;
  _Atomic(u32) state;
  _Atomic(u32) hits;
  const u8 *bytes;
  u32 len;
  u32 gen;  // g_codegen when emitted; stale if code changed (SMC)
};
static struct SharedEntry g_shared[PKJIT_SHARED];

struct LocalHook {
  u64 virt;
  u32 idx;
  u32 gen;
};
static _Thread_local struct LocalHook *t_local;
static _Thread_local u8 *t_scratch;
static _Thread_local int t_wasloop;  // diag: last emit was a self-loop
static int g_pkjit_ready;
static _Atomic(u32) g_emits;
// Bumped whenever the guest writes an executable page (SMC); stale-gen blocks are
// never executed and get re-emitted. Called from smc.c AddPageToSmcQueue.
static _Atomic(u32) g_codegen;
void WasmJitFlushCode(void) {
  atomic_fetch_add_explicit(&g_codegen, 1, memory_order_release);
}

// Fixed module prefix: 3 types (t0 = block/handler (i32,i64,i64,i64)->() ;
// t1 = kAlu (i32,i64,i64)->i64 ; t2 = CommitStash (i32)->()), imports env.mem
// (shared {1,65536}) + env.tbl (funcref), func(t0), export "b". Only the code
// section (appended) varies.
static const u8 kPrefix[] = {
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x13, 0x03, 0x60, 0x04, 0x7f, 0x7e, 0x7e, 0x7e, 0x00,  // t0
    0x60, 0x03, 0x7f, 0x7e, 0x7e, 0x01, 0x7e,                    // t1
    0x60, 0x01, 0x7f, 0x00,                                      // t2
    0x02, 0x1b, 0x02, 0x03, 0x65, 0x6e, 0x76, 0x03, 0x6d, 0x65,
    0x6d, 0x02, 0x03, 0x01, 0x80, 0x80, 0x04, 0x03, 0x65, 0x6e,
    0x76, 0x03, 0x74, 0x62, 0x6c, 0x01, 0x70, 0x00, 0x00,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x05, 0x01, 0x01, 0x62, 0x00, 0x00,
};

#define OFF_IP  ((u32)offsetof(struct Machine, ip))
#define OFF_WEG ((u32)offsetof(struct Machine, weg))
#define LOC_GPR 4  // wasm local index of GPR 0 (locals 4..19)

// ── wasm byte emitter ───────────────────────────────────────────────────────
struct Buf {
  u8 *p;
  u32 n, cap;
  int ovf;
};
static void bput(struct Buf *b, u8 x) {
  if (b->n < b->cap) b->p[b->n] = x;
  else b->ovf = 1;
  b->n++;
}
static void bputs(struct Buf *b, const void *s, u32 k) {
  const u8 *q = (const u8 *)s;
  for (u32 i = 0; i < k; ++i) bput(b, q[i]);
}
static void bleb_u(struct Buf *b, u64 v) {
  do {
    u8 x = (u8)(v & 0x7f);
    v >>= 7;
    if (v) x |= 0x80;
    bput(b, x);
  } while (v);
}
static void bleb_s(struct Buf *b, i64 v) {
  for (;;) {
    u8 x = (u8)(v & 0x7f);
    v >>= 7;
    if ((v == 0 && !(x & 0x40)) || (v == -1 && (x & 0x40))) {
      bput(b, x);
      return;
    }
    bput(b, x | 0x80);
  }
}
static u32 leb_u_size(u64 v) {
  u32 n = 0;
  do {
    ++n;
    v >>= 7;
  } while (v);
  return n;
}
static void EGet(struct Buf *b, u32 i) { bput(b, 0x20); bleb_u(b, i); }   // local.get
static void ESet(struct Buf *b, u32 i) { bput(b, 0x21); bleb_u(b, i); }   // local.set
static void ELoad(struct Buf *b, u32 off) {  // i64.load [m + off]
  bput(b, 0x29); bleb_u(b, 3); bleb_u(b, off);
}
static void EStore(struct Buf *b, u32 off) {  // i64.store [m + off]
  bput(b, 0x37); bleb_u(b, 3); bleb_u(b, off);
}
static void EConst(struct Buf *b, i64 v) { bput(b, 0x42); bleb_s(b, v); }  // i64.const
static void EConstI(struct Buf *b, i32 v) { bput(b, 0x41); bleb_s(b, v); } // i32.const
static void EBin(struct Buf *b, u8 op) { bput(b, op); }  // i64/i32 binop

#define OFF_FLAGS ((u32)offsetof(struct Machine, flags))
#define OFF_OPLEN ((u32)offsetof(struct Machine, oplen))
#define OFF_STASH ((u32)offsetof(struct Machine, stashaddr))
#define OFF_ATT ((u32)offsetof(struct Machine, attention))
// scratch locals (declared after the 16 GPR locals): i64 20..23, i32 24..26
#define LT0 20
#define LT1 21
#define LT2 22
#define LT3 23
#define FL  24   // cached m->flags (i32)
#define HP0 25   // i32 scratch: tlb slot / host pointer (memory fast path)
#define HP1 26   // i32 scratch: offset within page
#define OFF_TLB   ((u32)offsetof(struct Machine, tlb))
#define OFF_INVAL ((u32)offsetof(struct Machine, invalidated))

// wasm opcodes
#define I64_ADD 0x7c
#define I64_SUB 0x7d
#define I64_AND 0x83
#define I64_OR  0x84
#define I64_XOR 0x85
#define I64_SHRU 0x88
#define I64_SHL 0x86
#define I64_MUL 0x7e
#define I64_LTU 0x54
#define I64_EQZ 0x50
#define I64_NE  0x52
#define I64_WRAP 0xa7
#define I64_SHRS 0x87
#define I64_NE  0x52
#define I32_OR  0x72
#define I32_AND 0x71
#define I32_SHL 0x74
#define I32_MUL 0x6c
#define I32_ADD 0x6a
#define I32_GTU 0x4b

// ── register cache ──────────────────────────────────────────────────────────
struct Rc {
  u8 loaded[16];
  u8 dirty[16];
  u8 fl_loaded;  // m->flags cached in local FL
  u8 fl_dirty;
};
static void FlagsEnsure(struct Buf *b, struct Rc *rc) {
  if (rc->fl_loaded) return;
  EGet(b, 0);
  bput(b, 0x28); bleb_u(b, 2); bleb_u(b, OFF_FLAGS);  // i32.load m->flags
  ESet(b, FL);
  rc->fl_loaded = 1;
}
static void FlagsSpill(struct Buf *b, struct Rc *rc) {  // FL local -> m->flags
  if (!rc->fl_dirty) return;
  EGet(b, 0);
  EGet(b, FL);
  bput(b, 0x36); bleb_u(b, 2); bleb_u(b, OFF_FLAGS);  // i32.store m->flags
  rc->fl_dirty = 0;
}
static void RcLoad(struct Buf *b, struct Rc *rc, int r) {  // ensure GPR r in local
  if (rc->loaded[r]) return;
  EGet(b, 0);                        // m
  ELoad(b, OFF_WEG + (u32)r * 8);    // i64.load weg[r]
  ESet(b, LOC_GPR + r);
  rc->loaded[r] = 1;
}
static void RcSpill(struct Buf *b, struct Rc *rc) {  // dirty locals -> memory
  for (int r = 0; r < 16; ++r) {
    if (!rc->dirty[r]) continue;
    EGet(b, 0);                        // m
    EGet(b, LOC_GPR + r);              // value
    EStore(b, OFF_WEG + (u32)r * 8);
    rc->dirty[r] = 0;
  }
  if (rc->fl_dirty) {
    EGet(b, 0);
    EGet(b, FL);
    bput(b, 0x36); bleb_u(b, 2); bleb_u(b, OFF_FLAGS);  // i32.store m->flags
    rc->fl_dirty = 0;
  }
}
static void RcInval(struct Rc *rc) {
  memset(rc->loaded, 0, sizeof(rc->loaded));
  memset(rc->dirty, 0, sizeof(rc->dirty));
  rc->fl_loaded = 0;  // flags cache invalidated too (was the hello-go bug)
  rc->fl_dirty = 0;
}

// ── block compiler ──────────────────────────────────────────────────────────
static void EmitHandler(struct Buf *b, u64 rde, i64 disp, u64 uimm0, u32 hidx) {
  EGet(b, 0);
  EConst(b, (i64)rde);
  EConst(b, disp);
  EConst(b, (i64)uimm0);
  bput(b, 0x41); bleb_s(b, (i64)(i32)hidx);  // i32.const handler idx
  bput(b, 0x11); bleb_u(b, 0); bput(b, 0x00);  // call_indirect t0 table0
}

// After a handler, drain a page-crossing store's stash: if (m->stashaddr)
// CommitStash(m). The interpreter does this per instruction (machine.c:2158);
// a chained block would otherwise lose a page-crossing write.
static void EmitCommitStash(struct Buf *b) {
  EGet(b, 0);
  bput(b, 0x29); bleb_u(b, 3); bleb_u(b, OFF_STASH);  // i64.load m->stashaddr
  EConst(b, 0);
  bput(b, 0x52);              // i64.ne  -> i32 cond
  bput(b, 0x04); bput(b, 0x40);  // if (void)
  EGet(b, 0);
  bput(b, 0x41); bleb_s(b, (i64)(i32)(uintptr_t)&CommitStash);  // i32.const
  bput(b, 0x11); bleb_u(b, 2); bput(b, 0x00);  // call_indirect t2 (i32)->()
  bput(b, 0x0b);             // end
}

// push (FL >> n) & 1 as i32 (flag bit n)
static void EmitBit(struct Buf *b, int n) {
  EGet(b, FL);
  if (n) { EConstI(b, n); bput(b, 0x76); }  // i32.shr_u
  EConstI(b, 1); bput(b, 0x71);             // i32.and
}
#define I32_XOR 0x73
// push i32 (1 if jcc condition `cc` is taken), from FL. cc 10/11 (P/NP, lazy
// parity) are excluded by the caller. CF=bit0 ZF=6 SF=7 OF=11.
static void EmitCond(struct Buf *b, int cc) {
  switch (cc) {
    case 0: EmitBit(b, 11); break;                                  // JO
    case 1: EmitBit(b, 11); EConstI(b, 1); bput(b, I32_XOR); break; // JNO
    case 2: EmitBit(b, 0); break;                                   // JB
    case 3: EmitBit(b, 0); EConstI(b, 1); bput(b, I32_XOR); break;  // JAE
    case 4: EmitBit(b, 6); break;                                   // JE
    case 5: EmitBit(b, 6); EConstI(b, 1); bput(b, I32_XOR); break;  // JNE
    case 6: EmitBit(b, 0); EmitBit(b, 6); bput(b, I32_OR); break;   // JBE CF|ZF
    case 7: EmitBit(b, 0); EmitBit(b, 6); bput(b, I32_OR);
            EConstI(b, 1); bput(b, I32_XOR); break;                 // JA !(CF|ZF)
    case 8: EmitBit(b, 7); break;                                   // JS
    case 9: EmitBit(b, 7); EConstI(b, 1); bput(b, I32_XOR); break;  // JNS
    case 12: EmitBit(b, 7); EmitBit(b, 11); bput(b, I32_XOR); break;  // JL SF^OF
    case 13: EmitBit(b, 7); EmitBit(b, 11); bput(b, I32_XOR);
             EConstI(b, 1); bput(b, I32_XOR); break;                  // JGE
    case 14: EmitBit(b, 6); EmitBit(b, 7); EmitBit(b, 11);
             bput(b, I32_XOR); bput(b, I32_OR); break;                // JLE ZF|(SF^OF)
    default: EmitBit(b, 6); EmitBit(b, 7); EmitBit(b, 11);           // JG (15)
             bput(b, I32_XOR); bput(b, I32_OR);
             EConstI(b, 1); bput(b, I32_XOR); break;
  }
}

// dst = kAlu[t][log2](m, dst, y); y is GPR src or (imm) immediate.
static void EmitAlu(struct Buf *b, struct Rc *rc, int t, int log2, int dst,
                    int src, bool imm, u64 immv, bool wb) {
  RcLoad(b, rc, dst);
  if (!imm && src >= 0) RcLoad(b, rc, src);
  FlagsSpill(b, rc);          // kAlu[adc/sbb] reads CF from m->flags
  EGet(b, 0);                 // m
  EGet(b, LOC_GPR + dst);     // x
  if (imm) EConst(b, (i64)immv);
  else EGet(b, src >= 0 ? LOC_GPR + src : LT3);  // y (src<0: mem value in LT3)
  u32 fidx = (u32)(uintptr_t)kAlu[t][log2];
  bput(b, 0x41); bleb_s(b, (i64)(i32)fidx);   // i32.const kAlu fn idx
  bput(b, 0x11); bleb_u(b, 1); bput(b, 0x00);  // call_indirect t1 table0 -> i64
  if (wb) {
    ESet(b, LOC_GPR + dst);
    rc->loaded[dst] = 1;
    rc->dirty[dst] = 1;
  } else {
    bput(b, 0x1a);  // drop (cmp/test: flags only)
  }
  rc->fl_loaded = 0;  // kAlu wrote m->flags; our FL cache is stale
}

// rm = kBsu[op][log2](m, rm, count); shift/rotate via blink's own leaf (exact
// flags incl. the x86 count-masking + undefined-flag quirks). count is imm or CL.
static void EmitBsu(struct Buf *b, struct Rc *rc, int op, int log2, int dst,
                    bool imm, u64 immv) {
  RcLoad(b, rc, dst);
  if (!imm) RcLoad(b, rc, 1);   // CL = rcx low byte
  FlagsSpill(b, rc);            // RCL/RCR read CF from m->flags
  EGet(b, 0);                   // m
  EGet(b, LOC_GPR + dst);       // value
  if (imm) {
    EConst(b, (i64)immv);
  } else {
    EGet(b, LOC_GPR + 1); EConst(b, 0xff); EBin(b, I64_AND);  // cl = rcx & 0xff
  }
  u32 fidx = (u32)(uintptr_t)kBsu[op][log2];
  bput(b, 0x41); bleb_s(b, (i64)(i32)fidx);   // i32.const kBsu fn idx
  bput(b, 0x11); bleb_u(b, 1); bput(b, 0x00);  // call_indirect t1 table0 -> i64
  ESet(b, LOC_GPR + dst);       // 32-bit leaf returns zero-extended, so full store ok
  rc->loaded[dst] = 1;
  rc->dirty[dst] = 1;
  rc->fl_loaded = 0;            // kBsu wrote m->flags; FL cache stale
}

#define I64_SHRS 0x87
#define I64_ROTL 0x89
#define I64_ROTR 0x8a

// Inline shift/rotate as a pure wasm op - VALUE semantics only, so the caller
// must confirm ALL flags are dead downstream (x86 shifts touch CF/OF/ZF/SF/PF
// and leave them unchanged for count==0; with dead flags only the value
// matters). RCL/RCR are excluded (value reads CF); 32-bit rotates excluded
// (no cheap 32-in-64 rotate). Count masks &63/&31 exactly like x86.
static void EmitBsuInline(struct Buf *b, struct Rc *rc, int op, int log2,
                          int dst, bool imm, u64 immv) {
  bool w32 = (log2 == 2);
  i64 cmask = w32 ? 31 : 63;
  RcLoad(b, rc, dst);
  if (!imm) RcLoad(b, rc, 1);  // CL = RCX low bits
  // value operand
  EGet(b, LOC_GPR + dst);
  if (w32) {
    if (op == 7) {  // SAR: sign-extend low 32 first
      EConst(b, 32); EBin(b, I64_SHL);
      EConst(b, 32); EBin(b, I64_SHRS);
    } else {
      EConst(b, 0xffffffff); EBin(b, I64_AND);
    }
  }
  // count operand (masked)
  if (imm) {
    EConst(b, (i64)(immv & (u64)cmask));
  } else {
    EGet(b, LOC_GPR + 1); EConst(b, cmask); EBin(b, I64_AND);
  }
  switch (op) {
    case 0: EBin(b, I64_ROTL); break;              // rol (64-bit only)
    case 1: EBin(b, I64_ROTR); break;              // ror (64-bit only)
    case 5: EBin(b, I64_SHRU); break;              // shr
    case 7: EBin(b, I64_SHRS); break;              // sar
    default: EBin(b, I64_SHL); break;              // shl/sal (4/6)
  }
  if (w32) { EConst(b, 0xffffffff); EBin(b, I64_AND); }  // 32-bit zero-extend
  ESet(b, LOC_GPR + dst);
  rc->loaded[dst] = 1;
  rc->dirty[dst] = 1;
}

// Decide if this insn is an inlinable register-direct 32/64-bit shift/rotate.
static bool BsuDecode(nexgen32e_f h, u64 rde, u64 uimm0, int *op, int *log2,
                      int *dst, bool *imm, u64 *immv) {
  if (h != OpBsuwiImm && h != OpBsuwiCl) return false;
  if (!IsModrmRegister(rde) || Lock(rde)) return false;
  int lg = RegLog2(rde);
  if (lg != 2 && lg != 3) return false;  // only 32/64-bit
  *log2 = lg;
  *op = (int)ModrmReg(rde);   // BSU_ROL..BSU_SAR
  *dst = (int)RexbRm(rde);
  *imm = (h == OpBsuwiImm);
  *immv = uimm0;
  return true;
}

// dst = areg * (breg | imm), low bits only; 32-bit masks operands + result
// (imul r32 zero-extends into r64). When `need`, also produce imul's EXACT
// blink flags (divmul.c AluImul): CF = OF = signed-overflow of the product,
// every other flag (incl. lazy parity byte) untouched. 32-bit overflow test is
// z != (i64)(i32)z on the exact 64-bit product of sign-extended operands; the
// 64-bit test needs the high half, computed from four 32-bit partial products
// (mulhu, then the Hacker's Delight signed adjustment), then hs != lo>>s63.
static void EmitImul(struct Buf *b, struct Rc *rc, int dst, int areg, int breg,
                     bool bimm, u64 immv, int log2, int need) {
  bool w32 = (log2 == 2);
  if (!need) {  // CF/OF dead: lean path, no flag work
    RcLoad(b, rc, areg);
    EGet(b, LOC_GPR + areg);
    if (w32) { EConst(b, 0xffffffff); EBin(b, I64_AND); }
    if (bimm) {
      EConst(b, w32 ? (i64)(u64)(u32)immv : (i64)immv);
    } else {
      RcLoad(b, rc, breg);
      EGet(b, LOC_GPR + breg);
      if (w32) { EConst(b, 0xffffffff); EBin(b, I64_AND); }
    }
    EBin(b, I64_MUL);
    if (w32) { EConst(b, 0xffffffff); EBin(b, I64_AND); }
    ESet(b, LOC_GPR + dst);
    rc->loaded[dst] = 1;
    rc->dirty[dst] = 1;
    return;
  }
  RcLoad(b, rc, areg);
  if (!bimm) RcLoad(b, rc, breg);
  FlagsEnsure(b, rc);
  // LT0 = a, LT1 = b|imm (sign-extended for 32-bit so the i64 product is exact)
  EGet(b, LOC_GPR + areg);
  if (w32) { EConst(b, 32); EBin(b, I64_SHL); EConst(b, 32); EBin(b, I64_SHRS); }
  ESet(b, LT0);
  if (bimm) {
    EConst(b, w32 ? (i64)(i32)immv : (i64)immv);  // blink Load32+(i32) sign-ext
  } else {
    EGet(b, LOC_GPR + breg);
    if (w32) { EConst(b, 32); EBin(b, I64_SHL); EConst(b, 32); EBin(b, I64_SHRS); }
  }
  ESet(b, LT1);
  // LT2 = low product ; dst = LT2 (32-bit: zero-extend low half)
  EGet(b, LT0); EGet(b, LT1); EBin(b, I64_MUL); ESet(b, LT2);
  EGet(b, LT2);
  if (w32) { EConst(b, 0xffffffff); EBin(b, I64_AND); }
  ESet(b, LOC_GPR + dst);
  rc->loaded[dst] = 1;
  rc->dirty[dst] = 1;
  // FL = (FL & ~(CF|OF)) | of * 0x801
  EGet(b, FL); EConstI(b, ~0x801); EBin(b, I32_AND);
  if (w32) {
    // of = LT2 != (i64)(i32)LT2
    EGet(b, LT2);
    EGet(b, LT2); EConst(b, 32); EBin(b, I64_SHL); EConst(b, 32); EBin(b, I64_SHRS);
    EBin(b, I64_NE);
  } else {
    // LT3 = u = a1*b0 + ((a0*b0) >> 32)   (all partials < 2^64, no overflow)
    EGet(b, LT0); EConst(b, 32); EBin(b, I64_SHRU);
    EGet(b, LT1); EConst(b, 0xffffffff); EBin(b, I64_AND);
    EBin(b, I64_MUL);
    EGet(b, LT0); EConst(b, 0xffffffff); EBin(b, I64_AND);
    EGet(b, LT1); EConst(b, 0xffffffff); EBin(b, I64_AND);
    EBin(b, I64_MUL); EConst(b, 32); EBin(b, I64_SHRU);
    EBin(b, I64_ADD); ESet(b, LT3);
    // hi(unsigned) = a1*b1 + (u >> 32) + ((a0*b1 + (u & 0xffffffff)) >> 32)
    EGet(b, LT0); EConst(b, 32); EBin(b, I64_SHRU);
    EGet(b, LT1); EConst(b, 32); EBin(b, I64_SHRU);
    EBin(b, I64_MUL);
    EGet(b, LT3); EConst(b, 32); EBin(b, I64_SHRU);
    EBin(b, I64_ADD);
    EGet(b, LT0); EConst(b, 0xffffffff); EBin(b, I64_AND);
    EGet(b, LT1); EConst(b, 32); EBin(b, I64_SHRU);
    EBin(b, I64_MUL);
    EGet(b, LT3); EConst(b, 0xffffffff); EBin(b, I64_AND);
    EBin(b, I64_ADD); EConst(b, 32); EBin(b, I64_SHRU);
    EBin(b, I64_ADD);
    // hi(signed) = hi - ((a>>s63)&b) - ((b>>s63)&a)
    EGet(b, LT0); EConst(b, 63); EBin(b, I64_SHRS);
    EGet(b, LT1); EBin(b, I64_AND);
    EBin(b, I64_SUB);
    EGet(b, LT1); EConst(b, 63); EBin(b, I64_SHRS);
    EGet(b, LT0); EBin(b, I64_AND);
    EBin(b, I64_SUB);
    // of = hs != (lo >>s 63)
    EGet(b, LT2); EConst(b, 63); EBin(b, I64_SHRS);
    EBin(b, I64_NE);
  }
  EConstI(b, 0x801); EBin(b, I32_MUL);
  EBin(b, I32_OR);
  ESet(b, FL);
  rc->fl_dirty = 1;
}

// Decide if this insn is an inlinable register-direct 32/64-bit imul. Two forms:
// OpImulGvqpEvqp reg=reg*rm ; OpImulGvqpEvqpImm reg=rm*imm.
static bool ImulDecode(nexgen32e_f h, u64 rde, u64 uimm0, int *dst, int *areg,
                       int *breg, bool *bimm, u64 *immv, int *log2) {
  int lg = RegLog2(rde);
  if (lg != 2 && lg != 3) return false;
  if (!IsModrmRegister(rde) || Lock(rde)) return false;  // reg-direct rm only
  *log2 = lg;
  *dst = 0; *areg = 0; *breg = 0; *bimm = false; *immv = 0;
  if (h == OpImulGvqpEvqp) {        // reg = reg * rm
    *dst = *areg = (int)RexrReg(rde); *breg = (int)RexbRm(rde); *bimm = false;
    return true;
  }
  if (h == OpImulGvqpEvqpImm) {     // reg = rm * imm
    *dst = (int)RexrReg(rde); *areg = (int)RexbRm(rde); *bimm = true; *immv = uimm0;
    return true;
  }
  return false;
}

// i64 binop per ALU op index (0 add,1 or,4 and,5 sub,6 xor,7 cmp=sub).
static const u8 kBin[8] = {I64_ADD, I64_OR, 0, 0, I64_AND, I64_SUB, I64_XOR,
                           I64_SUB};

// Inline ALU: z = x OP y in wasm (no call), exact blink flags into the cached
// flags local FL. Used for add/or/and/sub/xor/cmp/test (not adc/sbb). LT0=x,
// LT1=y, LT2=z; operands masked to width so the 32/64-bit paths share formulas.
static void EmitAluInline(struct Buf *b, struct Rc *rc, int t, int log2, int dst,
                          int src, bool imm, u64 immv, bool wb, bool keepcf,
                          int needed) {
  bool w32 = (log2 == 2);
  i64 mask = w32 ? (i64)0xffffffffLL : (i64)-1;
  int signsh = w32 ? 31 : 63;
  int kind = (t == 1 || t == 4 || t == 6) ? 0 : (t == 0 ? 1 : 2);  // 0 log,1 add,2 sub
  // inc/dec (keepcf) preserve CF: keep its bit and don't recompute it.
  i32 keep = keepcf ? 0x00fff72f : 0x00fff72e;
  // x -> LT0 (dst<0: x is the memory operand value already in LT3, no wb)
  if (dst >= 0) {
    RcLoad(b, rc, dst);
    EGet(b, LOC_GPR + dst);
  } else {
    EGet(b, LT3);
  }
  if (w32) { EConst(b, mask); EBin(b, I64_AND); }
  ESet(b, LT0);
  // y -> LT1 (src<0: y is the memory operand value already in LT3)
  if (imm) {
    EConst(b, w32 ? (i64)(u64)(u32)immv : (i64)immv);
  } else if (src >= 0) {
    RcLoad(b, rc, src);
    EGet(b, LOC_GPR + src);
    if (w32) { EConst(b, mask); EBin(b, I64_AND); }
  } else {
    EGet(b, LT3);
    if (w32) { EConst(b, mask); EBin(b, I64_AND); }
  }
  ESet(b, LT1);
  // z = (x OP y) & mask -> LT2
  EGet(b, LT0);
  EGet(b, LT1);
  EBin(b, kBin[t]);
  if (w32) { EConst(b, mask); EBin(b, I64_AND); }
  ESet(b, LT2);
  // flags: FL = (FL & keep) | cf | zf<<6 | sf<<7 | of<<11 | af<<4 | (z&0xFF)<<24
  // DEAD-FLAG ELIMINATION: skip entirely if no downstream reader needs them.
  if (needed) {
  FlagsEnsure(b, rc);
  EGet(b, FL);
  EConstI(b, keep);
  EBin(b, I32_AND);  // clears ZF SF OF AF parity (+ CF unless keepcf)
  // zf<<6
  EGet(b, LT2); EBin(b, I64_EQZ); EConstI(b, 6); EBin(b, I32_SHL); EBin(b, I32_OR);
  // sf<<7
  EGet(b, LT2); EConst(b, signsh); EBin(b, I64_SHRU); EBin(b, I64_WRAP);
  EConstI(b, 7); EBin(b, I32_SHL); EBin(b, I32_OR);
  if (kind != 0) {  // cf, of, af (logical leaves them 0)
    if (!keepcf) {  // cf: add z<y ; sub x<z (inc/dec preserve CF)
      if (kind == 1) { EGet(b, LT2); EGet(b, LT1); }
      else { EGet(b, LT0); EGet(b, LT2); }
      EBin(b, I64_LTU); EBin(b, I32_OR);  // cf<<0
    }
    // of<<11
    if (kind == 1) {  // ((z^x)&(z^y))
      EGet(b, LT2); EGet(b, LT0); EBin(b, I64_XOR);
      EGet(b, LT2); EGet(b, LT1); EBin(b, I64_XOR);
    } else {  // ((x^y)&(z^x))
      EGet(b, LT0); EGet(b, LT1); EBin(b, I64_XOR);
      EGet(b, LT2); EGet(b, LT0); EBin(b, I64_XOR);
    }
    EBin(b, I64_AND); EConst(b, signsh); EBin(b, I64_SHRU);
    EConst(b, 1); EBin(b, I64_AND); EBin(b, I64_WRAP);
    EConstI(b, 11); EBin(b, I32_SHL); EBin(b, I32_OR);
    // af<<4: add (z&15)<(y&15) ; sub (x&15)<(z&15)
    if (kind == 1) { EGet(b, LT2); EConst(b, 15); EBin(b, I64_AND);
                     EGet(b, LT1); EConst(b, 15); EBin(b, I64_AND); }
    else { EGet(b, LT0); EConst(b, 15); EBin(b, I64_AND);
           EGet(b, LT2); EConst(b, 15); EBin(b, I64_AND); }
    EBin(b, I64_LTU); EConstI(b, 4); EBin(b, I32_SHL); EBin(b, I32_OR);
  }
  // parity byte (z & 0xFF) << 24
  EGet(b, LT2); EConst(b, 0xff); EBin(b, I64_AND); EBin(b, I64_WRAP);
  EConstI(b, 24); EBin(b, I32_SHL); EBin(b, I32_OR);
  ESet(b, FL);
  rc->fl_dirty = 1;
  }  // if (needed)
  if (wb) {
    EGet(b, LT2);
    ESet(b, LOC_GPR + dst);
    rc->loaded[dst] = 1;
    rc->dirty[dst] = 1;
  }
}

// dst = src (or imm), with 32-bit zero-extend. No flags. (mov)
static void EmitMov(struct Buf *b, struct Rc *rc, int dst, int src, bool imm,
                    u64 immv, int log2) {
  if (imm) {
    EConst(b, log2 == 2 ? (i64)(u64)(u32)immv : (i64)immv);
  } else {
    RcLoad(b, rc, src);
    EGet(b, LOC_GPR + src);
    if (log2 == 2) { EConst(b, 0xffffffff); bput(b, 0x83); }  // i64.and (zero-ext)
  }
  ESet(b, LOC_GPR + dst);
  rc->loaded[dst] = 1;
  rc->dirty[dst] = 1;
}

// dst = disp + base + (index << scale), address-arithmetic only (no flags,
// no memory access). This is `lea`, which gcc emits pervasively as a cheap
// 3-operand add/shift. RIP-relative folds to a compile-time constant.
static void EmitLea(struct Buf *b, struct Rc *rc, int dst, int base, int index,
                    int scale, i64 dispv, bool hasBase, bool hasIndex,
                    bool riprel, bool legacy, int log2, u64 pc_next) {
  EConst(b, dispv + (riprel ? (i64)pc_next : 0));  // disp (+ rip)
  if (hasBase) { RcLoad(b, rc, base); EGet(b, LOC_GPR + base); EBin(b, I64_ADD); }
  if (hasIndex) {
    RcLoad(b, rc, index);
    EGet(b, LOC_GPR + index);
    if (scale) { EConst(b, (i64)scale); EBin(b, I64_SHL); }
    EBin(b, I64_ADD);
  }
  if (legacy) { EConst(b, (i64)0xffffffff); EBin(b, I64_AND); }  // 32-bit addr
  if (log2 == 2) { EConst(b, (i64)0xffffffff); EBin(b, I64_AND); }  // 32-bit dst ze
  ESet(b, LOC_GPR + dst);
  rc->loaded[dst] = 1;
  rc->dirty[dst] = 1;
}

// Decide if this insn is an inlinable lea with a 32/64-bit dst and non-16-bit
// addressing. Fills the address components. 16-bit dst/addr -> fallback.
static bool LeaDecode(nexgen32e_f h, u64 rde, i64 disp, int *dst, int *base,
                      int *index, int *scale, i64 *dispv, bool *hasBase,
                      bool *hasIndex, bool *riprel, bool *legacy, int *log2) {
  if (h != OpLeaGvqpM) return false;
  int lg = RegLog2(rde);
  if (lg != 2 && lg != 3) return false;      // 16-bit dst preserves upper: fallback
  int eam = Eamode(rde);
  if (eam == XED_MODE_REAL) return false;    // 16-bit addressing: fallback
  *legacy = (eam == XED_MODE_LEGACY);
  *log2 = lg;
  *dst = (int)RexrReg(rde);
  *dispv = disp;
  *base = *index = *scale = 0;
  *hasBase = *hasIndex = *riprel = false;
  if (!SibExists(rde)) {
    if (IsRipRelative(rde)) *riprel = true;
    else { *hasBase = true; *base = (int)RexbRm(rde); }
  } else {
    if (SibHasBase(rde)) { *hasBase = true; *base = (int)RexbBase(rde); }
    if (SibHasIndex(rde)) {
      *hasIndex = true;
      *index = (int)(Rexx(rde) << 3 | SibIndex(rde));
      *scale = (int)SibScale(rde);
    }
  }
  return true;
}

// ── inline memory operands (softmmu TLB fast path) ──────────────────────────
// The wasm build has NO linear mapping (CAN_64BIT=0): every guest access goes
// FindPageTableEntry (m->tlb[(page>>12)&31], PAGE_V) -> GetPageAddress ->
// g_hostpages.p[(entry&PAGE_TA)>>12] + (addr&4095). We inline exactly that TLB
// hit path for READS; any miss / permission fail / m->invalidated / page cross
// falls back to the real handler inline, so semantics match by construction.
// (Writes stay on the handler: SetWriteAddr does fork-COW splitting and the
// RW lookup path enqueues SMC invalidation - phase 2, not inlined yet.)

// Decode a MEMORY rm operand's effective address parts. Only the flat common
// case is inlined: long mode, 64-bit addressing, no segment override, no LOCK,
// userland (!metal: ds/ss/es bases are 0, no ROM aliasing, Cpl always 3).
static bool MemEaDecode(struct Machine *m, u64 rde, int *base, int *index,
                        int *scale, bool *hasBase, bool *hasIndex,
                        bool *riprel) {
  if (IsModrmRegister(rde) || Lock(rde)) return false;
  if (m->metal) return false;
  if (Mode(rde) != XED_MODE_LONG || Eamode(rde) != XED_MODE_LONG) return false;
  if (Sego(rde)) return false;  // fs/gs override -> handler
  *base = *index = *scale = 0;
  *hasBase = *hasIndex = *riprel = false;
  if (!SibExists(rde)) {
    if (IsRipRelative(rde)) *riprel = true;
    else { *hasBase = true; *base = (int)RexbRm(rde); }
  } else {
    if (SibHasBase(rde)) { *hasBase = true; *base = (int)RexbBase(rde); }
    if (SibHasIndex(rde)) {
      *hasIndex = true;
      *index = (int)(Rexx(rde) << 3 | SibIndex(rde));
      *scale = (int)SibScale(rde);
    }
  }
  return true;
}

// LT3 = disp + base + (index << scale) [+ pc_next when rip-relative].
static void EmitEaAddr(struct Buf *b, struct Rc *rc, int base, int index,
                       int scale, i64 dispv, bool hasBase, bool hasIndex,
                       bool riprel, u64 pc_next) {
  EConst(b, dispv + (riprel ? (i64)pc_next : 0));
  if (hasBase) { RcLoad(b, rc, base); EGet(b, LOC_GPR + base); EBin(b, I64_ADD); }
  if (hasIndex) {
    RcLoad(b, rc, index);
    EGet(b, LOC_GPR + index);
    if (scale) { EConst(b, (i64)scale); EBin(b, I64_SHL); }
    EBin(b, I64_ADD);
  }
  ESet(b, LT3);
}

// LT3 holds the guest addr. Opens block $done { block $slow { ... and emits the
// TLB-hit checks; on the fast path HP0 = host pointer for [addr, addr+size).
// Any check failing branches to $slow (the handler fallback in EmitMemEnd).
static void EmitMemBegin(struct Buf *b, int size) {
  bput(b, 0x02); bput(b, 0x40);   // block $done
  bput(b, 0x02); bput(b, 0x40);   // block $slow
  // if (m->invalidated) goto slow  (interp resets the TLB before trusting it)
  EGet(b, 0); bput(b, 0x2d); bleb_u(b, 0); bleb_u(b, OFF_INVAL);  // i32.load8_u
  bput(b, 0x0d); bleb_u(b, 0);    // br_if $slow
  // HP0 = m + ((addr>>12) & 31)*16   (tlb slot; OFF_TLB applied at each load)
  EGet(b, LT3); EConst(b, 12); EBin(b, I64_SHRU); bput(b, I64_WRAP);
  EConstI(b, 31); bput(b, I32_AND); EConstI(b, 4); bput(b, I32_SHL);
  EGet(b, 0); bput(b, I32_ADD); ESet(b, HP0);
  // tlb.page != (addr & -4096) -> slow
  EGet(b, HP0); bput(b, 0x29); bleb_u(b, 0); bleb_u(b, OFF_TLB);
  EGet(b, LT3); EConst(b, -4096); EBin(b, I64_AND);
  bput(b, I64_NE); bput(b, 0x0d); bleb_u(b, 0);
  // entry -> LT2; need PAGE_V|PAGE_U|PAGE_HOST with PAGE_RSRV clear
  // (read perms = LookupAddress2(mask=need=PAGE_U) at Cpl 3; RSRV can't be in
  // the TLB, checked anyway; !HOST would mean s->real, not inlined)
  EGet(b, HP0); bput(b, 0x29); bleb_u(b, 0); bleb_u(b, OFF_TLB + 8);
  bput(b, 0x22); bleb_u(b, LT2);  // local.tee
  EConst(b, (i64)(PAGE_V | PAGE_U | PAGE_HOST | PAGE_RSRV)); EBin(b, I64_AND);
  EConst(b, (i64)(PAGE_V | PAGE_U | PAGE_HOST));
  bput(b, I64_NE); bput(b, 0x0d); bleb_u(b, 0);
  // HP1 = addr & 4095; page-crossing access -> slow (handler stashes those)
  EGet(b, LT3); bput(b, I64_WRAP); EConstI(b, 4095); bput(b, I32_AND);
  if (size > 1) {
    bput(b, 0x22); bleb_u(b, HP1);  // local.tee
    EConstI(b, 4096 - size); bput(b, I32_GTU); bput(b, 0x0d); bleb_u(b, 0);
  } else {
    ESet(b, HP1);
  }
  // HP0 = g_hostpages.p[(entry & PAGE_TA) >> 12] + HP1
  EConstI(b, (i32)(uintptr_t)&g_hostpages.p);
  bput(b, 0x28); bleb_u(b, 0); bleb_u(b, 0);   // i32.load p
  EGet(b, LT2); EConst(b, (i64)PAGE_TA); EBin(b, I64_AND);
  EConst(b, 12); EBin(b, I64_SHRU); bput(b, I64_WRAP);
  EConstI(b, 2); bput(b, I32_SHL); bput(b, I32_ADD);
  bput(b, 0x28); bleb_u(b, 0); bleb_u(b, 0);   // i32.load p[idx] (page base)
  EGet(b, HP1); bput(b, I32_ADD); ESet(b, HP0);
}

// Close the fast path and emit the $slow fallback: spill the PRE-instruction
// dirty state, run the real handler (ip/oplen set for fault rewind), drain the
// stash, then RELOAD every local the post-instruction cache state says is
// valid - so both paths rejoin with the identical compile-time cache state.
static void EmitMemEnd(struct Buf *b, const struct Rc *pre, const struct Rc *rc,
                       u64 rde, i64 disp, u64 uimm0, u32 hidx, u64 pc,
                       u32 oplen) {
  int r;
  bput(b, 0x0c); bleb_u(b, 1);    // fast path done: br $done
  bput(b, 0x0b);                  // end $slow
  for (r = 0; r < 16; ++r) {      // spill pre-instruction dirty regs
    if (!pre->dirty[r]) continue;
    EGet(b, 0); EGet(b, LOC_GPR + r); EStore(b, OFF_WEG + (u32)r * 8);
  }
  if (pre->fl_dirty) {
    EGet(b, 0); EGet(b, FL);
    bput(b, 0x36); bleb_u(b, 2); bleb_u(b, OFF_FLAGS);  // i32.store m->flags
  }
  EGet(b, 0); EConst(b, (i64)pc); EStore(b, OFF_IP);
  EGet(b, 0); EConstI(b, (i32)oplen);
  bput(b, 0x3a); bleb_u(b, 0); bleb_u(b, OFF_OPLEN);    // i32.store8
  EmitHandler(b, rde, disp, uimm0, hidx);
  EmitCommitStash(b);
  for (r = 0; r < 16; ++r) {      // resync locals the fast path left valid
    if (!rc->loaded[r]) continue;
    EGet(b, 0); ELoad(b, OFF_WEG + (u32)r * 8); ESet(b, LOC_GPR + r);
  }
  if (rc->fl_loaded) {
    EGet(b, 0);
    bput(b, 0x28); bleb_u(b, 2); bleb_u(b, OFF_FLAGS);  // i32.load m->flags
    ESet(b, FL);
  }
  bput(b, 0x0b);                  // end $done
}

// Try to inline an instruction whose rm is a MEMORY operand being READ, with
// a 32/64-bit register destination (or flags-only). Returns false (emitting
// nothing) if the form isn't covered - caller falls back to the handler.
static bool TryEmitMemRead(struct Machine *m, struct Buf *b, struct Rc *rc,
                           nexgen32e_f h, u64 rde, const struct XedDecodedInst *x,
                           u64 pc_next, u32 oplen) {
  int lg = (int)RegLog2(rde);
  int kind;  // 0 mov load->reg ; 1 alu reg,[mem] ; 2 cmp/test [mem],reg ; 3 cmp [mem],imm
  int t = 0, dst = 0, ysrc = 0, size = 0;
  u8 loadop = 0;
  bool wb = false, sx32 = false;
  if (lg != 2 && lg != 3) return false;  // 16-bit dst keeps upper bits: fallback
  if (h == OpMovGvqpEvqp) {              // mov reg, [mem]
    kind = 0; dst = (int)RexrReg(rde);
    size = 1 << lg; loadop = lg == 2 ? 0x35 : 0x29;  // i64.load32_u / i64.load
  } else if (h == OpMovzbGvqpEb) {       // movzx reg, byte
    kind = 0; dst = (int)RexrReg(rde); size = 1; loadop = 0x31;
  } else if (h == OpMovzwGvqpEw) {       // movzx reg, word
    kind = 0; dst = (int)RexrReg(rde); size = 2; loadop = 0x33;
  } else if (h == OpMovsbGvqpEb) {       // movsx reg, byte
    kind = 0; dst = (int)RexrReg(rde); size = 1; loadop = 0x30; sx32 = lg == 2;
  } else if (h == OpMovswGvqpEw) {       // movsx reg, word
    kind = 0; dst = (int)RexrReg(rde); size = 2; loadop = 0x32; sx32 = lg == 2;
  } else if (h == OpMovslGdqpEd) {       // movsxd reg, dword
    kind = 0; dst = (int)RexrReg(rde); size = 4; loadop = 0x34; sx32 = lg == 2;
  } else if (h == OpAluFlip) {           // ALU reg, [mem] (writeback to reg)
    kind = 1; t = (int)((Opcode(rde) & 070) >> 3); dst = (int)RexrReg(rde);
    wb = true; size = 1 << lg; loadop = lg == 2 ? 0x35 : 0x29;
  } else if (h == OpAluFlipCmp) {        // cmp reg, [mem] (flags only)
    kind = 1; t = ALU_SUB; dst = (int)RexrReg(rde);
    size = 1 << lg; loadop = lg == 2 ? 0x35 : 0x29;
  } else if (h == OpAluCmp) {            // cmp [mem], reg (flags only)
    kind = 2; t = ALU_SUB; ysrc = (int)RexrReg(rde);
    size = 1 << lg; loadop = lg == 2 ? 0x35 : 0x29;
  } else if (h == OpAluTest) {           // test [mem], reg (flags only)
    kind = 2; t = ALU_AND; ysrc = (int)RexrReg(rde);
    size = 1 << lg; loadop = lg == 2 ? 0x35 : 0x29;
  } else if (h == OpAlui && (int)ModrmReg(rde) == ALU_CMP) {  // cmp [mem], imm
    kind = 3; t = ALU_CMP;
    size = 1 << lg; loadop = lg == 2 ? 0x35 : 0x29;
  } else {
    return false;
  }
  int base, index, scale;
  bool hb, hi, rip;
  if (!MemEaDecode(m, rde, &base, &index, &scale, &hb, &hi, &rip)) return false;
  EmitEaAddr(b, rc, base, index, scale, x->op.disp, hb, hi, rip, pc_next);
  struct Rc pre = *rc;            // the $slow fallback spills THIS state
  EmitMemBegin(b, size);
  EGet(b, HP0);
  bput(b, loadop); bleb_u(b, 0); bleb_u(b, 0);  // load [HP0] (align 0)
  if (kind == 0) {
    if (sx32) { EConst(b, 0xffffffff); EBin(b, I64_AND); }  // 32-bit dst ze
    ESet(b, LOC_GPR + dst);
    rc->loaded[dst] = 1;
    rc->dirty[dst] = 1;
  } else {
    ESet(b, LT3);                 // memory operand value (addr no longer needed)
    int need = GetNeededFlags(m, pc_next, CF | ZF | SF | OF | AF | PF);
    if (kind == 1) {
      if (t == 2 || t == 3) EmitAlu(b, rc, t, lg, dst, -1, false, 0, wb);
      else EmitAluInline(b, rc, t, lg, dst, -1, false, 0, wb, false, need);
    } else if (kind == 2) {
      EmitAluInline(b, rc, t, lg, -1, ysrc, false, 0, false, false, need);
    } else {
      EmitAluInline(b, rc, t, lg, -1, 0, true, x->op.uimm0, false, false, need);
    }
  }
  EmitMemEnd(b, &pre, rc, rde, x->op.disp, x->op.uimm0, (u32)(uintptr_t)h,
             pc_next, oplen);
  return true;
}

// Decide if this insn is an inlinable register-direct 32/64-bit mov.
static bool MovDecode(nexgen32e_f h, u64 rde, u64 uimm0, int *dst, int *src,
                      bool *imm, u64 *immv, int *log2) {
  int lg = RegLog2(rde);
  if (lg != 2 && lg != 3) return false;
  *log2 = lg;
  *dst = 0; *src = 0; *imm = false; *immv = 0;
  // every out-param gets a defined value on every true path: callers pass
  // these (by value) into the emitters, and an indeterminate value there is
  // poison the optimizer may act on
  if (h == OpMovZvqpIvqp) {  // mov reg, imm
    *dst = (int)RexbSrm(rde); *imm = true; *immv = uimm0;
    return true;
  }
  if (!IsModrmRegister(rde)) return false;  // reg-reg only (memory = later)
  if (h == OpMovEvqpGvqp) {  // mov rm, reg
    *dst = (int)RexbRm(rde); *src = (int)RexrReg(rde); *imm = false;
    return true;
  }
  if (h == OpMovGvqpEvqp) {  // mov reg, rm
    *dst = (int)RexrReg(rde); *src = (int)RexbRm(rde); *imm = false;
    return true;
  }
  return false;
}

// Decide if this insn is an inlinable register-direct 32/64-bit ALU op.
static bool AluDecode(nexgen32e_f h, u64 rde, u64 uimm0, int *t, int *log2,
                      int *dst, int *src, bool *imm, u64 *immv, bool *wb) {
  if (!IsModrmRegister(rde)) return false;  // memory operand -> fallback (M2)
  if (Lock(rde)) return false;
  int lg = RegLog2(rde);
  if (lg != 2 && lg != 3) return false;  // only 32/64-bit
  *log2 = lg;
  *src = 0;    // defined on every true path (see MovDecode)
  *immv = 0;
  int op = (int)((Opcode(rde) & 070) >> 3);
  int rm = (int)RexbRm(rde), reg = (int)RexrReg(rde);
  *t = 0; *dst = 0; *src = 0; *imm = false; *immv = 0; *wb = false;
  if (h == OpAluw) {  // ALU Ev,Gv (add/or/adc/sbb/and/sub/xor), writeback to rm
    *t = op; *dst = rm; *src = reg; *imm = false; *wb = true;
    return true;
  }
  if (h == OpAluFlip) {  // ALU Gv,Ev, writeback to reg
    *t = op; *dst = reg; *src = rm; *imm = false; *wb = true;
    return true;
  }
  if (h == OpAluCmp) {  // cmp Ev,Gv: flags of (rm - reg), no writeback
    *t = ALU_SUB; *dst = rm; *src = reg; *imm = false; *wb = false;
    return true;
  }
  if (h == OpAluFlipCmp) {  // cmp Gv,Ev: flags of (reg - rm), no writeback
    *t = ALU_SUB; *dst = reg; *src = rm; *imm = false; *wb = false;
    return true;
  }
  if (h == OpAluTest) {  // test Ev,Gv: flags of (rm & reg), no writeback
    *t = ALU_AND; *dst = rm; *src = reg; *imm = false; *wb = false;
    return true;
  }
  if (h == OpAlui) {  // group1 ALU rm, imm; op = ModrmReg
    *t = (int)ModrmReg(rde); *dst = rm; *src = 0; *imm = true; *immv = uimm0;
    *wb = (*t != ALU_CMP);
    return true;
  }
  return false;
}

// A block whose ONLY control transfer is a backward Jcc to its own start, with
// an all-inline body, is emitted as a wasm `loop` that iterates entirely in wasm
// (the engine JITs it to ~native) instead of returning to the C dispatcher each
// iteration. Returns true and fills bb/rc if it emitted one.
//
// The pre-pass decodes each insn ONCE into ops[] and the emit loop replays the
// record. (The original re-decoded the stream in a second pass; the duplicated
// classification let LLVM jump-thread a "can't happen" decode combination into
// a literal `unreachable` that fired at runtime - see memory x86-wasm-jit.)
struct SlOp {
  u8 kind;                        // kSl* below
  u8 t, lg, d, s, im, w;          // alu/mov/bsu: op, width, regs, imm?, writeback?
  u8 lb, li, lsc, lhb, lhi, lrp, llg;  // lea: base/index/scale/flags
  u8 b;                           // imul: second source reg
  int need;                       // alu-inline: flags needed downstream
  u64 iv;                         // immediate value
  i64 ldv;                        // lea displacement
  u64 pcn;                        // pc after this insn (lea rip base)
};
enum { kSlAluCall, kSlAluInline, kSlMov, kSlLea, kSlBsu, kSlBsuInline,
       kSlImul };
#define PKJIT_SLMAX 48  // self-loop insn cap (hot loops are short; bounds ops[])

static bool EmitSelfLoop(struct Machine *m, u64 ip, struct Buf *bb,
                         struct Rc *rc) {
  struct XedDecodedInst x;
  struct SlOp ops[PKJIT_SLMAX];
  u64 pc = ip;
  u16 regs = 0;
  int cc = -1;
  u64 fall = 0;
  int n, cnt = 0;
  for (n = 0; n < PKJIT_SLMAX; ++n) {  // pre-pass: decode + record + collect regs
    if (GetInstruction(m, pc, &x)) return false;
    u64 rde = x.op.rde;
    u32 ol = Oplength(rde);
    if (!ol) return false;
    nexgen32e_f h = GetOp(Mopcode(rde));
    u64 pcn = pc + ol;
    if (h == OpJcc) {  // terminator
      int c = (int)(Opcode(rde) & 15);
      if (c == 0xa || c == 0xb) return false;  // JP/JNP need lazy parity
      if (!(n > 0 && (u64)(pcn + x.op.disp) == ip)) return false;
      cc = c;
      fall = pcn;
      break;
    }
    struct SlOp *o = &ops[cnt];
    memset(o, 0, sizeof(*o));
    o->pcn = pcn;
    int t = 0, lg = 0, d = 0, s = 0;
    bool im = false, w = false;
    u64 iv = 0;
    int lb = 0, li = 0, lsc = 0; i64 ldv = 0;
    bool lhb = false, lhi = false, lrp = false, llg = false;
    if (AluDecode(h, rde, x.op.uimm0, &t, &lg, &d, &s, &im, &iv, &w)) {
      o->kind = (t == 2 || t == 3) ? kSlAluCall : kSlAluInline;  // adc/sbb call
      o->t = t; o->lg = lg; o->d = d; o->s = s; o->im = im; o->w = w; o->iv = iv;
      o->need = GetNeededFlags(m, (i64)pcn, CF | ZF | SF | OF | AF | PF);
      regs |= 1u << d;
      if (!im) regs |= 1u << s;
    } else if (MovDecode(h, rde, x.op.uimm0, &d, &s, &im, &iv, &lg)) {
      o->kind = kSlMov;
      o->d = d; o->s = s; o->im = im; o->iv = iv; o->lg = lg;
      regs |= 1u << d;
      if (!im) regs |= 1u << s;
    } else if (LeaDecode(h, rde, x.op.disp, &d, &lb, &li, &lsc, &ldv, &lhb, &lhi,
                         &lrp, &llg, &lg)) {
      o->kind = kSlLea;
      o->d = d; o->lb = lb; o->li = li; o->lsc = lsc; o->ldv = ldv;
      o->lhb = lhb; o->lhi = lhi; o->lrp = lrp; o->llg = llg; o->lg = lg;
      regs |= 1u << d;
      if (lhb) regs |= 1u << lb;
      if (lhi) regs |= 1u << li;
    } else if (BsuDecode(h, rde, x.op.uimm0, &t, &lg, &d, &im, &iv)) {
      // flags dead + not rcl/rcr + not 32-bit rot -> pure wasm op, no call
      if (!GetNeededFlags(m, (i64)pcn, CF | ZF | SF | OF | AF | PF) &&
          t != 2 && t != 3 && (lg == 3 || t >= 4)) {
        o->kind = kSlBsuInline;
      } else {
        o->kind = kSlBsu;  // exact flags via kBsu leaf
      }
      o->t = t; o->lg = lg; o->d = d; o->im = im; o->iv = iv;
      regs |= 1u << d;
      if (!im) regs |= 1u << 1;  // CL form reads RCX
    } else if (ImulDecode(h, rde, x.op.uimm0, &d, &s, &lb, &im, &iv, &lg) &&
               !GetNeededFlags(m, (i64)pcn, CF | OF)) {
      o->kind = kSlImul;  // imul low result; CF/OF (its only flags) are dead
      o->d = d; o->s = s; o->b = lb; o->im = im; o->iv = iv; o->lg = lg;
      regs |= 1u << d;
      regs |= 1u << s;
      if (!im) regs |= 1u << lb;
    } else if ((h == OpIncEvqp || h == OpDecEvqp) && IsModrmRegister(rde) &&
               (RegLog2(rde) == 2 || RegLog2(rde) == 3)) {
      // inc/dec via kAlu[10/11](x,0): exact flags (AF=0, CF preserved)
      o->kind = kSlAluCall;
      o->t = h == OpIncEvqp ? 10 : 11;
      o->lg = (u8)RegLog2(rde); o->d = (u8)RexbRm(rde);
      o->im = 1; o->w = 1;
      regs |= 1u << o->d;
    } else {
      return false;
    }
    ++cnt;
    pc = pcn;
  }
  if (cc < 0 || !cnt) return false;

  // preamble: hoist reg + flags loads OUT of the loop (persist across iterations)
  for (int r = 0; r < 16; ++r) if (regs & (1u << r)) RcLoad(bb, rc, r);
  FlagsEnsure(bb, rc);
  bput(bb, 0x02); bput(bb, 0x40);  // block $B
  bput(bb, 0x03); bput(bb, 0x40);  // loop $L
  // attention check at loop top: if (m->attention) { m->ip = ip; exit }
  EGet(bb, 0); bput(bb, 0x2d); bleb_u(bb, 0); bleb_u(bb, OFF_ATT);  // i32.load8_u
  bput(bb, 0x04); bput(bb, 0x40);  // if
  EGet(bb, 0); EConst(bb, (i64)ip); EStore(bb, OFF_IP);
  bput(bb, 0x0c); bleb_u(bb, 2);   // br 2 -> after block
  bput(bb, 0x0b);                  // end if
  // body: replay the recorded ops (RcLoad no-ops since pre-loaded)
  for (int i = 0; i < cnt; ++i) {
    struct SlOp *o = &ops[i];
    switch (o->kind) {
      case kSlAluCall:
        EmitAlu(bb, rc, o->t, o->lg, o->d, o->s, o->im, o->iv, o->w);
        break;
      case kSlAluInline:
        EmitAluInline(bb, rc, o->t, o->lg, o->d, o->s, o->im, o->iv, o->w,
                      false, o->need);
        break;
      case kSlMov:
        EmitMov(bb, rc, o->d, o->s, o->im, o->iv, o->lg);
        break;
      case kSlBsu:
        EmitBsu(bb, rc, o->t, o->lg, o->d, o->im, o->iv);
        break;
      case kSlBsuInline:
        EmitBsuInline(bb, rc, o->t, o->lg, o->d, o->im, o->iv);
        break;
      case kSlImul:
        EmitImul(bb, rc, o->d, o->s, o->b, o->im, o->iv, o->lg);
        break;
      default:  // kSlLea
        EmitLea(bb, rc, o->d, o->lb, o->li, o->lsc, o->ldv, o->lhb, o->lhi,
                o->lrp, o->llg, o->lg, o->pcn);
        break;
    }
  }
  // condition: loop back if the jcc is taken
  FlagsEnsure(bb, rc);
  EmitCond(bb, cc);
  bput(bb, 0x0d); bleb_u(bb, 0);   // br_if 0 -> loop $L
  // not taken: m->ip = fall-through, exit
  EGet(bb, 0); EConst(bb, (i64)fall); EStore(bb, OFF_IP);
  bput(bb, 0x0c); bleb_u(bb, 1);   // br 1 -> after block
  bput(bb, 0x0b);                  // end loop
  bput(bb, 0x0b);                  // end block
  RcSpill(bb, rc);                 // postamble: flush cache to memory
  t_wasloop = 1;
  return true;
}

static bool WasmJitEmit(struct Machine *m, u64 ip, const u8 **out, u32 *outlen) {
  if (!t_scratch && !(t_scratch = (u8 *)malloc(PKJIT_SCRATCH))) return false;
  struct Buf bb = {t_scratch, 0, PKJIT_SCRATCH, 0};
  struct Rc rc;
  RcInval(&rc);
  t_wasloop = 0;
  struct XedDecodedInst xedd;
  u64 pc = ip;
  int count = 0;
  bool ip_dirty = false;   // true if m->ip != pc (an inline op advanced pc)
  bool terminated = false;
  // SELF-LOOP: a hot backward-Jcc loop with an all-inline body compiles to a
  // single wasm `loop` that iterates entirely in wasm (~native speed; the alu
  // bench drops 1059ms -> 16ms). ON by default; -DPKJIT_NO_SELFLOOP disables
  // for debugging. (The historic 2nd-self-loop "unreachable" crash was
  // uninit-local poison jump-threaded by -O3 into a literal unreachable in the
  // old re-decoding body pass - fixed by decode-once replay + defined decode
  // outs; see memory x86-wasm-jit.)
#ifndef PKJIT_NO_SELFLOOP
  if (EmitSelfLoop(m, ip, &bb, &rc)) goto assemble;
#else
  (void)EmitSelfLoop;
#endif
  for (; count < PKJIT_MAXINSN; ++count) {
    if (GetInstruction(m, pc, &xedd)) break;
    u64 rde = xedd.op.rde;
    u32 oplen = Oplength(rde);
    if (!oplen) break;
    nexgen32e_f h = GetOp(Mopcode(rde));
    pc += oplen;
    int t = 0, log2 = 0, dst = 0, src = 0;
    bool imm = false, wb = false;
    u64 immv = 0;
    int lbase = 0, lindex = 0, lscale = 0; i64 ldispv = 0;
    bool lhb = false, lhi = false, lrip = false, lleg = false;
    int iareg = 0, ibreg = 0;
    if (AluDecode(h, rde, xedd.op.uimm0, &t, &log2, &dst, &src, &imm, &immv,
                  &wb)) {
      // pc is already past this insn; skip flag emission if all flags are dead.
      int need = GetNeededFlags(m, pc, CF | ZF | SF | OF | AF | PF);
      if (t == 2 || t == 3) EmitAlu(&bb, &rc, t, log2, dst, src, imm, immv, wb);
      else EmitAluInline(&bb, &rc, t, log2, dst, src, imm, immv, wb, false, need);
      ip_dirty = true;  // inline op: m->ip not updated
    } else if (MovDecode(h, rde, xedd.op.uimm0, &dst, &src, &imm, &immv,
                         &log2)) {
      EmitMov(&bb, &rc, dst, src, imm, immv, log2);
      ip_dirty = true;
    } else if (LeaDecode(h, rde, xedd.op.disp, &dst, &lbase, &lindex, &lscale,
                         &ldispv, &lhb, &lhi, &lrip, &lleg, &log2)) {
      EmitLea(&bb, &rc, dst, lbase, lindex, lscale, ldispv, lhb, lhi, lrip, lleg,
              log2, pc);  // pc already advanced past this insn = rip base
      ip_dirty = true;
    } else if (BsuDecode(h, rde, xedd.op.uimm0, &t, &log2, &dst, &imm, &immv)) {
      // flags dead + not rcl/rcr + not 32-bit rot -> pure wasm op, no call
      if (!GetNeededFlags(m, pc, CF | ZF | SF | OF | AF | PF) &&
          t != 2 && t != 3 && (log2 == 3 || t >= 4)) {
        EmitBsuInline(&bb, &rc, t, log2, dst, imm, immv);
      } else {
        EmitBsu(&bb, &rc, t, log2, dst, imm, immv);  // t = BSU_* op index
      }
      ip_dirty = true;
    } else if (ImulDecode(h, rde, xedd.op.uimm0, &dst, &iareg, &ibreg, &imm,
                          &immv, &log2)) {
      // exact CF/OF emitted only if a downstream reader needs them
      EmitImul(&bb, &rc, dst, iareg, ibreg, imm, immv, log2,
               GetNeededFlags(m, pc, CF | OF));
      ip_dirty = true;
    } else if ((h == OpIncEvqp || h == OpDecEvqp) && IsModrmRegister(rde) &&
               (RegLog2(rde) == 2 || RegLog2(rde) == 3)) {
      // inc/dec reg via blink's own kAlu[INC/DEC](x, 0) - exact flag semantics
      // (AF=0, CF preserved) that a hand-inlined add/sub-1 would get subtly wrong.
      EmitAlu(&bb, &rc, h == OpIncEvqp ? 10 : 11, (int)RegLog2(rde),
              (int)RexbRm(rde), 0, true, 0, true);
      ip_dirty = true;
    } else if (TryEmitMemRead(m, &bb, &rc, h, rde, &xedd, pc, oplen)) {
      ip_dirty = true;  // fast path leaves m->ip stale ($slow sets it itself)
    } else {
      RcSpill(&bb, &rc);                    // handler reads regs from memory
      EGet(&bb, 0); EConst(&bb, (i64)pc); EStore(&bb, OFF_IP);  // m->ip = pc
      // m->oplen = this insn's length, so RestoreIp (m->ip -= m->oplen) rewinds
      // correctly if the handler faults (matches JitlessDispatch, machine.c:2105).
      EGet(&bb, 0); EConstI(&bb, (i32)oplen);
      bput(&bb, 0x3a); bleb_u(&bb, 0); bleb_u(&bb, OFF_OPLEN);  // i32.store8
      EmitHandler(&bb, rde, xedd.op.disp, xedd.op.uimm0, (u32)(uintptr_t)h);
      EmitCommitStash(&bb);                 // drain page-crossing store stash
      RcInval(&rc);                         // handler may have changed regs
      ip_dirty = false;
      if (ClassifyOp(rde) != kOpNormal) {   // branch/precious ends the block
        terminated = true;
        ++count;
        break;
      }
    }
    if (bb.ovf) return false;
  }
  if (!count || bb.ovf) return false;
  if (!terminated) {  // finalize: flush cache + set m->ip to the fall-through
    RcSpill(&bb, &rc);
    if (ip_dirty) { EGet(&bb, 0); EConst(&bb, (i64)pc); EStore(&bb, OFF_IP); }
  }

assemble:
  if (bb.ovf) return false;
  u32 body_len = 5 + bb.n + 1;  // locals(02 14 7e 01 7f) + instrs + end
  u32 content_len = 1 + leb_u_size(body_len) + body_len;
  u32 total = (u32)sizeof(kPrefix) + 1 + leb_u_size(content_len) + content_len;
  u8 *mem = (u8 *)malloc(total);
  if (!mem) return false;
  struct Buf mb = {mem, 0, total, 0};
  bputs(&mb, kPrefix, (u32)sizeof(kPrefix));
  bput(&mb, 0x0a);
  bleb_u(&mb, content_len);
  bleb_u(&mb, 1);         // 1 function body
  bleb_u(&mb, body_len);
  bput(&mb, 0x02); bput(&mb, 0x14); bput(&mb, 0x7e);  // 20 i64 (16 GPR + LT0..3)
  bput(&mb, 0x03); bput(&mb, 0x7f);                   // 3 i32 (FL, HP0, HP1)
  bputs(&mb, bb.p, bb.n);
  bput(&mb, 0x0b);        // end
  if (mb.ovf) { free(mem); return false; }
  *out = mem;
  *outlen = mb.n;
  return true;
}

// ── driver ──────────────────────────────────────────────────────────────────
static int WasmJitEnabled(void) {
  int r = g_pkjit_ready;
  if (r) return r > 0;
  const char *e = getenv("BLINK_WASMJIT");
  g_pkjit_ready = r = (e && e[0] == '0') ? -1 : 1;
  return r > 0;
}
static inline u32 HashShared(u64 ip) {
  return (u32)((ip * 2654435761u) >> 11) & (PKJIT_SHARED - 1);
}
static inline u32 HashLocal(u64 ip) {
  return (u32)((ip * 2654435761u) >> 11) & (PKJIT_LOCAL - 1);
}

static nexgen32e_f InstantiateLocal(struct SharedEntry *s, u64 ip, u32 gen,
                                    int first_compile) {
  int idx = pk_jit_install(s->bytes, (int)s->len, first_compile);
  if (idx <= 0) return 0;
  struct LocalHook *lh = &t_local[HashLocal(ip)];
  lh->idx = (u32)idx;
  lh->gen = gen;
  lh->virt = ip;
  return (nexgen32e_f)(uintptr_t)idx;
}

nexgen32e_f WasmJitLookup(struct Machine *m, u64 ip) {
  if (!t_local) {
    if (!WasmJitEnabled()) return 0;
    t_local = (struct LocalHook *)calloc(PKJIT_LOCAL, sizeof(struct LocalHook));
    if (!t_local) { g_pkjit_ready = -1; return 0; }
  }
  u32 curgen = atomic_load_explicit(&g_codegen, memory_order_acquire);
  struct LocalHook *lh = &t_local[HashLocal(ip)];
  if (lh->virt == ip && lh->idx && lh->gen == curgen)
    return (nexgen32e_f)(uintptr_t)lh->idx;

  struct SharedEntry *s = &g_shared[HashShared(ip)];
  u64 sv = atomic_load_explicit(&s->virt, memory_order_acquire);
  if (sv == 0) {
    u64 expect = 0;
    if (atomic_compare_exchange_strong_explicit(&s->virt, &expect, ip,
                                                memory_order_acq_rel,
                                                memory_order_acquire)) {
      sv = ip;
    } else {
      sv = expect;
    }
  }
  if (sv != ip) return 0;

  u32 st = atomic_load_explicit(&s->state, memory_order_acquire);
  if (st == kReady) {
    if (s->gen == curgen) return InstantiateLocal(s, ip, curgen, 0);
    // stale code (SMC): grab the re-emit, else another thread has it
    u32 want = kReady;
    if (!atomic_compare_exchange_strong_explicit(&s->state, &want, kEmitting,
                                                 memory_order_acq_rel,
                                                 memory_order_acquire)) {
      return 0;
    }
    // fall through to re-emit
  } else if (st == kEmitting) {
    return 0;
  } else {  // warming
    if (atomic_fetch_add_explicit(&s->hits, 1, memory_order_relaxed) + 1 <
        PKJIT_THRESHOLD) {
      return 0;
    }
    u32 want = kWarming;
    if (!atomic_compare_exchange_strong_explicit(&s->state, &want, kEmitting,
                                                 memory_order_acq_rel,
                                                 memory_order_acquire)) {
      return 0;
    }
  }
  const u8 *bytes;
  u32 len;
  if (!WasmJitEmit(m, ip, &bytes, &len)) {
    atomic_store_explicit(&s->state, kEmitting, memory_order_release);  // park
    return 0;
  }
  s->bytes = bytes;  // old bytes (if re-emit) intentionally leaked: other threads
  s->len = len;      // may still read them; SMC is rare so the leak is bounded
  s->gen = curgen;
  atomic_store_explicit(&s->state, kReady, memory_order_release);
  u32 seq = atomic_fetch_add_explicit(&g_emits, 1, memory_order_relaxed) + 1;
  return InstantiateLocal(s, ip, curgen, t_wasloop ? -(int)seq : (int)seq);
}

#endif  // HAVE_WASM_JIT
