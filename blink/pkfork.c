// pk910.de: real fork - the child in its own wasm instance.
//
// fork() serialises the calling thread's machine and its address space into
// one buffer (registers, TLS segments, the signal table, brk/mmap state, the
// fd list, every present or reserved page) and hands it to the kernel, which
// spawns a child worker that restores it and continues after the syscall
// with rax = 0. The parent returns the child's pid at once and keeps
// running: fork semantics, not vfork's. (vfork keeps the in-place child in
// syscall.c: that IS vfork's contract.)
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blink/assert.h"
#include "blink/builtin.h"
#include "blink/bus.h"
#include "blink/errno.h"
#include "blink/dll.h"
#include "blink/endian.h"
#include "blink/fds.h"
#include "blink/linux.h"
#include "blink/machine.h"
#include "blink/macros.h"

#ifdef __EMSCRIPTEN__

// the JS side (blink-lib.js): send a snapshot, get a pid or -errno; in the
// child, fetch the snapshot the worker was started with
extern int js_fork(const void *buf, unsigned len);
extern unsigned js_fork_snapshot_size(void);
extern void js_fork_snapshot_read(void *buf, unsigned len);

#define PK_SNAP_MAGIC 0x31304b524f464b50ull  // "PKFORK01"
#define PK_PAGE_RESERVED ((u64)-1)
#define PK_PAGE_FLAGS (PAGE_RW | PAGE_U | PAGE_XD | PAGE_G | PAGE_GROW)

struct PkHdr {
  u64 magic;
  u64 npages;  // page entries
  u64 ndata;   // of which carry 4096 bytes
  u64 nfds;
  u64 fdbytes;  // fd records incl. paths, padded to 8
  u64 total;
};

struct PkCpu {
  u8 beg[128];
  u8 xmm[16][16];
  struct MachineFpu fpu;
  struct DescriptorCache seg[8];
  u64 ip;
  u64 sigmask;
  u32 flags;
  u32 mxcsr;
};

struct PkSys {
  i64 brk;
  i64 automap;
  i64 codestart;
  i64 codesize;
  u64 blinksigs;
  struct sigaction_linux hands[64];
  struct rlimit_linux rlim[RLIM_NLIMITS_LINUX];
  sigset_t exec_sigmask;
  u8 dlab;
  u8 brkchanged;
  u8 pad[6];
};

struct PkPage {
  u64 vaddr;
  u64 flags;  // PK_PAGE_FLAGS bits
  u64 slot;   // data page index, or PK_PAGE_RESERVED
};

struct PkFdRec {
  i32 fildes;
  i32 oflags;
  i32 socktype;
  u8 norestart;
  u8 pad[3];
  u32 pathlen;  // bytes of path following (no NUL), record padded to 8
};

struct PkWalk {
  struct PkPage *pages;  // NULL while counting
  u8 *data;
  u64 npages;
  u64 ndata;
};

static i64 Canonical(u64 v) {
  return (i64)(v << 16) >> 16;
}

// Visits every leaf under `table`; with w->pages set it records them.
static void WalkTable(struct System *s, struct PkWalk *w, u64 table, int shift, bool is_cr3, u64 base) {
  int i;
  u8 *page, *slot;
  u64 entry, vaddr, backed;
  page = GetPageAddress(s, table, is_cr3);
  for (i = 0; i < 512; ++i) {
    slot = page + i * 8;
    entry = LoadPte(slot);
    if (!(entry & PAGE_V)) continue;
    vaddr = base | ((u64)i << shift);
    if (shift > 12 && !(entry & PAGE_PS)) {
      WalkTable(s, w, entry, shift - 9, false, vaddr);
      continue;
    }
    // a leaf: backed pages carry data, reserved anonymous ones only their flags
    backed = (entry & (PAGE_HOST | PAGE_MAP | PAGE_MUG)) != 0;
    if (w->pages) {
      struct PkPage *p = &w->pages[w->npages];
      p->vaddr = Canonical(vaddr);
      p->flags = entry & PK_PAGE_FLAGS;
      if (backed) {
        p->slot = w->ndata;
        memcpy(w->data + w->ndata * 4096, GetPageAddress(s, entry & ~(u64)PAGE_RSRV, false), 4096);
      } else {
        p->slot = PK_PAGE_RESERVED;
      }
    }
    w->npages += 1;
    if (backed) w->ndata += 1;
  }
}

static u64 FdBytes(struct System *s, u64 *count) {
  struct Dll *e;
  u64 n = 0, bytes = 0;
  for (e = dll_first(s->fds.list); e; e = dll_next(s->fds.list, e)) {
    struct Fd *fd = FD_CONTAINER(e);
    bytes += ROUNDUP(sizeof(struct PkFdRec) + (fd->path ? strlen(fd->path) : 0), 8);
    n += 1;
  }
  *count = n;
  return bytes;
}

// Serialises the machine; returns a malloc'd buffer (size in *len) or NULL.
static u8 *PkSerialize(struct Machine *m, u64 *len) {
  u8 *buf, *p;
  struct Dll *e;
  struct PkHdr hdr;
  struct PkCpu cpu;
  struct PkSys sys;
  struct PkWalk w;
  struct System *s = m->system;
  memset(&w, 0, sizeof(w));
  memset(&hdr, 0, sizeof(hdr));
  LOCK(&s->mmap_lock);
  WalkTable(s, &w, s->cr3, 39, true, 0);  // count
  hdr.magic = PK_SNAP_MAGIC;
  hdr.npages = w.npages;
  hdr.ndata = w.ndata;
  hdr.fdbytes = FdBytes(s, &hdr.nfds);
  hdr.total = sizeof(hdr) + sizeof(cpu) + sizeof(sys) + hdr.fdbytes + hdr.npages * sizeof(struct PkPage) + hdr.ndata * 4096;
  if (!(buf = (u8 *)malloc(hdr.total))) {
    UNLOCK(&s->mmap_lock);
    return NULL;
  }
  p = buf;
  memcpy(p, &hdr, sizeof(hdr));
  p += sizeof(hdr);
  // cpu
  memset(&cpu, 0, sizeof(cpu));
  memcpy(cpu.beg, m->beg, sizeof(cpu.beg));
  memcpy(cpu.xmm, m->xmm, sizeof(cpu.xmm));
  memcpy(&cpu.fpu, &m->fpu, sizeof(cpu.fpu));
  memcpy(cpu.seg, m->seg, sizeof(cpu.seg));
  cpu.ip = m->ip;
  cpu.sigmask = m->sigmask;
  cpu.flags = m->flags;
  cpu.mxcsr = m->mxcsr;
  memcpy(p, &cpu, sizeof(cpu));
  p += sizeof(cpu);
  // system
  memset(&sys, 0, sizeof(sys));
  sys.brk = s->brk;
  sys.automap = s->automap;
  sys.codestart = s->codestart;
  sys.codesize = s->codesize;
  sys.blinksigs = s->blinksigs;
  memcpy(sys.hands, s->hands, sizeof(sys.hands));
  memcpy(sys.rlim, s->rlim, sizeof(sys.rlim));
  memcpy(&sys.exec_sigmask, &s->exec_sigmask, sizeof(sys.exec_sigmask));
  sys.dlab = s->dlab;
  sys.brkchanged = s->brkchanged;
  memcpy(p, &sys, sizeof(sys));
  p += sizeof(sys);
  // fds
  for (e = dll_first(s->fds.list); e; e = dll_next(s->fds.list, e)) {
    struct Fd *fd = FD_CONTAINER(e);
    struct PkFdRec rec;
    u64 n = fd->path ? strlen(fd->path) : 0;
    memset(&rec, 0, sizeof(rec));
    rec.fildes = fd->fildes;
    rec.oflags = fd->oflags;
    rec.socktype = fd->socktype;
    rec.norestart = fd->norestart;
    rec.pathlen = n;
    memcpy(p, &rec, sizeof(rec));
    if (n) memcpy(p + sizeof(rec), fd->path, n);
    p += ROUNDUP(sizeof(rec) + n, 8);
  }
  // pages
  w.pages = (struct PkPage *)p;
  w.data = p + hdr.npages * sizeof(struct PkPage);
  w.npages = 0;
  w.ndata = 0;
  WalkTable(s, &w, s->cr3, 39, true, 0);
  UNLOCK(&s->mmap_lock);
  unassert(w.npages == hdr.npages && w.ndata == hdr.ndata);
  *len = hdr.total;
  return buf;
}

// fork(): the snapshot goes to the kernel, the child's pid comes back.
int PkForkRemote(struct Machine *m) {
  u8 *buf;
  u64 len;
  int rc;
  if (!(buf = PkSerialize(m, &len))) return enomem();
  if (getenv("PK_FORK_DEBUG")) fprintf(stderr, "blink: fork snapshot %llu bytes\n", (unsigned long long)len);
  rc = js_fork(buf, len);
  if (getenv("PK_FORK_DEBUG")) fprintf(stderr, "blink: fork -> %d\n", rc);
  free(buf);
  if (rc < 0) {
    errno = -rc;
    return -1;
  }
  return rc;
}

// The leaf PTE slot of a page ReserveVirtual just created, or NULL.
static u8 *LeafSlot(struct System *s, i64 vaddr) {
  int level;
  u8 *page, *slot;
  u64 entry = s->cr3;
  bool is_cr3 = true;
  for (level = 39; level >= 12; level -= 9) {
    page = GetPageAddress(s, entry, is_cr3);
    is_cr3 = false;
    slot = page + (((u64)vaddr >> level) & 511) * 8;
    entry = LoadPte(slot);
    if (!(entry & PAGE_V)) return NULL;
    if (level == 12) return slot;
  }
  return NULL;
}

// Restores a snapshot into `m` (a fresh machine); 0 on success.
static int PkRestore(struct Machine *m, const u8 *buf, u64 len) {
  u64 i;
  const u8 *p;
  struct PkHdr hdr;
  struct PkCpu cpu;
  struct PkSys sys;
  const struct PkPage *pages;
  const u8 *data;
  struct System *s = m->system;
  if (len < sizeof(hdr)) {
    fprintf(stderr, "blink: fork snapshot too short (%llu)\n", (unsigned long long)len);
    return -1;
  }
  memcpy(&hdr, buf, sizeof(hdr));
  if (hdr.magic != PK_SNAP_MAGIC || hdr.total != len) {
    fprintf(stderr, "blink: fork snapshot header mismatch (magic %llx, total %llu, len %llu)\n", (unsigned long long)hdr.magic, (unsigned long long)hdr.total, (unsigned long long)len);
    return -1;
  }
  p = buf + sizeof(hdr);
  memcpy(&cpu, p, sizeof(cpu));
  p += sizeof(cpu);
  memcpy(&sys, p, sizeof(sys));
  p += sizeof(sys);
  // fds: the kernel gave the child the same numbers
  for (i = 0; i < hdr.nfds; ++i) {
    struct PkFdRec rec;
    struct Fd *fd;
    memcpy(&rec, p, sizeof(rec));
    if ((fd = AddFd(&s->fds, rec.fildes, rec.oflags))) {
      fd->socktype = rec.socktype;
      fd->norestart = rec.norestart;
      if (rec.pathlen) {
        fd->path = (char *)malloc(rec.pathlen + 1);
        memcpy(fd->path, p + sizeof(rec), rec.pathlen);
        fd->path[rec.pathlen] = 0;
      }
    }
    p += ROUNDUP(sizeof(rec) + rec.pathlen, 8);
  }
  // pages: the root table first (LoadProgram's job in an exec)
  if ((s->cr3 = AllocatePageTable(s)) == (u64)-1) {
    fprintf(stderr, "blink: fork restore: no page table root\n");
    return -1;
  }
  pages = (const struct PkPage *)p;
  data = p + hdr.npages * sizeof(struct PkPage);
  if (getenv("PK_FORK_DEBUG")) fprintf(stderr, "blink: restore: %llu fds, %llu pages (%llu with data)\n", (unsigned long long)hdr.nfds, (unsigned long long)hdr.npages, (unsigned long long)hdr.ndata);
  for (i = 0; i < hdr.npages; ++i) {
    const struct PkPage *pg = &pages[i];
    u8 *slot;
    u64 entry, page;
    i64 got;
    if (getenv("PK_FORK_DEBUG")) fprintf(stderr, "blink: restore page %llx flags %llx %s\n", (unsigned long long)pg->vaddr, (unsigned long long)pg->flags, pg->slot == PK_PAGE_RESERVED ? "reserved" : "data");
    got = ReserveVirtual(s, pg->vaddr, 4096, pg->flags, -1, 0, false, true);
    if (got != pg->vaddr) {
      fprintf(stderr, "blink: fork restore: cannot map page %llx flags %llx (got %llx, errno %d)\n", (unsigned long long)pg->vaddr, (unsigned long long)pg->flags, (unsigned long long)got, errno);
      return -1;
    }
    if (pg->slot == PK_PAGE_RESERVED) continue;
    // fill it in place: allocate the anonymous page the first touch would
    if (!(slot = LeafSlot(s, pg->vaddr))) {
      fprintf(stderr, "blink: fork restore: no leaf for %llx\n", (unsigned long long)pg->vaddr);
      return -1;
    }
    entry = LoadPte(slot);
    if (!(entry & PAGE_RSRV)) {
      fprintf(stderr, "blink: fork restore: leaf %llx not reserved (%llx)\n", (unsigned long long)pg->vaddr, (unsigned long long)entry);
      return -1;
    }
    if ((page = AllocateAnonymousPage(s)) == (u64)-1) {
      fprintf(stderr, "blink: fork restore: out of pages at %llx\n", (unsigned long long)pg->vaddr);
      return -1;
    }
    memcpy(FindHostPage(page), data + pg->slot * 4096, 4096);
    StorePte(slot, (page & (PAGE_TA | PAGE_HOST)) | (entry & ~(u64)(PAGE_TA | PAGE_RSRV)));
    s->memstat.committed += 1;
    s->memstat.reserved -= 1;
    s->rss += 1;
  }
  // system
  s->brk = sys.brk;
  s->automap = sys.automap;
  s->codestart = sys.codestart;
  s->codesize = sys.codesize;
  s->blinksigs = sys.blinksigs;
  memcpy(s->hands, sys.hands, sizeof(s->hands));
  memcpy(s->rlim, sys.rlim, sizeof(s->rlim));
  memcpy(&s->exec_sigmask, &sys.exec_sigmask, sizeof(s->exec_sigmask));
  s->dlab = sys.dlab;
  s->brkchanged = sys.brkchanged;
  // cpu: the child returns 0 from fork
  memcpy(m->beg, cpu.beg, sizeof(m->beg));
  memcpy(m->xmm, cpu.xmm, sizeof(m->xmm));
  memcpy(&m->fpu, &cpu.fpu, sizeof(m->fpu));
  memcpy(m->seg, cpu.seg, sizeof(m->seg));
  m->ip = cpu.ip;
  m->sigmask = cpu.sigmask;
  m->flags = cpu.flags;
  m->mxcsr = cpu.mxcsr;
  Write64(m->ax, 0);
  return 0;
}

// The child's entry: fetch the snapshot from the worker and restore it.
int PkRestoreFork(struct Machine *m) {
  u8 *buf;
  unsigned len;
  int rc;
  len = js_fork_snapshot_size();
  if (!len || !(buf = (u8 *)malloc(len))) {
    fprintf(stderr, "blink: fork restore: no snapshot (%u bytes)\n", len);
    return -1;
  }
  js_fork_snapshot_read(buf, len);
  rc = PkRestore(m, buf, len);
  if (getenv("PK_FORK_DEBUG")) fprintf(stderr, "blink: fork restore %s (%u bytes)\n", rc ? "failed" : "ok", len);
  free(buf);
  return rc;
}

#endif /* __EMSCRIPTEN__ */
