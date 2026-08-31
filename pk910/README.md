# pk910 blink fork

Fork of [jart/blink](https://github.com/jart/blink) (ISC) carrying the changes
that let it run inside the pk910.de terminal: an emscripten JS-library sleep
shim and a copy-on-write `fork()` implementation for the single-instance wasm
build (blink's native fork relies on host `fork()`, which wasm lacks).

- base: tag `pk910-base` (upstream commit f006a4fc)
- work:  branch `pk910-fork`
- upstream update: `git fetch origin && git rebase <new-tag> pk910-fork`,
  then re-export `scripts/blink-emscripten.patch` from the main repo:
  `git -C vendor/blink diff pk910-base..pk910-fork -- blink/ > ../../scripts/blink-emscripten.patch`

`pk910/blink-lib.js` is the emscripten `--js-library` (our own file, no upstream
conflict). The C changes live in `blink/` and are what the patch tracks.
