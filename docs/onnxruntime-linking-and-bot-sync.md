# ONNX Runtime linking & ClashVisionRuntime ↔ BotClashOfClan sync

> **Read this first if you ever touch the `ort` dependency, the `Cargo.toml` of either
> repo, or you hit a YOLO/ONNX error like a timeout, a hang, or
> `Failed to initialize ORT API`.**
>
> This document exists because the way `ort` (the ONNX Runtime Rust bindings) is linked
> is *invisible in the code* but decides whether inference works at all — and getting it
> wrong produces confusing, platform-specific failures (a silent hang on macOS, a hard
> panic on Windows). It also documents how a fix in this library (the Rust inference API)
> reaches the bot that consumes it, because that propagation is **not** automatic.

---

## 1. The two crates and how they connect

| Crate | Repo | Role | Lib name |
|-------|------|------|----------|
| `ClashVisionRuntime` | `Maxime-Cllt/ClashVisionRuntime` | The **Rust inference API**: loads the YOLO model, builds the ONNX Runtime session, runs inference. Owns the `ort` dependency. | `clashvision` |
| `BotClashOfClan` | `Maxime-Cllt/BotClashOfClan` | The **bot** that drives the game. Calls into `clashvision` from a background worker thread (`src/yolo/yolo_session.rs`). | `clashlib` |

The bot depends on this library **as a git dependency**, not a path dependency:

```toml
# BotClashOfClan/Cargo.toml
ClashVisionRuntime = { git = "https://github.com/Maxime-Cllt/ClashVisionRuntime.git", branch = "master", optional = true, version = "^0.8.0" }
```

**Consequence:** the bot does **not** build the local `ClashVisionRuntime/` folder on your
disk. It builds the exact commit of `master` that is **pinned in the bot's `Cargo.lock`**.
This is the single most important fact for the sync workflow in §4.

---

## 2. How `ort` links ONNX Runtime — the thing that actually breaks

`ort` can obtain the native ONNX Runtime in two fundamentally different ways, selected by
**Cargo features**. The choice is not visible in `.rs` files — only in `Cargo.toml`.

### `download-binaries` (what we use) → **static link**

`ort-sys` downloads a prebuilt ONNX Runtime that **matches the `ort` version exactly**
(currently `ort 2.0.0-rc.12` → ONNX Runtime **1.24.2**, which provides **ORT API
version 24**) and **statically links it into our binary**.

Verifiable in `ort-sys`'s build script:

```rust
// ort-sys/build/main.rs
println!("cargo:rustc-link-lib=static=onnxruntime");
```

Result: **no `onnxruntime.dll` / `.dylib` / `.so` is loaded at runtime.** The runtime is
*inside* the executable. There is nothing to ship alongside the binary, and any ONNX
Runtime installed elsewhere on the machine is irrelevant. This is effectively a
"100% static" build for our purposes.

### `load-dynamic` (FORBIDDEN here) → **runtime `dlopen`**

With `load-dynamic`, `ort` does **not** link anything at build time. Instead it
`dlopen`s a library *by name* at first use:

```rust
// ort 2.0.0-rc.12, src/lib.rs (setup_api)
#[cfg(feature = "load-dynamic")]
let base = /* loads ORT_DYLIB_PATH, else "onnxruntime.dll" / "libonnxruntime.so" / "libonnxruntime.dylib" */;
#[cfg(not(feature = "load-dynamic"))]
let base = unsafe { ort_sys::OrtGetApiBase() };   // ← the statically-linked, version-matched symbol

let api = ((*base).GetApi)(ort_sys::ORT_API_VERSION); // ORT_API_VERSION = 24
NonNull::new(api).expect("Failed to initialize ORT API"); // panics if GetApi returns null
```

This is the root cause of every "YOLO doesn't work" incident so far. **Never enable
`load-dynamic` in either crate.** Here is exactly why it fails, per platform:

| Platform | What `load-dynamic` does | Symptom |
|----------|--------------------------|---------|
| **macOS** | Tries to `dlopen` `libonnxruntime.dylib`; the load **fails** (not on the path). `ort`'s error-construction path then re-enters the `ort::api()` `OnceLock` that is *still being initialised* → re-entrant `Once::call_once_force` → **deadlock**. | `SessionBuilder::new()` **hangs forever** at 0% CPU. The worker never becomes ready → warmup times out (180 s) → every `analyse_village()` times out (10 s) → **bot falls back to default placement on every attack**. No error is printed — it just hangs. |
| **Windows 11** | `dlopen("onnxruntime.dll")` **succeeds** but resolves to `C:\Windows\System32\onnxruntime.dll`, which Windows ships for *Windows ML* — an **old version (1.17.1, API ≤ 17)**. `GetApi(24)` returns null. | Hard panic: `The requested API version [24] is not available, only API versions [1, 17] are supported in this build. Current ORT Version is: 1.17.1` → `Failed to initialize ORT API` at `ort/src/lib.rs:211`. The worker disconnects; warmup fails in ~40 ms. |

Same single cause (`load-dynamic`), two different faces. Removing the feature switches both
platforms back to the static, version-matched `OrtGetApiBase()` path, which cannot version-mismatch.

---

## 3. CoreML is Apple-only and must stay gated

On macOS we *do* want the CoreML execution provider (ANE/GPU) — it brings inference to
~7–9 ms after warmup. But the CoreML EP only exists on Apple platforms (the registration
code in `src/session/ort_inference_session.rs` is `#[cfg(target_vendor = "apple")]`).

So the `coreml` `ort` feature is enabled **only for Apple targets**, via a
target-specific dependency. This keeps Windows/Linux on a clean `download-binaries`-only
build:

```toml
# ClashVisionRuntime/Cargo.toml
[dependencies]
# Do NOT add "load-dynamic" here (see §2). download-binaries == static link.
ort = { version = "2.0.0-rc.10", features = ["download-binaries"] }

[target.'cfg(target_vendor = "apple")'.dependencies]
ort = { version = "2.0.0-rc.10", features = ["coreml"] }
```

Cargo *unions* features for the same crate across `[dependencies]` and
`[target.…]` sections, so the effective feature set is:

- **Apple:** `download-binaries` + `coreml`
- **Windows / Linux:** `download-binaries` only

> Note: `coreml` does **not** change which prebuilt binary is downloaded (the download is
> keyed on `cuda`/`webgpu`/`training`/… only, never `coreml`). Gating it is about keeping
> the build minimal and intent-clear on non-Apple platforms; the version 17 vs 24 problem
> is caused by `load-dynamic`, never by `coreml`.

On macOS, the session builder (`OrtInferenceSession::optimized_builder`) registers CoreML
with `MLProgram` + `ComputeUnits::All`. The first inference is a one-off CoreML compile
(~1.8 s) absorbed by the warmup; subsequent inferences are single-digit milliseconds.

---

## 4. The sync workflow — propagating a fix from this library to the bot

Because the bot pins a **commit** of this repo (§1), editing this repo locally changes
nothing for the bot until you do all of the following, **in order**:

```bash
# 1. In ClashVisionRuntime: make the change, commit, PUSH (the bot reads the git remote)
cd ClashVisionRuntime
#   …edit Cargo.toml / src…
git add -p && git commit -m "fix: …"
git push origin master                      # <- the bot can only see pushed commits

# 2. In BotClashOfClan: re-point the lock at the new ClashVisionRuntime commit
cd ../BotClashOfClan
cargo update -p ClashVisionRuntime          # fetches latest master, rewrites Cargo.lock
#   verify it moved:  the line below must show the NEW sha
#   (unix)  grep -A2 'name = "ClashVisionRuntime"' Cargo.lock | grep source
#   (win)   findstr ClashVisionRuntime Cargo.lock

# 3. Commit & push the bot's updated Cargo.lock so other machines get it
git add Cargo.lock && git commit -m "chore: bump ClashVisionRuntime to <sha>"
git push origin master

# 4. On every other machine that runs the bot (e.g. the Windows test box)
git pull                                    # brings the committed Cargo.lock (new sha)
cargo run --release                         # cargo respects the locked sha; first build re-downloads ONNX Runtime
```

### Why each step is required

- **Push the library (1).** `cargo` fetches the git dependency from the **remote**. An
  unpushed local commit is invisible to the bot — even on the same machine.
- **`cargo update -p ClashVisionRuntime` (2).** The bot tracks `branch = "master"` but is
  *pinned to a sha* in `Cargo.lock`. `git pull` / `cargo build` will **not** move to a newer
  `master` commit on their own — only `cargo update -p …` (or deleting the lock entry) does.
- **Commit the bot's `Cargo.lock` (3).** This is what makes step 4 a simple `git pull`.
  Without it, every machine would need to run `cargo update` itself.
- **`git pull` on other machines (4)** is then enough, *provided the working copy is clean*
  (see §6, the "0.82 s, no rebuild" trap).

---

## 5. Verifying you are on the static path (do this when in doubt)

```bash
# load-dynamic must be ABSENT from the whole tree. Output should be empty.
cargo tree -e features | grep -i load-dynamic        # unix
cargo tree -e features | findstr load-dynamic        # windows

# Inspect the resolved ort feature set (must NOT contain load-dynamic):
cargo tree -p ort -f "{p} {f}"
#   apple   → …,coreml,download-binaries,…
#   win/lin → …,download-binaries,…           (no coreml, no load-dynamic)

# Which ClashVisionRuntime commit is the bot actually building?
grep -A2 'name = "ClashVisionRuntime"' Cargo.lock | grep source   # unix
findstr ClashVisionRuntime Cargo.lock                              # windows
```

A correctly linked build, at runtime, **never** prints an ONNX Runtime version line such as
`Current ORT Version is: 1.17.1` — because nothing is `dlopen`ed. If you see such a line,
`load-dynamic` is active somewhere.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| YOLO hangs / every attack falls back to default placement, no error, macOS | `load-dynamic` enabled → dylib not found → re-entrant `OnceLock` deadlock in `ort` | Remove `load-dynamic` (§2), then §4 to get it into the build. |
| `Failed to initialize ORT API`, `requested API version [24] … only [1,17] … Current ORT Version is: 1.17.1`, Windows | `load-dynamic` enabled → `dlopen` picked up System32's old `onnxruntime.dll` (Windows ML, 1.17.1) | Remove `load-dynamic` (§2), then §4. After the fix nothing is `dlopen`ed, so System32 is irrelevant. |
| `git pull` says **"Already up to date"** and `cargo run` finishes in **<1 s with no recompilation**, yet the old error persists | The build is **stale**: your local `Cargo.toml`/`Cargo.lock` were modified during debugging, so `git pull` left them as-is and the lock still pins the old library commit (or re-adds `load-dynamic`). `git pull` never overwrites locally-modified tracked files. | Realign the working copy with the remote, then force a clean rebuild:<br>`git fetch origin`<br>`git reset --hard origin/master`  *(discards local edits)*<br>`cargo clean`<br>`cargo run` |
| Build error: `ort does not provide prebuilt binaries for the target … with feature set …` | An EP feature (`cuda`, `webgpu`, …) has no prebuilt for this target | Drop the unsupported feature, or build ONNX Runtime from source — see <https://ort.pyke.io/setup/linking>. Not relevant to our `download-binaries`/`coreml` setup. |

### Environment variables that can sabotage the static path
These should all be **empty** on a machine that should use the static build:
`ORT_DYLIB_PATH`, `ORT_LIB_LOCATION`, `ORT_LIB_PATH`, `ORT_PREFER_DYNAMIC_LINK`.
(`ORT_STRATEGY=download` in `BotClashOfClan/.cargo/config.toml` is correct and expected — it
is a *build-time* download strategy, not runtime loading.)

---

## 7. Hard rules (the short version)

1. **Never** add `load-dynamic` to `ort` in either repo. `download-binaries` alone = static, version-matched, no DLL to ship.
2. Keep `coreml` **Apple-gated** (`[target.'cfg(target_vendor = "apple")'.dependencies]`).
3. A change here reaches the bot only via: **push library → `cargo update -p ClashVisionRuntime` in the bot → commit & push the bot's `Cargo.lock` → `git pull` elsewhere.**
4. If a `git pull` + run does *not* trigger a rebuild but the error persists, your working copy is dirty — `git reset --hard origin/master && cargo clean` (§6).
5. The `ort` crate version and the downloaded ONNX Runtime version are coupled (`rc.12` ↔ ORT 1.24.2 ↔ API 24). Bumping `ort` changes the required API version; never try to satisfy it with a system DLL.
