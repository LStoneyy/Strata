# aegis: Messungen und Befunde

Host: Ryzen 9 5950X, 14 GiB RAM, RX 5500 XT 8 GB (gfx1012) + RX 9060 XT 16 GB (gfx1200), Ubuntu 26.04, Kernel 7.0.0-42,
ROCm 7.1 (Ubuntu-Pakete unter /usr). Plan: [PLAN.md](../../PLAN.md).

## 2026-11-11: Gate A (Phase 1), `probe/hipblas_probe.cpp`

Build: `hipcc -O2 --offload-arch=gfx1012 --offload-arch=gfx1200 hipblas_probe.cpp -lhipblas -o hipblas_probe`

| Test | 9060 XT (gfx1200) | 5500 XT (gfx1012) |
|---|---|---|
| `hipblasCreate` | ok | **Abbruch:** `rocBLAS error: Cannot read .../rocblas/5.1.0/library/TensileLibrary.dat: Illegal seek for GPU arch : gfx1012` |
| `hipblasGemmEx` F16→F32, Compute F32 (384x256x512) | max. rel. Fehler 8,2e-6 | — |
| `hipblasHgemm` F16 | 7,5e-2 (F16-Akkumulation über K=512, plausibel) | — |

- Ubuntus `librocblas5` 7.1.0 enthält Tensile-Libraries nur für gfx908, gfx90a, gfx942, gfx1030, gfx1100, gfx1101, gfx1151,
  gfx1200 und gfx1201. Upstream rocBLAS 7.1 führt gfx1012 in `TARGET_LIST_ROCM_7.1`, und Tensile bildet es auf `navi14`
  (hip-Fallback-Logik) ab.
- Folge: Ohne eigenes rocBLAS bricht jeder Prozess ab, der auf der 5500 ein hipBLAS-Handle anlegt. Das betrifft auch
  Strata (`src/prefill/gemm.cu`, `cublasCreate`).
- Weg A2: `build-rocblas.sh` baut rocBLAS 7.1.0 aus den Ubuntu-Quellen inkl. Debian-Patches für `gfx1012;gfx1200` nach
  `~/opt/rocblas-7.1.0-gfx1012-gfx1200`. Strata läuft dann mit `LD_LIBRARY_PATH` auf dieses Präfix.
- Nummerierung in HIP: device 0 = RX 5500 XT (PCI 06:00.0), device 1 = RX 9060 XT (PCI 0a:00.0). Main = **1**.

### Gepinnter Host-RAM und GTT
`HIP_VISIBLE_DEVICES=1 ./hipblas_probe pin 2` bzw. `reg 2` (2 GiB, 10 s gehalten):

| | `mem_info_gtt_used` card0/card1 | MemAvailable |
|---|---|---|
| vorher | 13 / 14 MiB | 14021 MiB |
| `hipHostMalloc` 2 GiB | 13 / 14 MiB | 11971 MiB |
| `hipHostRegister` 2 GiB | 13 / 14 MiB | 12005 MiB |

Gepinnter HIP-Speicher (über KFD) zählt **nicht** gegen den amdgpu-GTT bzw. das TTM-Limit, er kostet nur RAM.
`STRATA_RESIDENT_PIN=1` darf also getestet werden, Grenze ist allein der RAM.

### Eigenes rocBLAS (Weg A2), Gate A bestanden
- `build-rocblas.sh`: Build in etwa 10 min (`-j8`, 14:47–14:57), 0 Fehler. Tensile-Libraries `gfx1012 gfx1200` in
  `~/opt/rocblas-7.1.0-gfx1012-gfx1200/lib/rocblas/5.1.0/library`.
- Probe mit `LD_LIBRARY_PATH=~/opt/rocblas-7.1.0-gfx1012-gfx1200/lib`: GemmEx F16→F32 auf **beiden** Karten mit max. rel.
  Fehler 8,2e-6 (identisch), auch mit beiden Karten in einem Prozess. `ldd` zeigt das eigene `librocblas.so.5`.
- Einbindung in Strata ohne Code-Patch: `"lib_dirs"` der Run-Config bzw. von `engine/BUILD.json` setzt `serve/server.py`
  vor `LD_LIBRARY_PATH` der Engine (`serve/server.py:4451`).

## 2026-11-11: Engine-Build (Phase 2), `build-engine.sh`
- Engine 0.1.59.1 (Upstream `6c642df7`), `gfx1012;gfx1200`, Clang 21.1.8, 573 Ninja-Schritte, keine Compile-Fehler.
  `engine/BUILD.json` mit `lib_dirs` auf das eigene rocBLAS. Setup übernimmt die Engine ("engine already built for this PC").
- ctest: **231/234 bestanden**, 19 übersprungen (WMMA/gfx11-spezifisch, fehlende Fixtures). Drei Fehler, keiner davon ein
  Rechenfehler:
  - `ple_parity`: braucht als Fixture eine Q2_0-Modelldatei (`../../Q2_0/...-00002-of-00002.gguf`), die es nicht gibt.
  - `platform_memory_test`: `mlock failed (raise ulimit -l)`. `ulimit -l` ist 8192 KiB. **Für den Dienst
    `LimitMEMLOCK=infinity`**, für interaktive Tests `/etc/security/limits.d/` (sudo).
  - `conversation_copy_streams_test`: Der erste Copy-Stream samt Event kostet **148 MiB VRAM pro Karte** (erwartet:
    < 16 MiB). Identisch auf gfx1012 und gfx1200, also ein Verhalten der ROCm-7.1-Laufzeit und nicht von gfx1012. Die
    1-GiB-Reserve deckt das ab.
- Modell-SHA256 stimmen mit den Pins aus `setup.py` (`CODER_FILES`) überein.

## 2026-11-11: Setup-Check (`setup.py --check --backend hip`)
- Ohne Patch: `GPU 0: AMD Radeon (gfx1012) ... not supported` (`AMD_ARCHS`, `setup.py:4191`), nur die 9060 nutzbar.
- Fork-Patch: Opt-in `STRATA_EXPERIMENTAL_GFX1012=1` (Muster wie `STRATA_EXPERIMENTAL_GFX1150`). Damit werden beide
  Karten "can be used" und der Layer-Split erkannt. Der Standardpfad ohne Variable bleibt unverändert.
- Setup-Einschätzung: Coder IQ1_M "needs ~30 GB of RAM + VRAM: fits", RAM-Warnung (15 GB < 32 GB).
- Python 3.14.4 + `requirements.txt`: Pins installieren ohne Probleme (numpy 2.5.3).
- Modell: `hf download` mit der Revision `5348543e` (Pin aus `setup.py:101`) nach `~/strata-data/models/coder-IQ1_M/`,
  etwa 100 MB/s.

## 2026-11-11: Setup (Modell vorbereiten), `--no-start`
`STRATA_EXPERIMENTAL_GFX1012=1 setup.py --backend hip --family coder --model IQ1_M --data-dir ~/strata-data
--gguf-dir ~/strata-data/models/coder-IQ1_M --gpu-roles 1=main,0=helper --context 32768 --vision no --port 8012
--low-ram on --vram-reserve-mib 1024 --no-start --yes`
- Engine übernommen, `packs/coder-iq1_m/experts.bin` (23 GB) geschrieben, MTP-Draft (`mtp/rt`, 6,5 GB) geladen und gebaut.
- Run-Config `strata-coder-iq1_m.json`: `--mmap-experts`, `--expert-cache auto`, `--spec 4`, `--kv int8`, 32K,
  `lib_dirs` = eigenes rocBLAS. Disk danach: **9,4 GB frei** (96 %).

## 2026-11-11: Smoke-Läufe (Phase 4/5), `smoke.sh` / `smoke.py`, Temperatur 0
| Lauf | VRAM-Cache | Treffer | chat (191 tok) | code (500 tok) | long (4,1K Prompt) | Peak VRAM frei 5500/9060 | Swap max |
|---|---|---|---|---|---|---|---|
| nur 9060, mmap | 9,1 GiB (4802 Slots) | 81–83 % | 13,7 tok/s | **1,6 tok/s** | Abbruch: no progress 60 s (Prompt, Layer 15, Threads in D-State) | – / 1104 MiB | 1014 MB |
| nur 5500, mmap (nur chat) | 0,8 GiB (427 Slots) | 19,8 % | 4,0 tok/s | – | – | 1233 / – | 695 MB |
| Split, mmap | 15,4 GiB (5500: 2991 Slots, 5,2 GiB; 9060: 5120, 10,1 GiB) | 93 % | 10,9 tok/s | **11,2 tok/s** | Abbruch nach 47 Tokens (0,3 tok/s): `verify: timed out at layer 10` | 1134 / 920 | 1369 MB |

- **gfx1012 rechnet korrekt:** Die Greedy-Ausgabe (Reasoning + Antwort, 908 Zeichen) der 5500 allein ist identisch mit der der 9060.
- Split auto: Layer 0–14 auf der 5500 (CUDA0), Layer 15–47 und der Head auf der 9060 (CUDA1). PCIe: 5500 3,1 GB/s
  (pcie_frac 0,08), 9060 14,3 GB/s (pcie_frac 0,39). MTP-Akzeptanz 72–77 %.
- PLE-Probe (n-gram-Tabelle von der SSD, direct): 24.527 Zeilen/s, "fine".
- Ursache der Abbrüche bei langen Prompts im mmap-Modus: Der Prompt berührt fast alle Experten, die file tier (14,3 bzw.
  8,0 GiB) wird über Page-Faults gelesen. Im Split-Lauf meldete der Kernel zusätzlich
  `amdgpu_amdkfd_restore_userptr_worker hogged CPU ... 515 times` und `svm_range_restore_work`: Registrierter
  Host-Speicher, den die GPU liest, wird unter RAM-Druck (Swap steigt, Page-Cache rotiert) verdrängt, und KFD muss die
  Queues ständig wiederherstellen. Gegenmittel: gesperrte RAM-Kopie (`--resident-budget-gib`), weniger RAM-Druck,
  ggf. `--pcie-frac 0`.
- Die Engine schlägt für den Split selbst einen kleineren Prompt-Chunk vor (8192 → 2048, Ring 16: 3,2 statt 54 GB
  SSD-Reads pro 32K-Prompt).

### Split + RAM-Budget (`--resident-budget-gib 6`, statt `--mmap-experts`)
| Lauf | Verteilung VRAM / RAM / SSD | chat | code | long (4.147 Tokens) |
|---|---|---|---|---|
| split-rb6 (Standard, Fused an) | 15,4 / 6,0 / 2,1 GiB | **29,6 tok/s** | **32,4 tok/s** (377 Tokens, `stop`) | Abbruch: Die 9060 rechnete 96 s an Layer 46 ("prompt's GPU work (GPU 1) still running", Host idle, 0 Threads im D-State, kein amdgpu-Fehler im Kernel-Log) |
| rb6 + `STRATA_PF_FUSED=0` | dito | – | – | **ok:** Prompt 109,9 tok/s (37,7 s), danach 14,5 tok/s |
| rb6 + `--prefill 2048` | dito | – | – | HTTP 400: `prefill PLE: native PLE postops launch: no kernel image is available` |

- Die feste RAM-Kopie ist page-locked: "mapped pinned cache complement ready: resident 6.00 GiB, pinned 6.00 GiB in 11 s".
  **Keine userptr/SVM-Meldungen mehr im Kernel-Log**, MemAvailable min. 4,9 GiB, Swap ≤ 1,1 GB.
- MTP-Akzeptanz 74–87 %. Treffer im GPU-Cache 92–94 %, dazu ~1 % per PCIe gelesen.

### Mixed-Arch-Fehler "no kernel image" (gfx1012 + gfx1200)
- Mit `AMD_LOG_LEVEL=1`: `Error: The program ISA amdgcn-amd-amdhsa--gfx1200 is not compatible with the device ISA
  amdgcn-amd-amdhsa--gfx1012:xnack-`, dann `create kernel metadata map using COMgr` / `Cannot create kernels`, direkt nach
  Tensiles Lazy-Loading (`Cannot find the function: Cijk_Alik_Bljk_HSS/BSS_...` = Fallback auf das nächste Code-Objekt).
- Die eigenen Kernel von Strata enthalten beide ISAs (`AMD_LOG_LEVEL=4`, Maske 0x4080: "Using Native code object: gfx1012" bzw.
  "gfx1200"). Ein rocBLAS-Handle über Gerätegrenzen hinweg geht ebenfalls (Probe `cross`). Alle TN-GEMMs in F16/BF16 in
  Prompt-Größen laufen auf beiden Karten (Probe `tn`).
- Ursache: Das rocBLAS ist mit `Tensile_SEPARATE_ARCHITECTURES=ON` (wie Debian) gebaut, also ein Code-Objekt pro
  Architektur. HIP legt ein per `hipModuleLoadData` geladenes Modul auf allen Geräten an, und das gfx1200-Objekt scheitert
  auf der gfx1012. Strata liest den Fehler beim nächsten eigenen Check (`cudaGetLastError`) und bricht die Anfrage ab.
  Er tritt nur bei Prompts mit mehreren Chunks auf (der zweite Chunk trifft die 5500 nach dem Lazy-Load auf der 9060).
  Auf Systemen mit gleichen Karten tritt das nie auf.
- Lösung: rocBLAS mit `Tensile_SEPARATE_ARCHITECTURES=OFF` (`build-rocblas.sh`, Standard jetzt `MERGED=1`).

### Merged-rocBLAS (`Tensile_SEPARATE_ARCHITECTURES=OFF`, `Tensile_LAZY_LIBRARY_LOADING=OFF`)
- Tensile erlaubt Lazy-Loading nur mit getrennten Architekturen. Der Merged-Build lädt die Library deshalb beim Anlegen
  des Handles. Präfix `~/opt/rocblas-7.1.0-gfx1012-gfx1200-merged`, Build etwa 10 min.
- rb6 + `STRATA_PF_FUSED=0` + `--prefill 2048` (mehrere Chunks): **0 ISA-Fehler**, long (4.147 Tokens) Prompt
  **229,8 tok/s** (mit Chunk auto 8192 und altem rocBLAS: 109,9), danach 30,1 tok/s.
- Mit dem Standard-Fused-Pfad (`STRATA_PF_FUSED=1`, gfx12-Default) hängt der Prompt weiterhin ("done up to token 4142",
  no progress 60 s), auch mit Merged-rocBLAS. Das ist ein eigenes Problem des fusionierten int8-Prompt-Kernels auf
  dieser 9060 bzw. mit ROCm 7.1 (Upstream hat ihn auf R9700/ROCm 7.14 gemessen). **Für aegis: `STRATA_PF_FUSED=0`.**

### RAM-Budget-Reihe (Merged-rocBLAS, `STRATA_PF_FUSED=0`, `--prefill 2048`), chat → code → long → long16
| Budget | chat | code | long (4.147) | long16 (6.026) | MemAvailable min. |
|---|---|---|---|---|---|
| 4 GiB | 26,7 tok/s | 28,6 | Hänger (Prompt, Layer 19, Chunk 2) | pp 235,6 / tg 32,3, Treffer 95 % | 7,8 GB |
| 6 GiB | 29,5 | 32,3 | Hänger | pp 258,5 / tg 32,7 | 5,6 GB |
| 8 GiB | 32,4 | 34,6 | pp 245,1 / tg 28,9 | pp 290,2 / tg 32,2 | 3,9 GB |

### Sporadische Hänger im Split (Prompt "GPU work still running" auf beiden Karten, Decode `verify: timed out at layer 5/10` auf der 5500)
- Muster: Ein langer Prompt auf einer frischen Engine läuft durch, nach Decode-Anfragen hängt er meist. Mit abgeschaltetem
  Parken (`--conversation-cache-mib 0`) hing stattdessen die Chat-Anfrage nach dem langen Prompt (verify-Timeout,
  Layer 5) → das Parken ist **nicht** die Ursache.
- Keine P2P-Nutzung (Layer-Split übergibt über Host-RAM; `cudaDeviceEnablePeerAccess` nur mit `--peer-device`). Topologie:
  5500 hinter dem B550-Chipsatz (PCIe 3.0 x4), 9060 an der CPU.
- Pipeline-Windows waren nicht aktiv.
- Ursache laut `docs/AMD_HIP.md` ("Linux: verify timeouts while the kernel reclaims host memory"): Die Queues der GPU
  werden angehalten, während der Kernel per KFD-userptr gepinnte Host-Seiten zurückholt. Passt zum Journal
  (`amdgpu_amdkfd_restore_userptr_worker hogged CPU ... 1027 times`) und zu 14 GiB RAM mit page-cache-gelesenen Experten.
- `HSA_USERPTR_FOR_PAGED_MEM=0`: **gefährlich auf aegis.** GTT stieg beim Laden auf 6,9 GB, Watchdog-Abbruch bei 6800 MiB
  (TTM-Limit 7,5 GiB = die Absturzzone von 2026-10-19/11-03). Danach keine TTM-Warnung, GTT wieder auf 13/14 MiB.
- `GPU_PINNED_MIN_XFER_SIZE=1048576`: chat → code → long ×2 **6/6 durchgelaufen**, GTT unauffällig. Prompt langsamer
  (149 statt ~230 tok/s, Kopien laufen über Staging). Die zweite Runde traf den Prompt-Cache (`cached_tokens` 4142).

### Stresstests: 3 Runden × (chat, code, long, long16), jedes Mal frischer Prompt (`SMOKE_NONCE=1`)
| Config | durchgelaufen | Bemerkung |
|---|---|---|
| rb6 + `GPU_PINNED_MIN_XFER_SIZE=1048576` | 3 Engine-Abbrüche in Runde 1–2 | Workaround reicht nicht, Test abgebrochen |
| rb8 (gepuffert) | 10/12 | Runde 2 long + long16 hingen. Der Prompt-Pfad liest trotzdem 9–10 GB Experten pro Anfrage aus den gemappten Dateien (geliehene Cache-Slots), Swap bis 2,1 GB |
| **rb8 + `STRATA_UNBUFFERED_LOAD=1`** | **12/12** | **0 Major Faults** (Direct-I/O statt mmap/Page-Cache). chat 34–36, code 36–37 tok/s, long pp 208–220 / tg 28–30, long16 pp 258–260 / tg 32–34. Swap max. 131 MB, aber MemAvailable min. **1,27 GB** |
| **rb6 + `STRATA_UNBUFFERED_LOAD=1`** (gewählt) | **12/12** | chat 29–32, code 29–30 tok/s, long pp 200–210 / tg 22–23, long16 pp 256 / tg 28–30. MemAvailable min. **2,57 GB**, Swap 131 MB. Peak frei auf der 9060 nur 929 MiB → `--vram-reserve-later-mib 1408` (danach 1314 MiB frei) |

- Ursache bestätigt: Der Code kennt den Mechanismus (`src/core/expert_source.cpp`, `file_tier_unbuffered`, #1752):
  "ROCm queues are suspended while the kernel reclaims around the process's HMM-tracked arena ... the cached reads churn
  the page cache and reclaim unmaps the arena". Upstream erzwingt die alte Regel nur für gfx906, gfx1201 gilt als
  "not affected". Die gfx1012 mit 14 GiB RAM ist betroffen.
- **Für aegis: `STRATA_UNBUFFERED_LOAD=1`** (die file tier liest per Direct-I/O in die eigenen gepinnten Puffer).

## 2026-11-11: Kontextgröße (`--max-context`, `--kv`), je ein Lauf mit frischen Prompts
KV-Cache nur für die 12 QSA-Layer + den Draft-Layer: ~13,7 KB pro Token mit int8 (q4_0 ~7,5 KB, k8v4 ~10,6 KB). Er liegt
im VRAM und geht vom Experten-Cache ab. Trainierter Kontext 262144. `--kv-grow` braucht eine GPU (hier aus),
KV-Streaming (`--kv-resident`) bräuchte RAM, der fehlt. Needle: `SMOKE_DEEP_TOKENS=100000` (87.9K Tokens aus
`serve/server.py`, Codename bei 40 %).

| Config | KV | Slots (5500 + 9060) | chat / code tok/s | long16 (6K) | 88K-Needle | MemAvailable min |
|---|---|---|---|---|---|---|
| 64K int8 | 0,9 GB | 2925 + 4742 = 7667 | ~27–30 | pp 255 / tg 28–29 | – | 2,57 GB |
| 128K int8 | 1,8 GB | 2793 + 4370 = 7163 | 26 / 20 | pp 213 / tg 21 | **gefunden**, pp 304 / tg 18 | 1,41 GB |
| 256K q4_0 | 2,0 GB | 2728 + 4216 = 6944 | 23 / 18 | pp 203 / tg 19 | **gefunden**, pp 282 / tg 19 | 1,48 GB |
| 256K int8 | 3,6 GB | 2527 + 3621 = 6148 | 19 / 13 | pp 188 / tg 14 | – | 3,00 GB |

- VRAM-Peak bei allen ≥ 1,12 GiB frei (5500) / 1,31 GiB (9060): Der Experten-Cache passt sich an, die Reserve hält.
- Der 88K-Prompt liest ~290 GB Experten von der SSD (jeder 2048er-Chunk streamt die geliehenen Slots), ~5 min.
- **Entscheidung 2026-11-11: 128K mit int8** (Genauigkeit vor 256K/q4_0).
- Stresstest der Dienst-Config (`strata-aegis.json`: rb6, Direct-I/O, Reserven 1280/1408) mit 128K: 2 Runden × (chat, code, long, long16, deep) mit frischen Prompts, **10/10**,
  beide Needles gefunden (pp 309 / 336, tg 20). chat 24–26, code 21–22, long pp 186–193, long16 pp 254–255 tok/s.
  Peak frei 1121 / 1311 MiB, MemAvailable min. 1,41 GB, Swap 129 MB, keine Kernel-Meldungen.

## 2026-11-22: Update auf Upstream 0.2.18 (Engine 0.1.64, eigener Engine-Code ohne ggml/llama.cpp-Kernel)
Gleiche Dienst-Config (128K int8, Direct-I/O, Reserven 1280/1408), Smoke `chat,code,long,long16` × 2 mit frischen Prompts,
alte und neue Engine direkt nacheinander am selben Tag. tok/s, Mittel aus 2 Läufen.

| Engine / Config | chat tg | code tg | long pp / tg | long16 pp / tg | Peak frei 5500 / 9060 | MemAvailable min |
|---|---|---|---|---|---|---|
| alt (Stand 11-11), rb6 | 24,7 | 20,4 | 190 / 17,3 | 233 / 24,7 | 1123 / 1315 MiB | 1,19 GB |
| 0.2.18, rb6, Standard | 22,5 | 17,4 | 188 / 14,7 | 238 / 21,5 | 1379 / 1550 MiB | 0,81 GB |
| 0.2.18, rb6, `STRATA_ROUTE_TAIL_SKIP=7` | 25,9 | 22,3 | 186 / 17,8 | 236 / 24,8 | 1379 / 1550 MiB | 0,92 GB |
| **0.2.18, rb 4.5, RTS 7 (gewählt)** | 24,5 | 19,9 | 172 / 18,0 | 219 / 23,3 | 1381 / 1551 MiB | 2,46 GB |

- 88K-Needle (`SMOKE_DEEP_TOKENS=88000`, 75.327 Tokens): mit rb6 Abbruch durch den Watchdog (MemAvailable 398 MiB, sinkt
  im Prompt stetig); mit **rb 4.5 gefunden** (`BLUE-HERON-7342`), pp 233 / tg 20,8, MemAvailable min. 2,04 GB.
- Die Engine kürzt rb6 ohnehin auf 5,38 GiB ("exceeds available RAM ... minus 4 GiB headroom").
- Laden 30–34 s, keine Hänger, keine Kernel-Meldungen.

