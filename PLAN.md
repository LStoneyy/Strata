# PLAN: Strata auf aegis (Fork LStoneyy/Strata)

Stand: 2026-11-11 · Basis: Upstream `Niko1221/Strata` @ `6c642df7` (Fork noch ohne eigene Commits)

## Stand 2026-11-11 (abends)

| Phase | Status |
|---|---|
| 0 Vorbereitung | erledigt: llama.cpp-Dienste gestoppt und disabled, `~/GGUF` aufgeräumt, ninja + Build-Pakete installiert, memlock unlimited |
| 1 Gate A | erledigt: Ubuntus rocBLAS ohne gfx1012 → eigenes rocBLAS (`deploy/aegis/build-rocblas.sh`, Merged-Build) |
| 2 Engine | erledigt: `deploy/aegis/build-engine.sh`, ctest 231/234 (3 Umgebungsgründe) |
| 3 Python + Modell | erledigt: `.venv` (Python 3.14), Coder IQ1_M (SHA256 ok), `experts.bin`, MTP-Draft |
| 4 Einzelkarten | erledigt: gfx1012 liefert bitgleiche Greedy-Ausgabe wie gfx1200 |
| 5 Split + Ebenen | erledigt: VRAM 15 GiB + RAM 6 GiB + SSD (Direct-I/O), **128K Kontext (int8-KV)**; Stresstests ohne Hänger. Pflicht-Einstellungen: [deploy/aegis/README.md](deploy/aegis/README.md) |
| 6 Dienst | Unit fertig (`deploy/aegis/strata.service`), Installation per sudo offen; ohne API-Key im Netz (Entscheidung 2026-11-11) |
| 7 Clients | später (Entscheidung des Nutzers) |

Abweichungen vom Plan, die sich bewährt haben: ungepufferte SSD-Reads (`STRATA_UNBUFFERED_LOAD=1`) statt mmap, Merged-rocBLAS,
`STRATA_PF_FUSED=0`, `--prefill 2048`. Details und Messungen: [deploy/aegis/FINDINGS.md](deploy/aegis/FINDINGS.md).

## 1. Ziel

aegis wird komplett zur Strata-Maschine. Modell: **Qwen3.8-Flash-Next Coder IQ1_M** (`--family coder --model IQ1_M`).

Die Experten des Modells verteilen sich auf drei Ebenen:

```
            ┌───────────────────────────────────────────────────────────────┐
  heißeste  │ VRAM  RX 9060 XT 16 GB (Main)  +  RX 5500 XT 8 GB (Helper)    │  Layer-Split
     ▲      ├───────────────────────────────────────────────────────────────┤
     │      │ RAM   --resident-budget-gib N   (heiße Experten, die keine     │  der Regler für
     │      │       Karte hält, nach Profil-Rang)                           │  ein RAM-Upgrade
     ▼      ├───────────────────────────────────────────────────────────────┤
  kälteste  │ SSD   Rest der Experten (mmap aus den GGUFs, OS-Cache)        │
            │       + n-gram-Tabelle (Shard 2, 28,8 GB, --ple-io direct)    │
            └───────────────────────────────────────────────────────────────┘
```

**Leitlinie:** Das Setup bleibt gleich. Nach einem RAM-Upgrade wird nur `--resident-budget-gib` erhöht. Ab genug RAM
wechselt man auf `--resident-experts`, dann kommen alle Experten ohne GPU-Platz aus dem RAM. Mehr ändert sich nicht.

Nicht-Ziele (vorerst): DeepSeek V4.1 (braucht 257 GB Disk, die Disk hat 228 GB), Qwen3.8-Flash-Next in voller Größe
(Q2_0 oder größer), Desktop-App/Tauri, Vision (mmproj), Upstream-PRs.

## 2. Ausgangslage aegis (gemessen am 2026-11-11)

| | |
|---|---|
| CPU | Ryzen 9 5950X, 16 Kerne / 32 Threads |
| RAM | **14 GiB** + 4 GiB Swap |
| GPU (Reihenfolge laut rocminfo) | gfx1012 RX 5500 XT 8176 MiB (PCIe 3.0 x4) · gfx1200 RX 9060 XT 16304 MiB (PCIe x16) |
| GTT / TTM | gtt_total je 7677 MiB, `ttm pages_limit` 1965520 Seiten (= 7,5 GiB) |
| SSD | Intenso NVMe 232,9 GB: **seq. 1,4 GB/s**, 64-KiB-Direct-Reads ~1,0 GB/s, **random 4 KiB QD1 ~8,7k IOPS** (114 µs) |
| Disk frei | 54 GB (davon ~120 GB alte GGUFs in `~/GGUF`) |
| OS | Ubuntu 26.04, Kernel 7.0.0-42 (`amdgpu.runpm=0`), Mesa 26.0.3 |
| ROCm | Ubuntu-Pakete 7.1 unter `/usr` (hipcc 7.1.1, libamdhip64 7.1.0, hipBLAS 7.1.0, rocBLAS 7.1.0, rocWMMA, rocPRIM). **Kein** `/opt/rocm` |
| rocBLAS-Tensile-Kernel | gfx1030 gfx1100 gfx1101 gfx1151 **gfx1200** gfx1201 gfx908 gfx90a gfx942. **Kein gfx1012!** |
| Toolchain | cmake 4.2.3, gcc 15.2, Python 3.14.4 (venv ok). **ninja fehlt**, node fehlt (wird nicht gebraucht) |
| sudo | nur interaktiv: der Nutzer führt sudo-Befehle selbst aus |
| Dienste | `llama-server.service` (:8012), `kallilex-server.service` (:8017) |

Lehren aus dem llama.cpp-Betrieb (siehe Memory `aegis-llm-server`): Überlauf der 9060 in den GTT bzw. TTM-Swapout hat
zweimal den amdgpu-Treiber korrumpiert (Reboot nötig). **Pro Karte bleibt unter Volllast ≥ 1 GiB VRAM frei.** Taucht ein
`WARNING ... ttm` in `journalctl -k` auf, wird ein Reboot eingeplant.

## 3. Erkenntnisse aus dem Strata-Code (verifiziert)

### 3.1 Engine und Backends
- Strata hat eine eigene Engine (C++/HIP/CUDA) auf dem mitgelieferten ggml (`third_party/`). Dazu kommen der Python-Server
  `serve/server.py`, das CLI `strata_cli/` und Setup bzw. Download über `setup.py`.
- **Inferenz läuft nicht über Vulkan.** Vulkan nutzt Strata nur für den Bild-Leser. AMD läuft also über HIP.
- `cmake/hip_backend.cmake:12-37`: gfx1200 = `community` (validiert), gfx1012 = `unvalidated` (baut mit WARNING).
  Mehrere Architekturen in einer Liste sind erlaubt (`gfx1012;gfx1200`). Die Liste landet als `STRATA_HIP_ARCHS` im
  Binary, und `src/core/device.cu` prüft beim Start jede Karte dagegen.
- `STRATA_GFX1012_PORTABLE_DOT` (Standard OFF, `hip_backend.cmake:99`): Laut Doku nimmt gfx1012 für den fehlenden
  signed dot4 die SDWA-Sequenz von llama.cpp, mit ON die portable Variante. Fürs erste bleibt der Standard. Die
  ON-Variante wird nur bei Fehlern oder Abweichungen getestet.
- WMMA-Kernel werden nur bei gfx11 gebaut bzw. aktiviert, sind also für uns irrelevant.
- `docs/OLDER_GPUS.md:310-320`: **"Setup does not build these. Build by hand and run `serve/server.py` with a config."**
  Für gfx1012 gibt es einen Community-Bericht: RX 5500 XT 8 GB, IQ3_S, Decode 15,3 tok/s. Eine RX 5700 XT 8 GB mit Coder
  IQ1_M und 32K Kontext schaffte 18–23 tok/s Decode (das System hatte aber RAM für alle Experten).
- **hipBLAS ist Pflicht** (`hip_backend.cmake:62`). Die Prompt-GEMMs (`src/prefill/gemm.cu`, `cublasCreate` →
  hipBLAS → rocBLAS) laufen auf der Karte, die den Layer hält. **Ohne rocBLAS-Kernel für gfx1012 ist das der größte
  Blocker** (→ Gate A).
- Ausweich-Option ohne BLAS für dichte Projektionen: `STRATA_DENSE_MMQ=1` (int8-MMQ, Build mit `-DSTRATA_PREFILL_MMQ=ON`;
  Greedy-Text kann nach einigen Dutzend Tokens leicht abweichen, `docs/AMD_HIP.md:272`). Es ist nicht verifiziert, dass
  danach gar kein rocBLAS-Aufruf mehr übrig bleibt.

### 3.2 Multi-GPU
- Layer-Split (Pipeline). Der Hand-off läuft über Host-RAM, Peer-to-Peer ist nicht nötig (`docs/MULTI_GPU.md`).
- **Karten mit ≤ 8 GB starten standardmäßig auf Off** (`strata_control/gpu_roles.py:22`, `OFF_AT_OR_BELOW_GB = 8.0`).
  Man schaltet sie über explizite Rollen ein: `--gpu-roles 1=main,0=helper` (Source-Setup) oder `"gpu": [..]` in der
  Run-Config.
- HIP-Nummerierung auf aegis (vermutlich wie rocminfo): **0 = 5500 XT, 1 = 9060 XT**, im Engine-Log zu prüfen. Main muss die
  9060 sein, denn sie hält die letzte Stage, den Output-Head und den MTP-Draft.
- `strata_control/split_engine.py` (der "all-cards split", eine Art Tensor-Parallel) ist für uns **nicht** relevant: Er
  verlangt gleiche Karten und alle Experten im VRAM. Es bleibt beim normalen Layer-Split.
- Feste VRAM-Reserve: `--vram-reserve-mib N` sowie `--vram-reserve-later-mib N` für die späteren Karten.

### 3.3 Speicher-Ebenen (Low-RAM-Modi, `docs/DETAILS.md:242-275`)
| Modus | Flag | Verhalten |
|---|---|---|
| Mapped | `--mmap-experts` | Experten werden direkt aus den GGUFs gemappt und über den OS-Cache von der SSD gelesen |
| Resident | `--resident-experts` | Alle Experten, die keine Karte hält, werden beim Start in den RAM kopiert. Danach liest nichts mehr von der SSD |
| **RAM-Budget** | `--resident-budget-gib N` | N GiB der laut Profil heißesten Nicht-GPU-Experten liegen im RAM, der Rest kommt von der SSD. Impliziert `--mmap-experts`, 4 GiB Headroom (`STRATA_RESIDENT_HEADROOM_GIB`) |

- Auf einem Layer-Split läuft Resident ab Engine 0.1.40 (`setup.py:6555` `RESIDENT_SPLIT_ENGINE`, `split_mmap()`
  `setup.py:2289`). Jede Karte cacht die heißesten Experten **ihrer** Layer, und die RAM-Kopie lässt weg, was eine Karte
  schon hält. **Ob `--resident-budget-gib` auf dem Split genauso greift, ist nicht verifiziert** → Phase 5.
- Die RAM-Kopie ist standardmäßig page-locked. `STRATA_RESIDENT_PIN=0` hält sie stattdessen in normalem, auslagerbarem
  Speicher. Dann übernimmt die GPU auch keine Fehltreffer über PCIe (`--pcie-frac 0`). **Unklar ist, ob gepinnter Host-RAM
  auf aegis gegen das GTT/TTM-Limit von 7,5 GiB zählt.** Deshalb starten wir konservativ mit `STRATA_RESIDENT_PIN=0`.
- Automatik in `setup.py:6553-6568`: Low-RAM greift, wenn `RAM < arena_gb + 10` (Coder: < 33,4 GB). Eine harte
  RAM-Untergrenze, die den Start mit 14 GiB verweigert, gibt es nicht.
- n-gram-Tabelle (Shard 2): wird per `--ple-io direct` ungepuffert gelesen, ein paar Zeilen pro Token. Beim Start prüft
  Strata die Lesegeschwindigkeit und warnt unter 15k Zeilen/s. Ein QD1-Wert von 8,7k IOPS liegt darunter; wie viel
  parallele Reads bringen, zeigt erst der Start-Probe.

### 3.4 Modell Coder IQ1_M (Hugging Face `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`)
| Datei | Größe |
|---|---|
| `IQ1_M/...-IQ1_M-00001-of-00002.gguf` (Dense + Experten) | 29,61 GB |
| `IQ1_M/...-IQ1_M-00002-of-00002.gguf` (n-gram/Engram-Tabelle) | 28,80 GB |
| MTP-Draft-Layer (lädt Setup beim ersten Start) | ~1 GB (`MTP_VRAM_GB = 1.0`) |
| `mmproj-...-BF16.gguf` (Vision) | 0,91 GB, **nicht** laden |

`setup.py:449`: `download_gb 58.4, ram_gb 32, arena_gb 23.4`. Die Experten (256 von 512 je Layer) belegen also 23,4 GiB.

### 3.5 Betrieb
- Server: OpenAI- und Anthropic-kompatible API (`/v1/...`), `--access lan` bindet auf 0.0.0.0 und erzeugt einen API-Key,
  dazu `--port`.
- Run-Config: `strata-<modell>.json` (args, env, gpu, layer_split usw.). Wird für den Headless-Start als systemd-Unit
  benutzt.
- Setup und Build aus dem Source-Checkout gehen davon aus, dass ROCm unter `/opt/rocm` liegt (`setup.py:5220`
  `ROCM_PATH`). Ubuntu legt die Bibliotheken aber unter `/usr/lib/x86_64-linux-gnu` ab. Den Build machen wir deshalb
  **von Hand mit cmake** und lassen ihn nicht über `setup.py` laufen.

## 4. Speicherbudget (Schätzung, wird in Phase 5 gemessen)

| Ebene | Kapazität | davon für Experten | Anteil an 23,4 GiB |
|---|---|---|---|
| 9060 XT (Main) | 16304 MiB | − 1 GiB Reserve − Dense-Anteil − Output-Head − KV − MTP-Draft (~1 GiB) ≈ **8–9 GiB** | ~36 % |
| 5500 XT (Helper) | 8176 MiB | − 1 GiB Reserve − Dense-Anteil − Puffer ≈ **3,5–4,5 GiB** | ~17 % |
| RAM | 14 GiB | − OS/Server/Engine-Puffer (~4–5 GiB) − 4 GiB Headroom ≈ **4–5 GiB** (`--resident-budget-gib`) | ~19 % |
| SSD | — | Rest ≈ **6–7 GiB** (kälteste Experten) + n-gram-Tabelle | ~28 % |

Weil die Experten nach Häufigkeit verteilt werden, landen die selten gerouteten auf der SSD, und die Trefferquote in
VRAM und RAM liegt deutlich über 72 %. Der Engine-Log zeigt die echten Werte (`resident RAM: ... blob reads from the file`,
Hit-Rate).

Kontext: Wir starten mit **32K** und `--kv k8v4` bzw. int8. Ziel sind **64K–128K**, sofern die Reserve von 1 GiB hält. KV
kostet laut Doku etwa 1,7 GB bei 128K (int8), und der Platz geht direkt vom Experten-Cache ab. `--kv-streaming` (KV im RAM)
kommt bei 14 GiB **nicht** infrage.

## 5. Phasen

Jede Phase endet mit einem **Gate**: einem messbaren Kriterium, das erfüllt sein muss, bevor die nächste Phase beginnt.
sudo-Schritte führt der Nutzer aus (`ssh -t aegis sudo ...`).

### Phase 0: Vorbereitung
1. Im Fork einen Branch `aegis` anlegen (`main` bleibt ein sauberer Upstream-Spiegel). Alle aegis-spezifischen Dateien
   kommen nach `deploy/aegis/`.
2. **Nutzer (sudo):** llama.cpp-Dienste stoppen, Units bleiben installiert (Rollback):
   `ssh -t aegis 'sudo systemctl stop llama-server kallilex-server'`
3. **Nutzer (sudo):** `ssh -t aegis 'sudo apt install ninja-build'`. Ob weitere Pakete fehlen, zeigt der cmake-Configure
   in Phase 2. Kandidaten sind `libhipblaslt-dev` (optional), `librocblas-dev` (vorhanden) und `rocm-cmake`.
4. **Disk:** ~62 GB werden gebraucht (58,4 Modell + ~1 MTP + Build ~2–3 GB). Frei sind 54 GB, es müssen also alte GGUFs
   weg oder ausgelagert werden (Entscheidung des Nutzers, siehe §8).
5. Vorher-Zustand sichern: `systemctl cat llama-server kallilex-server`, Presets und Startskripte bleiben unverändert in
   `~/GGUF`.

**Gate 0:** Dienste gestoppt, beide Karten mit ~0 VRAM belegt (`/sys/class/drm/card*/device/mem_info_vram_used`), ≥ 70 GB frei.

### Phase 1: Gate A, rocBLAS/hipBLAS auf gfx1012 (größtes Risiko zuerst)
Ein Mini-Programm (`deploy/aegis/probe/hipblas_probe.cpp`) erzeugt pro Karte (`HIP_VISIBLE_DEVICES=0` bzw. `=1`) ein
hipBLAS-Handle und führt `hipblasHgemm` und `hipblasGemmEx` (F16→F32) aus. Das Ergebnis wird gegen die CPU geprüft.

| Ergebnis auf gfx1012 | Weg |
|---|---|
| A1: läuft (z. B. Fallback-Kernel) | weiter zu Phase 2 |
| A2: Abbruch "Cannot read TensileLibrary ... gfx1012" | **rocBLAS 7.1.0 (gleicher Tag wie das Paket) für `gfx1012` selbst bauen**, nur die Tensile-Library. Ziel: `/opt/strata-rocblas/library`, darin die eigene gfx1012-Library plus Kopien der System-Dateien für gfx1200. Gesetzt wird `ROCBLAS_TENSILE_LIBPATH`. Der Build läuft mit `nice` und begrenzten Jobs (`-j4`, 14 GiB RAM!) und dauert vermutlich Stunden |
| A3: A2 klappt nicht | `-DSTRATA_PREFILL_MMQ=ON` + `STRATA_DENSE_MMQ=1` testen (dichte Prompt-Projektionen ohne BLAS). Wenn auch das nicht geht: **5500 XT aus**, die 9060 läuft allein und es landet entsprechend mehr auf RAM und SSD (Rückfall-Topologie) |

Zusätzlich prüfen: `hipDeviceProp` bzw. `rocminfo` für die Nummerierung, `hipMemGetInfo` pro Karte, `hipHostMalloc` und
`hipHostRegister` mit 1–2 GiB, dabei GTT (`mem_info_gtt_used`) beobachten. Das klärt, ob gepinnter Host-RAM gegen das
GTT-Limit zählt.

**Gate A:** GEMM korrekt auf beiden Karten (oder Rückfall-Topologie entschieden). Gepinnter Speicher und GTT-Verhalten
sind dokumentiert.

> **Ergebnis 2026-11-11** ([FINDINGS](deploy/aegis/FINDINGS.md)): gfx1200 ok. gfx1012 bricht in `hipblasCreate` ab
> (keine Tensile-Library), also **Weg A2**: `deploy/aegis/build-rocblas.sh`. Gepinnter HIP-Speicher zählt nicht gegen
> den GTT. HIP device 0 = 5500, 1 = 9060.

### Phase 2: Engine bauen (`deploy/aegis/build-engine.sh`)
Nach dem Muster von `~/llama.cpp.*`: ein Ordner pro Build mit `BUILD-INFO.txt`, gebaut mit `nice`.

```sh
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES="gfx1012;gfx1200" \
  -DCMAKE_HIP_COMPILER=/usr/bin/clang++   # bzw. der clang aus dem ROCm-Paket, im Configure prüfen
  -DCMAKE_PREFIX_PATH=/usr \
  -DSTRATA_BUILD_TESTS=ON
  # bei Weg A3 zusätzlich: -DSTRATA_PREFILL_MMQ=ON
nice cmake --build build-hip -j8   # 14 GiB RAM: bei OOM auf -j4
```

- Danach `ctest` auf beiden Karten, vor allem die HIP-Prefill-Tests (`hip_prefill_*`, Exitcode 77 = übersprungen).
- Die Engine registrieren: dorthin legen, wo `strata launch` bzw. `serve/server.py` sie sucht (`ROOT/engine/` +
  `BUILD.json`, `setup.py:5314-5362`, `strata_control/engine_component.py:191`). Den genauen Ablauf ermitteln wir in
  dieser Phase.
- Mögliche kleine Patches im Fork, nur wenn nötig und einzeln committet:
  - ROCm-Erkennung für das Ubuntu-Multiarch-Layout (`setup.py:5215-5268`).
  - gfx1012 in den Architektur-Tabellen von `setup.py` (`AMD_ARCHS`, `rocm_index`), falls Setup sonst den Start oder den
    Download verweigert.

**Gate 2:** Engine gebaut, ctest grün (bzw. Ausnahmen begründet), Engine-Selbsttest erkennt beide Karten.

### Phase 3: Python-Umgebung und Modell
1. `.venv` im Checkout auf aegis, Abhängigkeiten aus `requirements.txt`. Python 3.14 auf Pins prüfen.
2. Modell laden: bevorzugt über Setup (`./setup.sh --family coder --model IQ1_M ...`), damit Struktur, MTP-Draft und
   Profil stimmen. Scheitert Setup an gfx1012 oder dem ROCm-Pfad, laden wir die Dateien per `hf download` und übernehmen sie
   mit `strata models import <ordner> --plan`. Vision bleibt aus (kein mmproj).
3. Ablage: `~/strata-models/` auf der NVMe.

**Gate 3:** `strata launch --plan` (bzw. die Setup-Planausgabe) zeigt Coder IQ1_M, beide Karten und den Low-RAM-Modus,
ohne Fehler.

### Phase 4: Einzelkarten-Bring-up (Baseline)
1. **Nur 9060** (`--gpu 1`), `--mmap-experts`, 32K, `--vram-reserve-mib 1024`: Chat, eine Code-Aufgabe, ein kurzer
   Benchmark (pp2048/tg128 im Stil von `tools/strata_bench.py`).
2. **Nur 5500** (`--gpu 0`), gleiche Einstellungen: prüft die Korrektheit von gfx1012 (dot4-Pfad, rocBLAS). Die Antworten
   der beiden Karten werden bei Temperatur 0 verglichen. Bei Abweichungen zuerst `-DSTRATA_GFX1012_PORTABLE_DOT=ON` testen.
3. Während jedes Laufs läuft ein Watchdog. Vorlage ist `~/llama.cpp.vulkan-20261105/smoke/watchdog.sh` (die Kopien in
   `~/GGUF/*-tune` sind gelöscht), als `deploy/aegis/watchdog.sh`: VRAM-Peak und GTT pro Karte loggen und bei fallendem VRAM
   bzw. steigendem GTT per PID abbrechen.

**Gate 4:** Beide Karten liefern allein korrekte Antworten. VRAM-Peak ≤ (Gesamt − 1 GiB), kein GTT-Anstieg, keine
ttm-WARNING im Kernel-Log.

### Phase 5: Split + drei Ebenen (Zielkonfiguration)
Grundkonfiguration:
```
--gpu-roles 1=main,0=helper     (bzw. "gpu": [0, 1] + Rollen in der Run-Config)
--layer-split auto
--resident-budget-gib N
--vram-reserve-mib 1024 --vram-reserve-later-mib 1024
--kv k8v4   (Alternative int8 vergleichen)
--max-context 32768 → 65536 → 131072
STRATA_RESIDENT_PIN=0   (Startwert)
```
Messreihen (kurz: pp2048/tg128, plus ein 16K- und ein 32K-Prompt; kein Needle über den vollen Kontext ohne Absprache):
1. `N = 0` (reines mmap), danach 2, 4, 6 GiB. Zu messen: Decode, Prompt, verfügbarer RAM (`MemAvailable`), Swap-Nutzung,
   SSD-Reads pro Token. Gewählt wird das größte N, bei dem unter Last ≥ 1,5 GiB `MemAvailable` und kein Swap-Thrashing
   bleiben.
2. Prüfen, dass `--resident-budget-gib` auf dem Split wirklich wirkt (Log `resident RAM: ...`). Fällt es auf reines mmap
   zurück, testen wir `--resident-experts` bzw. die automatische RAM-Kopie der Engine, sonst wird das Verhalten im Fork
   gepatcht (eigener, begründeter Commit).
3. `STRATA_RESIDENT_PIN=1` gegen `0`, aber nur, wenn Phase 1 gezeigt hat, dass gepinnter Speicher **nicht** gegen das
   GTT-Limit läuft. Sonst bleibt es bei 0.
4. Layer-Split `auto` gegen einen manuellen Split (`--layer-split <n>`), falls `auto` die 5500 über- oder unterlädt.
5. Kontextgröße: so groß wie möglich bei 1 GiB Reserve pro Karte. Der Gewinn an Kontext wird gegen den Verlust an
   Experten-Cache (Decode-Tempo) abgewogen.
6. `--ple-io direct`: Start-Probe ansehen (Zeilen/s). Bei Warnung und Stalls `mmap` testen.
7. Alle Ergebnisse kommen nach `deploy/aegis/FINDINGS.md`, mit Datum, Engine-Commit und Messbedingungen.

**Gate 5:** Die Split-Konfiguration ist stabil (mehrere Agent-Sessions ohne Absturz, kein GTT-Spill, keine ttm-WARNING).
Decode ist besser als die 9060 allein. Andernfalls dokumentieren wir, warum die Einzelkarte gewinnt, und entscheiden mit dem
Nutzer.

### Phase 6: Dienstbetrieb
1. `deploy/aegis/strata.service` (System-Unit, analog zu `llama-server.service`): startet `serve/server.py` bzw.
   `strata launch --yes --no-tui` mit der festen Run-Config. `Restart=on-failure`, `Nice`, Umgebung aus
   `deploy/aegis/strata.env`.
2. Port: **8012 übernehmen**, damit Clients nur Modellnamen und Key anpassen müssen. Die Alternative ist der
   Strata-Standard 8080 (Entscheidung des Nutzers). `--access lan` mit API-Key.
3. **Nutzer (sudo):** Unit installieren, `llama-server` und `kallilex-server` **disablen** (nicht löschen), Strata
   enablen.
4. Smoke nach einem Reboot: Der Dienst startet von selbst, das erste Laden der Experten in RAM und Cache ist gemessen.

**Gate 6:** Strata überlebt einen Reboot und beantwortet Anfragen über das LAN.

### Phase 7: Clients umstellen (nach Absprache)
- Mac: `~/.pi/agent/models.json` (+ `settings.json`, `agents/*.md`), `~/.config/opencode/opencode.jsonc`
- futro: Hermes `~/.hermes/config.yaml` (`custom_providers`, `context_length` fest auf den Strata-Kontext setzen)
- Werkzeug: `strata connect` zeigt die passenden Einstellungen.

### Phase 8: Fork-Pflege
- Branch `aegis`, Upstream-Updates per `git fetch upstream && git merge upstream/main` und anschließendem Neubau
  (Phase 2 + Smoke aus Phase 4/5). Upstream wechselt mehrmals täglich.
- Inhalt des Forks: `deploy/aegis/` (build-engine.sh, probe/, strata.env, strata.service, run-config, FINDINGS.md,
  README.md) plus nur die nötigen Patches mit Begründung im Commit.
- Engine-Builds als datierte Ordner (`~/strata-engine-YYYYMMDD/`), Rollback über einen Symlink.

## 6. RAM-Upgrade-Pfad (Setup bleibt gleich)

| RAM | Änderung |
|---|---|
| 14 GiB (jetzt) | `--resident-budget-gib` ≈ 4–5 (Phase 5) |
| 32 GiB | Budget auf ~20 GiB, praktisch alle Nicht-GPU-Experten liegen im RAM. Alternativ `--resident-experts` |
| ≥ 64 GiB | `--resident-experts`, optional `--ple-io ram` (n-gram-Tabelle 28,8 GB im RAM, sobald RAM ≥ Bedarf + Tabelle + 16 GB). Ab hier wird auch Qwen3.8-Flash-Next Q2_0/IQ2_XS (alle 512 Experten) realistisch, braucht aber +66–76 GB Disk |

Achtung bei RAM > 32 GB: Mit `STRATA_RESIDENT_PIN=1` ist die RAM-Kopie page-locked. Das GTT- bzw. KFD-Verhalten aus
Phase 1 gilt dann erst recht.

## 7. Risiken

| Risiko | Wirkung | Gegenmaßnahme |
|---|---|---|
| rocBLAS ohne gfx1012-Kernel | 5500 XT kann keine Prompts rechnen | Gate A: eigener Tensile-Build oder MMQ oder Rückfall auf die Einzelkarte |
| gfx1012 "unvalidated" in Strata | falsche Ausgaben oder Abstürze | Phase 4: Temperatur-0-Vergleich beider Karten, Option `PORTABLE_DOT` |
| amdgpu-TTM/GTT-Korruption | Kernel hängt, Reboot nötig | 1 GiB VRAM-Reserve pro Karte, Watchdog, `STRATA_RESIDENT_PIN=0`, ttm-WARNING überwachen |
| 14 GiB RAM | Page-Cache-Druck, Swap-Thrashing, OOM bei Bauen oder Laden | konservatives Budget, `nice -j8/-j4` beim Bauen, `MemAvailable` überwachen |
| schwache SSD (Intenso, 8,7k IOPS QD1) | langsames Decode bei Fehltreffern, PLE-Stalls | heißeste Experten in VRAM und RAM, PLE-Probe ansehen, später mehr RAM |
| 5500 hängt nur an PCIe 3.0 x4 | langsamerer Hand-off und langsamere Prompts | Layer-Split manuell justieren, im Zweifel weniger Layer auf die 5500 |
| Python 3.14 (sehr neu) | Abhängigkeiten ohne Wheels | venv prüfen, sonst `uv` mit Python 3.12 (`STRATA_UV=1`) |
| Upstream ändert sich sehr schnell | Merge-Konflikte, Regressionen | eigene Änderungen isoliert in `deploy/aegis/`, Patches minimal |

## 8. Entscheidungen (Nutzer, 2026-11-11)
1. **Disk:** Erledigt. Gelöscht wurden Qwen3.6-35B-A3B und flint (samt ihren Preset-Abschnitten), alle alten Presets und
   Backups sowie die Tuning- und Test-Ordner. `~/GGUF` enthält nur noch Modelle, mmprojs, MTP-Drafts, Templates,
   `presets-5500-20261105.ini`, `hf-venv` und die zwei Startskripte der Dienste. Frei sind jetzt **98 GB**.
2. **Port:** **8012** (wie bisher llama-server).
3. **kallilex:** vorerst aus.
4. **Dienste:** Der Nutzer disabled `llama-server` und `kallilex-server`. Strata bekommt eine eigene systemd-Unit, und
   umgeschaltet wird über enable/disable (Strata **oder** llama.cpp, nie beide gleichzeitig: beide wollen Port 8012 und
   den VRAM).
5. **Clients (Phase 7):** erst später.

## 9. Rollback
`sudo systemctl disable --now strata && sudo systemctl enable --now llama-server kallilex-server`. Die llama.cpp-Builds,
Presets und Startskripte bleiben unverändert. Gelöschte GGUFs lassen sich nur über einen neuen Download oder die
Auslagerung zurückholen.
