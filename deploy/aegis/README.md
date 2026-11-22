# Strata auf aegis

Ryzen 9 5950X, 14 GiB RAM, **RX 9060 XT 16 GB (gfx1200, Main)** + **RX 5500 XT 8 GB (gfx1012, Helper)**, Ubuntu 26.04,
ROCm 7.1 aus den Ubuntu-Paketen. Modell: Qwen3.8-Flash-Next **Coder IQ1_M**. Plan: [../../PLAN.md](../../PLAN.md),
Messungen und Befunde: [FINDINGS.md](FINDINGS.md).

## Aufbau

| Ebene | Inhalt |
|---|---|
| VRAM | Layer 0–14 auf der 5500, Layer 15–47 + Head + MTP-Draft auf der 9060; Experten-Cache ~5 + ~9,6 GiB |
| RAM | `--resident-budget-gib 4.5`: die heißesten Experten, die keine Karte hält, page-locked (bis 0.2.13: 6) |
| SSD | Rest der Experten (`packs/coder-iq1_m/experts.bin`, Direct-I/O) + n-gram-Tabelle (Shard 2, `--ple-io direct`) |

Ergebnis (2026-11-11, frische Prompts, **128K Kontext**, int8-KV): chat/code ~22–26 tok/s, 4K-Prompt ~190 tok/s,
6K-Prompt ~255 tok/s, ein 88K-Prompt ~310–335 tok/s (4,5–5 min) mit 20 tok/s danach. VRAM im Peak ≥ 1,1 GiB frei pro
Karte, GTT unverändert, MemAvailable ≥ 1,4 GB.

## Dateien

| Datei | Zweck |
|---|---|
| `build-rocblas.sh` | rocBLAS 7.1.0 aus den Ubuntu-Quellen für `gfx1012;gfx1200` mit **zusammengeführten** Tensile-Code-Objekten → `~/opt/rocblas-7.1.0-gfx1012-gfx1200-merged` |
| `build-engine.sh` | Engine-Build von Hand (`build-hip/`, `engine/strata` + `engine/BUILD.json` mit `lib_dirs`) inkl. ctest |
| `strata-aegis.json` | Run-Config des Dienstes (die maßgebliche Konfiguration) |
| `strata.service` | systemd-Unit (Port 8012, `LimitMEMLOCK=infinity`, `Conflicts=` mit den llama.cpp-Diensten) |
| `smoke.sh`, `smoke.py`, `watchdog.sh`, `variant.py` | Testläufe mit VRAM/GTT/RAM-Watchdog, Config-Varianten |
| `probe/hipblas_probe.cpp` | Gate-A-Probe: hipBLAS pro Karte, TN-GEMMs, Handle über Gerätegrenzen, gepinnter Speicher vs. GTT |

## Pflicht-Einstellungen (warum)

| Einstellung | Grund |
|---|---|
| eigenes rocBLAS über `"lib_dirs"` | Ubuntus rocBLAS hat keine gfx1012-Kernel: `hipblasCreate` bricht auf der 5500 ab |
| rocBLAS `Tensile_SEPARATE_ARCHITECTURES=OFF` | sonst scheitert ein gfx1200-Code-Objekt auf der gfx1012 und hinterlässt "no kernel image" (Prompts mit mehreren Chunks) |
| `STRATA_UNBUFFERED_LOAD=1` | gepufferte Reads (mmap/Page-Cache) + 14 GiB RAM: Der Kernel holt per KFD-userptr gepinnte Seiten zurück, die GPU-Queues stehen → `verify: timed out` / "no progress for 60 s" |
| `STRATA_PF_FUSED=0` | der fusionierte int8-Prompt-Kernel (gfx12-Default) hängt hier im Prompt |
| `--max-context 131072`, `--kv int8` | Entscheidung 2026-11-11: 128K mit int8 statt 256K mit q4_0 (Genauigkeit). KV ~13,7 KB/Token (1,8 GB) geht vom Experten-Cache ab; 256K int8 kostet 20 % der Slots ([FINDINGS.md](FINDINGS.md)) |
| `--prefill 2048` | kleinere Prompt-Chunks: weniger SSD-Reads und schneller (Engine-Empfehlung 2048–2816) |
| `STRATA_ROUTE_TAIL_SKIP=7` (ab 0.2.17) | 0.2.17 stellt den Route-Tail-Skip auf RX 9000 standardmäßig ab; ohne ihn ist tg hier 10–15 % langsamer. 7 ist das bisherige Verhalten (Antworten können minimal von „aus“ abweichen) |
| `--resident-budget-gib 4.5` (ab 0.2.18) | mit 6 GiB fiel MemAvailable im 75K-Prompt auf 398 MiB (Watchdog-Abbruch); mit 4,5 bleiben ≥ 2 GB |
| `--vram-reserve-mib 1280`, `--vram-reserve-later-mib 1408` | ≥ 1 GiB VRAM pro Karte frei auch im Prompt-Peak |
| `--gpu-roles 1=main,0=helper` (`"gpu": [0,1]`) | HIP-Gerät 1 = 9060 (Main), 0 = 5500; Setup schaltet 8-GB-Karten sonst aus |
| `STRATA_EXPERIMENTAL_GFX1012=1` (nur für `setup.py`) | Fork-Patch: Setup akzeptiert die 5500 als Karte |
| `"host": "0.0.0.0"`, `"api_key": "none"`, `"allowed_hosts"` | Entscheidung 2026-11-11: ohne Key wie früher llama-server. IPs passieren immer, Namen nur aus der Liste (`aegis`, `aegis.local`); über Tailscale die IP nehmen oder den MagicDNS-Namen lokal ergänzen (nicht committen) |

**Nicht** setzen: `HSA_USERPTR_FOR_PAGED_MEM=0` (GTT stieg beim Laden auf 6,9 GB, Absturzzone des amdgpu-TTM).

## Betrieb

```sh
# Dienst
sudo systemctl status strata
journalctl -u strata -f                       # Server-Log
tail -f ~/code/Strata-0.2.18/terminal/strata-coder-iq1_m.log  # Engine-Log (ab 0.2.16: unter terminal/)

# Umschalten auf llama.cpp und zurück (Conflicts= stoppt jeweils den anderen)
sudo systemctl disable --now strata && sudo systemctl enable --now llama-server kallilex-server
sudo systemctl disable --now llama-server kallilex-server && sudo systemctl enable --now strata

# Test gegen eine Config (vorher den Dienst stoppen: Port und VRAM)
SMOKE_ONLY=chat,code,long,long16 SMOKE_REPEAT=2 SMOKE_NONCE=1 deploy/aegis/smoke.sh deploy/aegis/strata-aegis.json name
```

## Nach einem RAM-Upgrade

Nur `--resident-budget-gib` in `strata-aegis.json` erhöhen (Faustregel: MemAvailable unter Last ≥ 2,5 GB lassen). Bei
15,4 GiB Experten im VRAM liegen 8,05 GiB außerhalb: Ab ~24 GiB RAM passt alles (`--resident-budget-gib 8.1` oder
`--resident-experts`), dann liest die SSD nur noch die n-gram-Tabelle. Mehr RAM erlaubt außerdem mehr Kontext und den
Conversation-Cache.

## Engine-Update (Upstream)

Ab 0.2.16 liegen Setup, Server, `.venv` und die gebaute Engine unter `terminal/`, die Engine-Quellen unter `engine/`.
`build-engine.sh`, `smoke.sh` und `smoke.py` sind darauf umgestellt. Auf aegis startet Aitheris (`/etc/aitheris/config.yaml`,
Backend `strata`) `terminal/serve/server.py` mit dieser Run-Config.

```sh
git fetch upstream --tags && git worktree add -b aegis-<ver> ~/code/Strata-<ver> v<ver>   # neuer Checkout neben dem alten
git cherry-pick <aegis-Commits>                    # Konflikt meist nur in terminal/setup.py (AMD_ARCHS)
cd terminal && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
SKIP_TESTS=1 deploy/aegis/build-engine.sh          # baut terminal/build-hip, schreibt terminal/engine/BUILD.json
# Smoke (chat,code,long,long16 + deep), dann Pfade in der Aitheris-Config umstellen und aitheris neu starten
```

Update 2026-11-22 auf 0.2.18 (Engine 0.1.64): siehe [FINDINGS.md](FINDINGS.md), letzter Abschnitt.
