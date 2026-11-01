# Kangaroo4U

A GPU search for the private key behind a secp256k1 public key in a known range,
set up out of the box for Bitcoin puzzle #140. It runs the SOTA v2 kangaroo
method from [RCKangaroo](https://github.com/RetiredC) by RetiredCoder, keeps
every distinguished point (DP) it finds in a table on disk, and serves a live
dashboard at http://localhost:8899.

Runs on Linux and Windows with an NVIDIA GPU. RTX 40xx and 50xx cards use
hand-written turbo kernels; older cards (GTX 10xx and up) use the regular ones.

## What it does

- Runs a mixed herd of one tame kangaroo per two wild ones, the method's
  default, and stores both kinds of DP.
- Checks every new DP against the table as it arrives. When a tame and a wild
  (or two wilds) land on the same point, it solves the key, prints it, and
  writes it to `data/RESULTS.TXT`.
- Saves the table every 5 minutes and again on exit, so stopping and starting
  again picks up where it left off. Every session adds to the same table.

## Requirements

- An NVIDIA GPU and a current driver
- Python 3.8 or newer
- To build the solver: the CUDA Toolkit 12.x and CMake 3.18+, plus
  - Linux: g++
  - Windows: Visual Studio 2022 or its Build Tools, with the C++ workload

## Build

Linux:

```
./build.sh
```

Windows: double-click `build.bat`, or run it from a command prompt.

Both put the solver and its kernel files in `bin/`. If CMake cannot find
`nvcc`, pass it in, for example `./build.sh -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

## Run

Linux:

```
./run.sh              # search and dashboard
./run.sh dashboard    # dashboard only
./run.sh bench        # solve random 76-bit keys to check the GPU works
```

Windows: double-click `run.bat`, or run `run.bat`, `run.bat dashboard` or
`run.bat bench` from a command prompt.

The dashboard opens in your browser. It listens on this machine only.

## Stop

Press **Ctrl+C once** in the window the solver is running in and wait for
`DP table saved`. Closing the window instead loses whatever was found since the
last save (up to 5 minutes).

## Settings

Everything lives in `settings.ini`:

| setting | meaning |
|---|---|
| `pubkey`, `start`, `range`, `dp` | the search: public key, start of the range in hex, range width in bits, DP bits |
| `gpus` | which GPUs to use, as digits (`01` = GPUs 0 and 1); empty = all |
| `power_limit_watts` | optional power cap applied at start; needs sudo (Linux) or Administrator (Windows) |
| `memory_clock_mhz` | optional memory clock lock; the search barely uses memory, so a lower clock can leave more of the power budget for the cores |
| `save_every_seconds` | how often the table is saved |
| `port`, `open_browser` | dashboard |

The table file is named after the key, range and DP, so pointing the settings at
a different search starts a new table instead of mixing the two.

## Files

```
data/table-*.rcdb     the DP table (keep it; it is all the work done so far)
data/solver.log       solver output
data/RESULTS.TXT      the key, once found
```

## Odds

Puzzle #140 is a 139-bit range. Expected work is about 2^70 group operations,
which is about 2,000 years on one RTX 4090. Every DP stored still counts
toward that total, and the dashboard shows the progress against it.

## License

GPLv3, see `LICENSE.TXT`. The solver is RCKangaroo v4.0 (c) 2024-2026
RetiredCoder, with additions for a persistent DP table, graceful stop and
Windows file handling.
