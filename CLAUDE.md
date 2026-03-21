# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**boomtitan** is a fork of [torchtitan](https://github.com/pytorch/torchtitan) — a PyTorch-native pretraining platform — customized for training the **BOOM/SmolLM** family of LLMs (135M, 360M, 3B, 13B). The primary model is the `boom` model (in `torchtitan/models/boom/`), trained on the SL3 datamix (2.6T tokens). Training runs on CSCS/Leonardo HPC clusters using SLURM.

## Commands

### Install
```bash
pip install -r requirements.txt       # production deps
pip install -e ".[dev]"               # editable install with dev deps (pre-commit, pytest)
pre-commit install                    # set up linting hooks
```

### Run training (local, single-node)
```bash
# Defaults to smollm3B config, 8 GPUs
./run_train.sh

# Override config or GPU count
CONFIG_FILE=./torchtitan/models/boom/train_configs/debug_smollm135M.toml NGPU=4 ./run_train.sh

# Pass extra CLI args (override any TOML key)
./run_train.sh --training.steps 100 --checkpoint.enable_checkpoint true
```

### Run training (multi-node SLURM on Leonardo)

All scripts live in `scripts/slurm/`. The cluster defaults (account, partition, venv, paths) are in `scripts/slurm/setup_env_leonardo.sh`.

```bash
# Single-replica (no TorchFT), 2 nodes × 4 GPUs, local_bs=4
TIME=01:00:00 scripts/slurm/jobs/1B_fw_edu_leonardo.sh

# Manual single-replica via submit.sh
CONFIG_FILE=./torchtitan/models/boom/train_configs/fw-edu-local/local_fw_edu_1B_leonardo.toml \
LOCAL_BATCH_SIZE=4 NUM_NODES=2 NUM_GPUS=4 TIME=01:00:00 \
  scripts/slurm/submit.sh --nodes=2 --gres=gpu:4 scripts/slurm/train.sh

# Multi-replica (TorchFT) — batch math computed automatically
# GLOBAL_BATCH_SIZE = NUM_REPLICAS × NUM_NODES × NUM_GPUS × GRAD_ACC × LOCAL_BS
GLOBAL_BATCH_SIZE=64 NUM_REPLICAS=2 NUM_NODES=2 NUM_GPUS=4 \
  SYNC_METHOD=local_sgd CONFIG_FILE=./torchtitan/models/boom/train_configs/fw-edu-local/local_fw_edu_1B_leonardo.toml \
  bash scripts/slurm/launch_replicas.sh --name my_exp

# Resume a multi-replica run
GLOBAL_BATCH_SIZE=64 NUM_REPLICAS=2 NUM_NODES=2 NUM_GPUS=4 SYNC_METHOD=local_sgd \
  bash scripts/slurm/launch_replicas.sh --resume my_exp
```

**Lighthouse** (TorchFT coordinator, required for multi-replica):
```bash
sbatch scripts/slurm/lighthouse.sh
# Writes address to $BASE_CHECKPOINT_PATH/lighthouse_addr.txt — auto-discovered by launch_replicas.sh
```

**Key SLURM scripts**:

| Script | Purpose |
|---|---|
| `setup_env_leonardo.sh` | Module load, venv activate, paths, NCCL, proxy |
| `submit.sh` | Wrapper that adds `--account`/`--partition` defaults |
| `train.sh` | Single-replica torchrun job |
| `replica.sh` | TorchFT replica job (one per replica) |
| `lighthouse.sh` | TorchFT lighthouse coordinator |
| `launch_replicas.sh` | Submit lighthouse + all replicas, compute batch math |
| `jobs/1B_fw_edu_leonardo.sh` | Preset job for 1B model on fw-edu data |

**Overridable env vars** (all scripts):

- `TIME` — sbatch time limit (e.g. `01:00:00`)
- `CONFIG_FILE` — TOML config path
- `LOCAL_BATCH_SIZE`, `GRAD_ACC_STEPS` — batch size
- `NUM_NODES`, `NUM_GPUS` — topology
- `SYNC_METHOD` — `local_sgd` | `diloco` | `hsdp`
- `SYNC_STEPS` — outer sync frequency (default 30)
- `EXTRA_SBATCH_ARGS` — extra sbatch flags
- `EXTRA_TRAIN_ARGS` — extra torchtitan CLI args

### Tests
```bash
pytest tests/unit_tests/              # unit tests (no GPU required)
pytest tests/                         # all tests
python tests/integration_tests.py     # integration tests (requires GPUs)
```

### Linting
```bash
pre-commit run --all-files
```

### Tokenizer / checkpoint utilities
```bash
python scripts/download_tokenizer.py --repo_id meta-llama/Llama-3.2-1B --tokenizer_path ./assets/tokenizer/
python scripts/download_ckp.py       # download checkpoints from HF
```

## Architecture

### Training entry point
`torchtitan/train.py` — contains the `Trainer` class (implements `Stateful`) that orchestrates the full training loop: config loading → model init → parallelization → dataloader → optimizer → main loop → checkpointing.

### Config system
- All config lives in TOML files under `torchtitan/models/boom/train_configs/`
- `torchtitan/config/job_config.py` defines the full `JobConfig` dataclass (nested: `Job`, `Model`, `Training`, `Optimizer`, `LRScheduler`, `Parallelism`, `Checkpoint`, `ActivationCheckpoint`, `Float8`, `Metrics`, `Profiling`, `Validation`, `FaultTolerance`)
- TOML keys map directly to dataclass fields; CLI args (`--section.field value`) override TOML
- Model-specific config overrides happen in `TransformerModelArgs.update_from_config()`

### BOOM model (`torchtitan/models/boom/`)
- `model/args.py` — `TransformerModelArgs` dataclass: key fields are `freq_nope` (disable RoPE every Nth layer), `use_qk_norm`, `use_flex_attn`, `attn_mask_type`
- `model/model.py` — Transformer implementation with optional FlexAttention and per-layer NoPE
- `infra/parallelize.py` — applies FSDP2/TP/CP to the boom model
- `infra/pipeline.py` — pipeline parallel split for boom model
- `__init__.py` — registers the `boom` model via `TrainSpec`

Predefined flavors: `smollm135M`, `smollm360M`, `smollm3B`, `smollm13B`, `8B`, `70B`, `405B`.

### Parallelism strategy
Controlled via `[parallelism]` in TOML / `--parallelism.*` CLI flags:
- `data_parallel_shard_degree = -1` → FSDP2 across all GPUs
- `data_parallel_replicate_degree > 1` → HSDP (hybrid sharded)
- `tensor_parallel_degree`, `pipeline_parallel_degree`, `context_parallel_degree`
- **Constraint**: FlexAttention with `block_causal`/`document_causal` mask does **not** support PP or CP

### Fault tolerance (TorchFT)
- `torchtitan/components/ft.py` — `FTManager` wraps the optimizer/model for semi-synchronous training
- Activated via `--fault_tolerance.enable` with `--fault_tolerance.group_size` and `--fault_tolerance.replica_id`
- Three sync methods in the SLURM launcher: `hsdp`, `local_sgd`, `diloco`
- Requires a running TorchFT Lighthouse server (`run_lighthouse.slurm`)

### Data pipeline
- `torchtitan/datasets/` — `TokenizedBytesDataset` (default, binary shards), `HFDataset` (HuggingFace), `DatatroveDataset`
- Multi-dataset mixing is configured via `dataset_read_path` (list) + `dataset_weights` in `[[data_stages]]`
- The SL3 datamix has 40 sources; data lives in `./data/SL3_2.6TT/<source>/`
- `data_stages` list supports curriculum learning (different seq_len or data mix at different steps)

### Extension / protocol system
- `torchtitan/protocols/train_spec.py` — `TrainSpec` protocol; new models register by returning a `TrainSpec` from a factory and registering it in the model `__init__.py`
- `torchtitan/protocols/model_converter.py` — `ModelConverter` protocol for post-init transforms (e.g., float8 conversion)

### Key TOML config directories
| Path | Purpose |
|---|---|
| `train_configs/sl3-datamix/` | Production configs (smollm3B, smollm13B) |
| `train_configs/debug_*.toml` | Small fast configs for local testing |
| `train_configs/nope/`, `qknorm/`, `doc_masking/` | Ablation experiment configs |

### Cluster paths (CSCS Alps)

- Python env: `/opt/titan/bin/activate`
- Working dir: `/iopsstor/scratch/cscs/mciccone/exp/boomtitan`
- HF cache: `$SCRATCH/huggingface/`
- WANDB project: `boomtitan`, entity: `mciccone`

### Cluster paths (Leonardo @ CINECA)

- Account: `IscrB_Decentro`, partition: `boost_usr_prod` (A100 64GB)
- Python env: `/leonardo_scratch/fast/IscrB_Decentro/mciccone/envs/boom/bin/activate`
- Working dir: `/leonardo/home/userexternal/mciccone/exp/boomtitan`
- Checkpoints: `/leonardo_scratch/fast/IscrB_Decentro/mciccone/checkpoints/`
- Logs: `/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs/`
- HF cache: `/leonardo_scratch/fast/IscrB_Decentro/mciccone/huggingface/`
- fw-edu data: `/leonardo_work/IscrB_Decentro/mciccone/sl3/datasets/fw-edu/sample/10BT`
- SSH alias: `ssh leonardo`
- Rsync: `rsync -av --exclude='boom_outputs/' --exclude='results/' --exclude='data/' --exclude='wandb/' --exclude='logs/' /Users/marcociccone/exp/boomtitan_claude/ leonardo:/leonardo/home/userexternal/mciccone/exp/boomtitan/`

### WandB logging

- Project: `boomtitan`, entity: `mciccone`
- Enabled via `metrics.enable_wandb = true` in TOML
- Run name set via `WANDB_NAME` env var (set automatically by `launch_replicas.sh` as `${RUN_NAME}_replica_${i}`)
- The proxy (`scripts/slurm/setup_env_leonardo.sh`) enables online WandB from compute nodes via a SOCKS5 tunnel; port stored in `$HOME/scripts/.leonardo_port`
- **Only** `HTTP_PROXY`/`HTTPS_PROXY` are set — never `ALL_PROXY` (breaks Gloo/torchft raw TCP); `GRPC_PROXY_OVERRIDE=""` prevents torchft gRPC from routing through the tunnel
- If proxy is unavailable, falls back to `WANDB_MODE=offline` automatically

### Observed performance (1B model, 2 nodes × 4 A100-64GB, FSDP2, no AC)

- MFU: ~44%, TFLOPs: ~137, throughput: ~15,200 tps/GPU
- Memory: ~33.5 GiB / 64 GiB (52%)
- Config: FlexAttention + document_causal + QKNorm + NoPE(freq=4) + torch.compile
