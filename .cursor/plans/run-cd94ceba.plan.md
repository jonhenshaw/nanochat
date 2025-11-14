<!-- cd94ceba-5c79-4026-bd44-9e3b4e546ddd 4c5e307f-2727-42b1-a2d5-acf0ce1965d5 -->
# Plan: Run speedrun.sh on single RTX 4090

### Required changes

1) Update `speedrun.sh` for single-GPU and safe batch sizes

- Set processes to 1 (and allow override):
  - Replace `NPROC_PER_NODE=1` with `NPROC_PER_NODE=${NPROC_PER_NODE:-1}`
- Pass tunable per-GPU batch size and (optionally) shorter context length (4090 profile):
  - Base pretrain: change `--device_batch_size=16` to `--device_batch_size=${DEVICE_BS:-4}` and add `--max_seq_len=${SEQ_LEN:-1024}`
  - Midtraining: add `--device_batch_size=${DEVICE_BS:-4} --max_seq_len=${SEQ_LEN:-1024}`
  - SFT: add `--device_batch_size=${DEVICE_BS:-2}` (SFT is lighter; 2 is safe), you can keep seq len default
- Keep your W&B default `WANDB_RUN=djhenny-nanochat-speedrun-4090` as set.

Minimal edit snippets (for `speedrun.sh`):

```bash
# Number of processes/GPUs to use
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# pretrain the d20 model (4090 profile: bs=4, seq=1024)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
  --depth=20 --device_batch_size=${DEVICE_BS:-4} --max_seq_len=${SEQ_LEN:-1024} --run=$WANDB_RUN

# midtraining (match bs/seq for stability)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- \
  --device_batch_size=${DEVICE_BS:-4} --max_seq_len=${SEQ_LEN:-1024} --run=$WANDB_RUN

# SFT (lighter; bs=2 is safe)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
  --device_batch_size=${DEVICE_BS:-2} --run=$WANDB_RUN
```

Notes

- Users reported successful single-4090 runs with bs≈4 and seq_len=1024, taking ~80 hours for pretrain; see [discussion 212 comment](https://github.com/karpathy/nanochat/discussions/212#discussioncomment-14838417).
- If OOM occurs, set `DEVICE_BS=2` (and/or keep `SEQ_LEN=1024`). If it fits easily, try `DEVICE_BS=3-4`.

### Run commands

- Log in to W&B once (inside venv):
  - `source .venv/bin/activate && wandb login`
- Launch with screen and logs (4090 profile):
  - `DEVICE_BS=4 SEQ_LEN=1024 NPROC_PER_NODE=1 screen -L -Logfile /home/henny/workplace/nanochat/speedrun.log -S speedrun bash /home/henny/workplace/nanochat/speedrun.sh`
- If OOM, retry: `DEVICE_BS=2`.

### Optional improvements (can include now or later)

A) Add periodic checkpoints and resume support to `scripts/base_train.py`

- Add settings: `ckpt_every=500`, `resume_tag=""`, `resume_step=-1`
- Save checkpoints every N steps using existing `save_checkpoint(...)`
- On start, if `resume_tag` set, load model/optimizer with `load_checkpoint(...)` and reduce remaining `num_iterations` accordingly

B) Quality-of-life

- Keep `WANDB_RUN` overridable per-run while keeping your default in the script
- Document how to tail logs and watch dataset/report dirs

### Validation

- Verify GPU visibility: `python -c "import torch;print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"`
- Start run; ensure no `invalid device ordinal` and no OOM at base pretrain start
- Confirm W&B run appears with the chosen name

### To-dos

- [x] Set NPROC_PER_NODE default to 1 in speedrun.sh and make overridable
- [x] Pass DEVICE_BS to base_train in speedrun.sh (default 4 for 4090)
- [x] Pass SEQ_LEN to base_train and mid_train (default 1024)
- [x] Pass DEVICE_BS to mid_train in speedrun.sh (default 4)
- [x] Pass DEVICE_BS to chat_sft in speedrun.sh (default 2)
- [ ] Document and run screen + wandb login + launch command
- [ ] Add periodic checkpoints/resume to scripts/base_train.py (ckpt_every/resume)