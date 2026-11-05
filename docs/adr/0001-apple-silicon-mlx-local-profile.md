# ADR-0001: Apple Silicon (MLX) local profile instead of a Linux GPU

- Status: accepted (2026-11-05)
- Context: the team has no Linux/NVIDIA GPU yet. The dev machine is an Apple M4 Pro with
  24 GB unified memory. vLLM, bitsandbytes and FlashAttention need CUDA.
- Decision: all model work goes through the `zllf.llm.base` interfaces with three backends:
  `fake` (tests/CI), `mlx` (Apple Silicon: `mlx-lm` for generation, LoRA/QLoRA training on
  4-bit models, and its OpenAI-compatible server for serving), and `vllm` (Linux/CUDA, later).
  Hardware presets: `fake`, `toy` (0.5B models, minutes), `mac` (1.5B student, 7B teacher).
- Consequences: R1/R2 can be developed and demonstrated end to end on the Mac. Absolute
  numbers (model size, time-to-model) are below the PRD's 8B/48 GB target; the PRD targets are
  re-validated on CUDA hardware once available. The trainer and serving code have a backend
  seam so a PEFT/TRL + vLLM implementation can be added without touching the stages.
