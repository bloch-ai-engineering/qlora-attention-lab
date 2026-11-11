# QLoRA & Attention Lab

**We fine-tuned two small LLMs with QLoRA to explain things at three levels (child, student, expert), using a synthetic instruction dataset generated with Llama-3.3-70B. We also built scaled dot-product and multi-head attention from scratch to see what attention actually looks at.**

![Python](https://img.shields.io/badge/python-3.11+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-attention-ee4c2c) ![PEFT](https://img.shields.io/badge/PEFT-QLoRA-yellow) ![bitsandbytes](https://img.shields.io/badge/bitsandbytes-4--bit%20NF4-green) ![TRL](https://img.shields.io/badge/TRL-SFT-black)

## Part 1: Level-adaptive QA with QLoRA

```
Dolly-15k questions ──► Llama-3.3-70B (Nebius, async batched) ──► 3 answers per question
                                                                   child · student · expert
                                                                          │
                     ┌────────────────────────────────────────────────────┤
                     ▼                                                    ▼
      SmolLM2-360M-Instruct (decoder-only)              flan-t5-small (encoder-decoder)
      4-bit NF4 + LoRA r=8, α=32 on q/v projections     same QLoRA recipe, seq2seq format
                     │                                                    │
                     └──────────► base vs. fine-tuned on 54 held-out cases ◄┘
```

| Step | Details |
|---|---|
| Dataset | Questions filtered from Dolly-15k. The LLM wrote one answer per expertise level, with JSON repair for malformed responses. Result: 1,656 training and 54 test examples |
| Fine-tuning | QLoRA: base weights frozen in 4-bit NF4, adapters only on the attention projections, 3 epochs, learning rate 2e-4, on a Colab T4 |
| Formatting | Each architecture got its own format: a `### Question / ### Expertise level / ### Answer` prompt for the causal LM, and an input→target pair for T5 |

### Results

| Model | Improved after fine-tuning | Verdict |
|---|---|---|
| **SmolLM2-360M-Instruct** | 22 / 54 (41%) | Clearly learned to adapt its register. Child: *"like the boss's office"*. Expert: *"designated as such by the state's constitution"* |
| flan-t5-small | 21 / 54 (39%) | The counts look similar, but the outputs collapsed into gibberish |

**Takeaway:** a raw "improved" count hides a lot. Read the outputs. At this scale, the decoder-only instruct model handled level-adaptive generation far better than the ~77M-parameter encoder-decoder model. Both still hallucinate facts, which is expected at this size.

The notebook also answers conceptual questions: why QLoRA saves memory, the trade-offs of LoRA rank, and when fine-tuning beats prompt engineering.

## Part 2: Attention from scratch

- Scaled dot-product attention, `softmax(QKᵀ/√d_k)·V`, implemented in plain PyTorch, with attention heatmaps over token embeddings.
- **Change one word** ("cat" → "dog"): the attention pattern shifts slightly, following the change in meaning.
- **Multiple heads** with separate W_Q, W_K and W_V projections focus on different tokens. This is the intuition behind multi-head attention: each head attends in its own representation subspace.

## Run it

Built for Google Colab with a T4 GPU. Set `NEBIUS_API_KEY` as a Colab secret (only needed to regenerate the dataset).

```bash
pip install -r requirements.txt
jupyter notebook notebooks/qlora_finetuning_and_attention.ipynb
```

## Context

This was homework 3 of the **LLM Architectures** module in **Nebius Academy: AI Performance Engineering**. The task formulations are kept inline in the notebook.

## Authors

**Nathanael Bloch** · **Tomer Porat**
