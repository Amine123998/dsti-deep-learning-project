# DSTI Deep Learning Project — Transfer Learning Strategy Comparison

DSTI "Deep Learning with Python" group course project (2026). The group compares different
**transfer learning strategies** applied to the same base model, each implemented by a
different teammate.

- **Task:** Text generation (causal language modeling)
- **Model:** [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (gated — requires accepting the license on Hugging Face)
- **Dataset:** [`Saminx22/medical_data_for_slm`](https://huggingface.co/datasets/Saminx22/medical_data_for_slm) — ~44,400 raw medical documents (PubMed abstracts, PMC full text, WHO/CDC/NICE clinical guidelines)
- **Compute:** Google Colab free tier (T4 GPU, 16 GB VRAM)
- **Project brief:** [`DEEP-LEARNING-Project.pdf`](DEEP-LEARNING-Project.pdf)

## Techniques

| Folder | Technique | Author | Notebook |
|---|---|---|---|
| [`freezing/`](freezing/) | Layer freezing | Amine | [`Freezing_experiment.ipynb`](freezing/Freezing_experiment.ipynb) |
| [`distillation/`](distillation/) | Knowledge distillation | Styr0Karam | [teacher](distillation/Disiliation_teacher.ipynb) / [student](distillation/Disiliation-student.ipynb) |
| [`fine_tuning/`](fine_tuning/) | Full fine-tuning (QLoRA) | youceflgr | [`KYC_Llama3.1_FineTuning.ipynb`](fine_tuning/KYC_Llama3.1_FineTuning.ipynb) |

Each folder has its own README with that technique's method, results and how to run it.

## Results comparison

| Technique | Baseline perplexity | Final perplexity | Change | Trainable params |
|---|---|---|---|---|
| Freezing | 8.66 | 7.78 | **−10.2%** | 436M / 4.76B (9.17%) |
| Distillation (student: Llama-3.2-1B) | TBD — no baseline measured yet | 22.54 (val. perplexity) | TBD | ~852K (0.069%) |
| Fine-tuning (QLoRA) | — (different dataset/task, see note) | — | — | ~42M (~0.5%) |

**Note on comparability:**
- Freezing and distillation both run on `Saminx22/medical_data_for_slm`, but train different
  model sizes (8B vs. 1B) — the *absolute* perplexity numbers are not directly comparable to
  each other, only each technique's improvement over its own baseline.
- The distillation notebooks don't currently measure a pre-distillation baseline, so its
  relative improvement isn't established yet.
- The fine-tuning folder trains on a different dataset (bank KYC compliance Q&A, not the shared
  medical dataset) with its own task-specific metrics (ROUGE-L, semantic similarity, keyword
  recall — see [`fine_tuning/README.md`](fine_tuning/README.md)), so it isn't shown on the
  perplexity table above.

## Reproducibility

Each technique folder pins its own dependencies (`requirements.txt`) and documents how to run
its notebook on Colab. Per the project brief, trained model weights are not committed to this
repository (see [`.gitignore`](.gitignore)).
