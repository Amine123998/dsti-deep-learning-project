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

## Reproducibility

Each technique folder pins its own dependencies (`requirements.txt`) and documents how to run
its notebook on Colab. Per the project brief, trained model weights are not committed to this
repository (see [`.gitignore`](.gitignore)).
