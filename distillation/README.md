# Medical Language Model Distillation

A small knowledge-distillation experiment for a medical language model. The project uses a gated Llama teacher to generate top-k logits for 100 training examples, then trains a quantized Llama student with LoRA.

## Project Files

- `Disiliation_teacher.ipynb`: generates `teacher_logits.pt` from the first 100 training examples.
- `Disiliation-student.ipynb`: trains the LoRA adapter, runs generation, and evaluates 50 validation examples.
- `distilled_medical_student/`: saved LoRA adapter for `meta-llama/Llama-3.2-1B-Instruct`.

## Requirements

- Windows with Python 3.10 or newer
- NVIDIA GPU with CUDA support
- At least 12 GB system RAM recommended for CPU offloading
- Hugging Face account with access to:
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `meta-llama/Llama-3.2-1B-Instruct`

Install the CUDA-enabled PyTorch build that matches your driver. For the CUDA 12.1 build used during development:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Run

1. Open `Disiliation_teacher.ipynb` in VS Code and select the CUDA-enabled Python environment.
2. Run the dependency cell, then the login cell. Enter the Hugging Face token when prompted; never commit it.
3. Run the teacher cell. It processes `train[:100]` and writes `teacher_logits.pt` in the project directory.
4. Open `Disiliation-student.ipynb` and run its cells in order.

The student validation currently evaluates `validation[:50]`. The recorded result was a validation loss of `3.1154` and perplexity of `22.54` for that 50-example sample.

## Limitations

This is an experiment, not a medical device or clinical decision-support system. The model can produce inaccurate, biased, unsafe, or fabricated medical content. Do not use its output for diagnosis or treatment decisions.

The base Llama models are gated and remain subject to their respective Hugging Face licenses. This repository contains only the project notebooks and LoRA adapter; it does not redistribute the base models.
