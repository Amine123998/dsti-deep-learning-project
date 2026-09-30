# Fine-tuning Llama 3.1 8B Instruct for bank KYC questions

DSTI deep learning project. We adapt [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
with **QLoRA** so that it answers **KYC (Know Your Customer)** questions from bank staff in a short, direct style.
The questions cover onboarding steps, required documents, beneficial owners, PEPs, sanctions screening, CDD/EDD, risk
factors, red flags, reporting and ongoing monitoring.

The base model and the fine-tuned model go through the same evaluation, so the effect of fine-tuning can be measured directly.

## Repository contents

| Path | Description |
|------|-------------|
| [`KYC_Llama3.1_FineTuning.ipynb`](KYC_Llama3.1_FineTuning.ipynb) | The full pipeline: data, baselines, training, evaluation, publishing, inference |
| [`data/kyc_basic.jsonl`](data/kyc_basic.jsonl) | 15 KYC instruction/answer pairs (`instruction`, `output` fields) |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## Pipeline

| # | Step | What happens |
|---|------|--------------|
| 1 | Setup | Install libraries, log in to the Hugging Face Hub, check the GPU |
| 2 | Data | Load and explore the dataset, add paraphrased questions, build a held-out evaluation set |
| 3 | Base model | Load Llama 3.1 8B Instruct in 4-bit NF4 |
| 4 | Evaluation harness | Prompting strategies, metrics and behaviour tests |
| 5 | Before fine-tuning | Zero-shot, static few-shot and retrieval few-shot runs of the base model |
| 6 | Fine-tuning | QLoRA supervised fine-tuning with TRL `SFTTrainer` |
| 7 | After fine-tuning | Same tests on the fine-tuned model, plus a check for catastrophic forgetting |
| 8 | Comparison | Summary tables, charts and side-by-side answers |
| 9 | Publish | Save the LoRA adapter and a model card, optionally merge it and push to the Hub |
| 10 | Inference | Load the adapter and ask new KYC questions |

### Data

The dataset has only 15 Q&A pairs, which is too small to hold any of them out. Instead:

- **Training:** every original question is kept and gets 3 hand-written paraphrases, giving 60 examples.
- **Evaluation:** each fact gets one *different* paraphrase that never appears in training, with its reference answer and
  a list of keyword groups. The evaluation therefore measures generalisation to new wording.

Examples are stored as conversational prompt/completion pairs (system + user → assistant). The same system prompt is used for
training and for every evaluation run.

### Prompting strategies (baselines)

- **Zero-shot:** system prompt and question only.
- **Static few-shot:** 3 fixed example Q&As are added as earlier chat turns.
- **Retrieval few-shot:** the 3 most similar training Q&As (by `all-MiniLM-L6-v2` embeddings) are used as examples.
  With such a small knowledge base this works like a small RAG baseline.

### Metrics

Decoding is greedy, so runs are deterministic.

| Metric | What it measures |
|--------|------------------|
| ROUGE-L F1 | Word overlap with the reference answer |
| Semantic similarity | Cosine similarity of sentence embeddings (answer vs. reference) |
| Keyword recall | Share of key KYC concepts mentioned; a proxy for factual completeness |
| Words | Answer length |
| Latency | Seconds per answer |

### Behaviour tests

Seven pass/fail scenarios that are not in the training data and require applying the rules: no tipping-off, sanctioned
applicant, missing beneficial-owner information, relative of a PEP, the 25% UBO threshold, ongoing monitoring of a high-risk
client, and a concise definition of CDD.

### Fine-tuning configuration

| Setting | Value |
|---------|-------|
| Quantisation | 4-bit NF4, double quantisation, bf16/fp16 compute |
| LoRA | r = 16, alpha = 32, dropout = 0.05, on `q/k/v/o_proj` and `gate/up/down_proj` (~42M trainable params, ~0.5%) |
| Loss | Completion-only (answer tokens only) |
| Epochs | 3 |
| Learning rate | 2e-4, cosine schedule, 10% warmup |
| Batch size | 4 per device × 2 gradient accumulation = 8 |
| Optimiser | `paged_adamw_8bit`, max grad norm 0.3 |
| Max length | 512 tokens |
| Checkpointing | Eval loss every epoch; best checkpoint restored at the end |

## Getting started

### Requirements

- A GPU with **16 GB of VRAM or more** (Colab T4 / L4 / A100 or similar). Inference needs about 6 GB and training 10 to 14 GB.
- Access to the gated Llama 3.1 model: accept the licence on its
  [Hugging Face page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and create a token with read access
  (write access if you want to push the adapter).

### Installation

```bash
git clone https://github.com/Amine123998/dsti-deep-learning-project.git
cd dsti-deep-learning-project
pip install -r requirements.txt
```

On Colab, clone the repository in the first cell and `%cd` into it so that `requirements.txt` and `data/` are found.

### Hugging Face token

The notebook reads the token from the `HF_TOKEN` environment variable or the Colab secret of the same name. If neither is
set, it shows an interactive login prompt.

```bash
export HF_TOKEN=hf_...
```

### Configuration

Edit the configuration cell at the top of the notebook:

| Variable | Default | Purpose |
|----------|---------|---------|
| `BASE_MODEL_ID` | `meta-llama/Llama-3.1-8B-Instruct` | Model to fine-tune |
| `DATA_PATH` | `data/kyc_basic.jsonl` | Training data |
| `OUTPUT_DIR` | `outputs` | Where results, plots and the adapter are written |
| `HF_REPO_NAME` | `llama-3.1-8b-instruct-kyc-lora` | Hub repository name (under your username) |
| `PUSH_TO_HUB` | `False` | Push the LoRA adapter to the Hub (private repo) |
| `MERGE_AND_PUSH` | `False` | Also push a merged full-weight model (~16 GB, needs a lot of RAM) |

Then run the notebook from top to bottom.

## Outputs

Everything is written to `outputs/`:

| File | Content |
|------|---------|
| `llama31-kyc-lora/` | LoRA adapter, tokenizer and model card (`README.md`) |
| `eval_base.csv`, `eval_finetuned.csv`, `eval_all.csv` | Per-question answers and metrics |
| `summary.csv` | Mean metrics per model and prompting strategy |
| `behaviour_tests.csv` | Pass/fail results of the scenario tests |
| `training_curve.png` | Train and eval loss |
| `comparison.png` | Before/after comparison of quality metrics and answer length |

## Using the fine-tuned model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "<your-hf-username>/llama-3.1-8b-instruct-kyc-lora"  # or "outputs/llama31-kyc-lora"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()

messages = [
    {"role": "system", "content": "You are a KYC (Know Your Customer) compliance assistant for bank employees. "
                                  "Answer questions about client onboarding, due diligence, sanctions, PEPs and AML "
                                  "clearly, accurately and concisely."},
    {"role": "user", "content": "What extra checks do we run for a high-risk client?"},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Expected findings

Exact numbers depend on the GPU, library versions and seed. The pattern the notebook is designed to show:

1. **Style and conciseness improve the most.** The fine-tuned model gives short answers in the reference style, so ROUGE-L and
   semantic similarity rise and answer length and latency drop.
2. **Zero-shot fine-tuned ≈ few-shot base** without spending context tokens on examples.
3. **Retrieval few-shot stays strong on facts**, because the answer is in the prompt. In a real bank the best setup is usually
   fine-tuning for behaviour and style plus retrieval for facts and policies, since policies change.
4. **General abilities** (arithmetic, writing, translation) should be largely preserved, which the regression check verifies.

## Limitations and next steps

- 15 source facts is a teaching-sized dataset, keyword metrics are crude, and the evaluation only measures paraphrase generalisation.
- This is an educational project and **not legal or compliance advice**. Do not use it for onboarding decisions without human review.
- Next steps: build a larger reviewed dataset from real KYC/AML policies (EU AMLD, FATF, local regulators), add retrieval over
  internal procedures (RAG), evaluate with an LLM-as-judge and compliance experts, and try DPO on expert preference pairs to
  strengthen refusal and escalation behaviour.

## License

The fine-tuned adapter inherits the [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE).
