# LLM Training & Serving Infrastructure

A production-style, end-to-end LLM infrastructure project built to demonstrate the core systems behind training, evaluating, serving, and benchmarking language models on a laptop-scale environment.

This project mirrors real-world LLM platform patterns using a lightweight local setup:
- **PyTorch** for model training
- **PyTorch DDP** for distributed training simulation
- **FastAPI** for inference serving
- **ONNX** for model export
- **Triton-compatible model repository structure** for deployment readiness
- **Benchmarking + evaluation** for performance validation

While this implementation runs on **CPU** for accessibility, the code is intentionally structured to reflect how the same workflow would extend to **CUDA/NCCL-based GPU training and high-throughput inference systems**.

---

## Why this project stands out

Most small ML projects stop at training a model. This project goes further and demonstrates the full ML systems lifecycle:

- dataset ingestion and preprocessing
- tokenizer design
- decoder-only Transformer implementation
- training loop and checkpointing
- distributed training workflow with `torchrun`
- text generation and validation
- inference API serving
- latency benchmarking
- ONNX export
- Triton-ready repository layout
- modular codebase with tests and documentation

This makes the project relevant for roles in:

- ML Infrastructure
- LLM Platform Engineering
- AI/ML Engineering
- Inference Systems
- Applied ML Systems
- MLOps / Model Serving

---

## Project highlights

- Built a **decoder-only Transformer** from scratch for next-token prediction
- Implemented a full **training + validation + checkpointing** workflow in PyTorch
- Simulated **distributed training** with PyTorch DDP on CPU using `torchrun`
- Exposed inference through a **FastAPI service**
- Added a client and benchmark utility for end-to-end testing
- Exported the trained model to **ONNX**
- Organized artifacts into a **Triton-compatible model repository**
- Designed the project as a clean, modular, recruiter-facing GitHub portfolio piece

---

## Sample results

Representative local run results from the current implementation:

- **Validation Loss:** `1.5126`
- **Perplexity:** `4.54`
- **Average API Latency:** `0.2126s` over 10 requests

Example generated output:

ROMEO: then, I say he will poss'd!
Therefore shall be resign to husband them;
And three with his groan soft
## Architecture
Dataset -> Tokenizer -> Next-Token Dataset -> TinyGPT Training -> Checkpoint
                                                        |
                                                        v
                                               Evaluation / Generation
                                                        |
                                                        v
                                             FastAPI Inference Service
                                                        |
                                                        v
                                                 Benchmarking Client
                                                        |
                                                        v
                                                   ONNX Export
                                                        |
                                                        v
                                         Triton-Compatible Model Layout
##Repository structure
llm-infra-lab/
│
├── README.md
├── requirements.txt
├── .gitignore
├── setup.sh
├── setup.ps1
│
├── configs/
│   ├── train.yaml
│   └── infer.yaml
│
├── data/
│   └── raw/
│
├── docs/
│   └── project_explanation.md
│
├── scripts/
│   ├── download_tinyshakespeare.py
│   ├── run_train.sh
│   ├── run_ddp.sh
│   ├── run_api.sh
│   └── run_export.sh
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── train_ddp.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── export_onnx.py
│   └── benchmark.py
│
├── serving/
│   ├── __init__.py
│   ├── app.py
│   ├── schemas.py
│   └── client.py
│
├── tests/
│   ├── test_tokenizer.py
│   └── test_model.py
│
└── triton_repo/
    └── tiny_llm_onnx/
        ├── 1/
        └── config.pbtxt
Setup
Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
Quickstart
1. Download dataset
python scripts\download_tinyshakespeare.py
2. Train the model
python -m src.train
3. Generate sample text
python -m src.generate
4. Evaluate the trained model
python -m src.evaluate
5. Start the inference API
uvicorn serving.app:app --reload --port 8000
6. Test the API from a client

Open a new terminal:

cd C:\Users\JAIBALAYYA\desktop\llm-infra-lab
.venv\Scripts\Activate.ps1
python serving\client.py
7. Benchmark inference latency
python -m src.benchmark
8. Run distributed training simulation
torchrun --nproc-per-node=2 -m src.train_ddp
9. Export the model to ONNX
python -m src.export_onnx
Notes on ONNX export

If ONNX export fails in your environment due to version mismatch, ensure:

onnx

onnxruntime

onnxscript

are installed.

In newer PyTorch environments, using:

opset_version=18

dynamo=False

may be more stable for this project than older exporter defaults.

Design decisions
Why character-level tokenization?

Character tokenization keeps the pipeline fully transparent and easy to understand. It is ideal for learning core language-model mechanics before moving to BPE or Hugging Face tokenizers.

Why CPU instead of GPU?

The goal of this project is to teach architecture and systems thinking, not just maximize training speed. The same code structure can later be adapted to:

CUDA

NCCL

mixed precision

larger models

high-throughput inference engines

Why Triton-compatible structure even on a laptop?

A strong ML systems project should demonstrate deployment awareness. Organizing the ONNX model into a Triton repository shows production-oriented thinking even if the final server is not deployed on local hardware.

Current limitations

This is intentionally a laptop-scale project, so a few constraints are expected:

training runs on CPU instead of GPU

DDP uses CPU/Gloo rather than CUDA/NCCL

the model is intentionally small for reproducibility

Triton layout is prepared, but a full production Triton deployment is outside the laptop-only scope

vLLM and large-scale GPU optimizations are future extensions

These limitations are a design choice to make the project reproducible, understandable, and accessible.

Future improvements

Planned next steps:

switch from character tokenizer to subword tokenizer

train on a larger text corpus

add learning rate scheduling

add logging/metrics persistence

support mixed precision on GPU

migrate from CPU/Gloo to CUDA/NCCL

integrate Docker for serving workflows

validate ONNX output with ONNX Runtime

add Triton server inference test

extend to WSL/Linux + vLLM for advanced serving experiments

Testing

Run unit tests with:

pytest -q

Current tests cover:

tokenizer encode/decode correctness

model forward pass shape and loss generation

Resume-ready project summary

LLM Training & Serving Infrastructure
Built an end-to-end LLM infrastructure project using PyTorch, including tokenizer development, next-token dataset creation, Transformer training, checkpointing, text generation, evaluation, FastAPI-based model serving, latency benchmarking, ONNX export, and Triton-compatible deployment structure.

Interview talking points

This project gives strong material for technical interviews. Example talking points:

how next-token prediction datasets are constructed

why causal masking is required in decoder-only Transformers

how DDP mirrors production multi-worker training

how checkpointing and evaluation fit into training systems

how inference serving differs from training

how ONNX export helps bridge training and deployment

what changes would be required to move from CPU/Gloo to GPU/NCCL

how to scale this design for real LLM infrastructure

Takeaway

This project was built to demonstrate not just model training, but LLM systems engineering.

It shows the ability to think across the full stack:

model architecture

training workflow

distributed execution

inference serving

benchmarking

deployment preparation

reproducible software structure

That full lifecycle perspective is what makes this project relevant for ML infrastructure and AI platform roles.


A couple of small edits will make it even stronger:
replace `llm-infra-lab/` in the tree with your final repo name, and add your GitHub username once the repo is live.
