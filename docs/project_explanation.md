# Project Explanation

## Goal
This project teaches the full workflow of a small language model system.

## What happens in this project?
1. Download a text dataset.
2. Convert text into tokens.
3. Build training examples.
4. Train a tiny decoder-only transformer.
5. Generate text.
6. Evaluate the trained model.
7. Run distributed training on CPU.
8. Export to ONNX.
9. Serve the model with FastAPI.
10. Prepare Triton model repository.

## Why is this useful?
Even though it runs on CPU, it mirrors real LLM system design.
Later you can replace CPU parts with GPU infrastructure.