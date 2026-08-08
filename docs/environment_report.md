# Environment report

- Conda environment: `rlalpha`, Python 3.11.15
- Core versions: NumPy 2.3.5, pandas 3.0.5, PyArrow 25.0.0, SciPy 1.17.1, statsmodels 0.14.6, Zarr 3.1.6
- LLM stack: Torch 2.11.0+cu130, Transformers 5.10.4, vLLM 0.26.0,
  Ray 2.56.1, PEFT 0.20.0 and Verl commit
  `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`
- Qwen3.5-2B revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`
  at `/data/shared/huggingface/Qwen3.5-2B`
- Model weight fingerprint:
  `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`
- Acceptance GPUs: physical GPU 2 (A100 80GB), GPU 3 (H800), GPU 4
  (L40S), shared with existing GPUStack services
- Solvers: OSQP and CLARABEL available through CVXPY

Run `python -m rlalpha.cli doctor --config configs/experiment/preliminary_screen.yaml` for the current machine-readable report.
