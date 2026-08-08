# Environment report

- Conda environment: `rlalpha`, Python 3.11.15
- Core versions: NumPy 2.4.6, pandas 3.0.5, PyArrow 25.0.0, SciPy 1.17.1, statsmodels 0.14.6, Zarr 3.1.6
- GPU: unavailable to `nvidia-smi` on 2026-08-08 (driver communication failure)
- Qwen3.5-2B: not present under `/data/shared/huggingface`
- CPU M0–M4 dependencies: installed

Run `python -m rlalpha.cli doctor --config configs/experiment/preliminary_screen.yaml` for the current machine-readable report.
