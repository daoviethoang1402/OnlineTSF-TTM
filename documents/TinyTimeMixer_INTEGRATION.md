# TinyTimeMixer Integration with OnlineTSF

This document summarizes the changes made to integrate TinyTimeMixer (TTM) into the OnlineTSF framework and support the PROCEED pipeline.

## What was added

1. `models/TinyTimeMixer.py`
   - Added a new model wrapper for TinyTimeMixer.
   - The wrapper loads `TinyTimeMixerForPrediction` from the local `granite-tsfm` repository.
   - It supports two modes:
     - instantiate from scratch with `TinyTimeMixerForPrediction(config)`
     - load HuggingFace pretrained weights via `TinyTimeMixerForPrediction.from_pretrained(...)` when `args.pretrained_model_name_or_path` is provided.
   - The wrapper returns the model's forecast tensor using the repository's expected forward interface.

2. `run.py`
   - Added CLI arguments:
     - `--pretrained_model_name_or_path` for HuggingFace TinyTimeMixer weights.
     - `--load_path` for loading local checkpoints.
   - These arguments allow the user to either bootstrap from HF weights or continue from saved model checkpoints.

3. `settings.py`
   - Added default hyperparameter entries for `TinyTimeMixer`.
   - Added pretraining learning rate mappings for `TinyTimeMixer` and `TinyTimeMixer_RevIN`.

4. `README.md`
   - Added a small CLI usage snippet for TinyTimeMixer and PROCEED.

## How it works

- `Exp_Main._build_model()` dynamically imports `models.TinyTimeMixer.Model` when `--model TinyTimeMixer` is selected.
- The `Model` wrapper in `models/TinyTimeMixer.py` constructs a TinyTimeMixer configuration from the repo's `args`.
- If `args.pretrained_model_name_or_path` is set, it loads pretrained weights before the model is wrapped by `Proceed`.
- `Exp_Proceed` then wraps the backbone in `adapter.proceed.Proceed`, so the online PROCEED pipeline can adapt TinyTimeMixer with adapters.

## How to run PROCEED with TinyTimeMixer

```bash
cd /home/daoviethoang/Desktop/thesis-code/OnlineTSF
python3 run.py \
  --model TinyTimeMixer \
  --online_method Proceed \
  --dataset ETTh1 \
  --seq_len 96 \
  --pred_len 24 \
  --batch_size 32 \
  --pretrained_model_name_or_path <hf_model_name_or_path>
```

To train both backbone and adapter together, ensure `--freeze` is omitted.

## Notes

- `granite-tsfm` must exist as a sibling folder of `OnlineTSF`.
- `TinyTimeMixer` weights are loaded via the local `granite-tsfm` package path.
- If the TinyTimeMixer architecture uses custom layer classes not covered by `adapter/proceed.py`, adapter injection may need extension.
