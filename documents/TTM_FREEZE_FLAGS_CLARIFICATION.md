# TTM Freeze Flags: `--freeze` vs `--no_finetune`

**Scope**: TinyTimeMixer (TTM) backbone only, branch `no-finetune`.
**Related**: [PHASES_AND_FREEZING_CLARIFICATION.md](PHASES_AND_FREEZING_CLARIFICATION.md) (generic PROCEED+PatchTST freezing — written before TTM was integrated; TTM's backbone-always-frozen policy and these two flags are additional, TTM-specific layers on top of that generic strategy). See also the root [../CHANGES.md](../CHANGES.md) for the full TTM integration history.

---

## Question: What does `--freeze` do, and is it the same as `--no_finetune`?

### ✅ Answer: They are two independent axes. `--freeze` is generic and lifelong; `--no_finetune` is TTM-only and only affects the online phase.

## What `--freeze` does

`--freeze` is not TTM-specific — it is PROCEED's generic "approach 1 vs. approach 2" switch, applied to whatever backbone is wrapped. It has three separate effects, all driven by the one flag:

1. **`adapter/proceed.py:28-29`** — freezes the *entire* wrapped model before adapters are injected:
   ```python
   if args.freeze:
       backbone.requires_grad_(False)
   self.backbone = add_adapters_(backbone, args)
   ```
2. **`add_adapters_()` → `freeze_weight=args.freeze`** (`adapter/proceed.py:105/107`, `adapter/module/base.py:11`) — every `Linear`/`Conv1d`/etc. layer wrapped in a `Down_Up` adapter has its *original* weight's `requires_grad` set from `args.freeze`. This is the actual "fine-tune the underlying model or not" switch, applied uniformly to every wrapped layer — backbone, and for TTM, decoder/head too.
3. **`Proceed.more_bias = not args.freeze`** (`adapter/proceed.py:35`) — toggles an extra learned bias term in PROCEED's own adaptation generator. This is a PROCEED-architecture detail, unrelated to any specific backbone.

`--freeze` therefore selects between the two approaches **for the entire pipeline lifetime**:

| | `--freeze` unset (approach 1) | `--freeze` set (approach 2) |
|---|---|---|
| TTM backbone | Frozen (always, TTM-specific policy — see below) | Frozen |
| TTM decoder + head | Fine-tune during pretraining | **Never** trainable, not even during pretraining |
| PROCEED adapters | Trained | Trained |
| `more_bias` term | Yes | No |

## What `--no_finetune` does

`--no_finetune` (`run.py`, `models/TinyTimeMixer.py`) is TTM-only and orthogonal to `--freeze`. It assumes decoder+head **do** fine-tune during pretraining (same as approach 1) and only asks: should that fine-tuning carry over into val/online, or stop there?

| | `--no_finetune` unset (default) | `--no_finetune` set |
|---|---|---|
| Pretraining (phase 1) | Decoder+head fine-tune | Decoder+head fine-tune |
| Val/online (phase 2) | Decoder+head **keep fine-tuning** (via the existing recent/current-batch alternation) | Decoder+head **permanently frozen** — PROCEED adapters must handle drift alone |

Implementation: `Model.__init__` stores `self._no_finetune`; `freeze_head()` (called once at the top of `Exp_Proceed.update_valid()`) is a no-op unless `self._no_finetune` is `True`, in which case it locks decoder+head off via `requires_grad_(False)` and sets `_head_frozen_for_online = True` so later `requires_grad_(True)` calls in the alternation loop no longer restore them.

## Side-by-side

| | `--freeze` | `--no_finetune` |
|---|---|---|
| Scope | Generic — any backbone PROCEED wraps | TTM-specific only |
| Side effects | Also flips `more_bias` and `Down_Up`'s `freeze_weight` everywhere | None — only touches TTM decoder+head trainability |
| Pretraining (phase 1) | If set, decoder+head **never** fine-tune | Decoder+head **always** fine-tune, regardless |
| Val/online (phase 2) | Already frozen the whole time (if set) | Fine-tuned in phase 1, then **locked frozen** before phase 2 |
| Meaningful when combined? | — | Only when `--freeze` is unset — if `--freeze` is set, decoder/head are already frozen everywhere and `--no_finetune` has nothing left to do |

So: `--freeze` asks *"should TTM ever be fine-tuned, from the very start?"* `--no_finetune` assumes yes during pretraining, and only asks *"should that fine-tuning carry over into the online phase, or stop there?"*

---

## `--normalization RevIN`: don't combine it with TTM

TTM already includes its own internal instance normalization — `TinyTimeMixerModel.scaler` (a
`TinyTimeMixerStdScaler`) — that runs inside `tinytimemixer.backbone` on every forward pass.
Wrapping TTM in `models.normalization.ForecastModel` via `--normalization RevIN` adds a
*second*, redundant normalization layer on top of TTM's own. **Recommendation: do not pass
`--normalization` when running `--model TinyTimeMixer`.** Everything in this document (the
freeze-timeline tables, `hasattr` checks, etc.) describes and was verified against the plain
path — `Proceed` wrapping the TTM `Model` directly, with no `ForecastModel` in between.

### Why `ForecastModel` hook-forwarding was still fixed

`freeze_head()`, `post_proceed_init()`, and the freeze-respecting `requires_grad_()` override
are reached via `hasattr(...)` checks in `adapter/proceed.py` and `exp/exp_proceed.py`. If
`--normalization RevIN` *were* used, `exp_main.py` would wrap the TTM `Model` in `ForecastModel`
*before* `Proceed` wraps it, and — before a fix — `ForecastModel` didn't forward these hooks, so
`hasattr` silently evaluated to `False` and none of the freeze logic above ran at all (this is
how the `--no_finetune` bug was first discovered: identical MSE/MAE regardless of the flag).
`ForecastModel` now forwards `post_proceed_init`/`freeze_head`/`requires_grad_` to
`self.backbone` (see `models/normalization.py`, and the root [../CHANGES.md](../CHANGES.md) §6
for the full writeup) — kept in place not because TTM needs it, but so **future foundation
models that lack their own internal normalization** can still be combined with `--normalization
RevIN`/`DishTS` and PROCEED correctly. Any TTM checkpoint trained with `--normalization RevIN`
before this fix does not reflect the freeze policy described in this document.

---

## FAQ

**Q: If I just want approach 2 (adapter-only, nothing about TTM ever trains), do I need `--no_finetune` too?**
A: No. `--freeze` alone already keeps decoder+head frozen for the entire pipeline. `--no_finetune` only matters for approach 1 (`--freeze` unset).

**Q: Can I combine `--freeze` and `--no_finetune`?**
A: You can pass both, but `--no_finetune` becomes a no-op — decoder+head are already frozen for all phases once `--freeze` is set.

**Q: Does `--no_finetune` affect the PROCEED adapters (mlp1/mlp2/generator) or `more_bias`?**
A: No — it only touches `tinytimemixer.decoder` and `tinytimemixer.head` trainability. Everything else (adapter weights, bias term, backbone-always-frozen policy) is unchanged.

**Q: Is the TTM backbone ever trainable under either flag?**
A: No. `self.tinytimemixer.backbone.requires_grad_(False)` is set unconditionally in `Model.__init__` and re-enforced in `post_proceed_init()` and the `requires_grad_()` override, independent of both `--freeze` and `--no_finetune`.

**Q: Should I pass `--normalization RevIN` when running TTM?**
A: No — TTM already normalizes internally via `tinytimemixer.backbone.scaler`. Adding `ForecastModel`/RevIN on top is redundant for TTM specifically; that wrapper is kept working correctly for future foundation models that don't have their own internal normalization.
