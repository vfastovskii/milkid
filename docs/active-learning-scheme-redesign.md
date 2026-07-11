# Active-learning scheme redesign — plan

Goal (user intent): the attention aggregator learns slower than embedder/predictor.
Reduce embedder+predictor LR at the right time while the aggregator keeps learning,
enriched by the active-prototype bank — with **all LR/curriculum actions linked to the
loss**, and a **professional, reproducible, correct** prototype bank.

Derived from the audit (see conversation): 286/647 train molecules are active (≥7.0);
the "~25" the user sees is `bank.num_active()` throttled by `max_prototypes_per_series=2 × 13
series`. Two LR reducers currently fight; the config query-ramp is dead; enrichment can be a
silent no-op; bank has a latent cross-series contamination bug + order-dependence.

## Design decisions (approved)
- **LR authority**: one loss-linked curriculum owns LR + query. Disable `ReduceLROnPlateau`.
- **Bank**: full professional rewrite (unify averaging, order-invariant construction, relax
  per-series cap, fix correctness + coupling).

## Phase machine (single authority, all transitions loss-linked)
Internal val-plateau detector on the checkpoint metric (`val_rmse`), params `plateau_patience`,
`plateau_min_delta` (reuse existing gap config where sensible).

- **Phase A — joint warmup**: all components at configured per-component LRs; bank fills.
  Exit A when: val plateaus (no improvement ≥ `plateau_min_delta` for `plateau_patience` epochs)
  **AND** `bank.num_active() ≥ min_active_prototypes`. If val plateaus but bank not ready → stay
  in A and log a WARN (don't freeze the extractor into an empty bank).
- **Phase B — aggregator focus**: on entry, scale `Embedder.`/`Predictor.` LR ×`focus_lr_factor`;
  keep `Aggregator.`/`Active query builder.` LR. Ramp query weight 0→`query_max_weight` over a
  short smoothing window `query_ramp_epochs` (a smoothing, not a schedule). Query injection is now
  guaranteed non-empty (entry gated on `num_active ≥ min_active`).
  Exit B (stop) when: val plateaus **again** since entering B (`plateau_patience` epochs). No
  hardcoded 25-epoch count; `attention_refinement_query_epochs`/`stop_after_query_epochs` removed.
- **Checkpoint**: keep `checkpoint_monitor=val_rmse`; lower `checkpoint_min_epoch` so the true
  best is captured (separate loss-diagnosis item, not strictly part of this rewrite).

## Files — Stage 1 (LR/curriculum unification)
- `ppl/models/lightning/optimization.py`: default scheduler to `none` when the curriculum owns LR
  (or honor `scheduler: none` from config); keep per-component AdamW groups.
- `ppl/models/lightning/training_curriculum.py`: replace gap-trigger + one-shot LR cut + hardcoded
  25-epoch ramp/stop with the A/B phase machine above; couple Phase B entry to `num_active`.
- `ppl/models/active_prototype_forward.py`: make the effective query weight **only** the curriculum
  weight (remove the dead config ramp precedence); state-drive it.
- config `run_config.yaml`: `scheduler: none`; drop dead `query_start_epoch/query_ramp_epochs/
  query_max_weight` duplication and `attention_refinement_query_epochs/stop_after_query_epochs`;
  add `focus_lr_factor`, `plateau_patience`, `plateau_min_delta`.
- Verify: characterization test — model builds, optimizer groups unchanged, a synthetic multi-epoch
  loop drives A→B→stop deterministically; forward output MD5 stable pre/post where behavior
  shouldn't change (warmup epochs).

## Files — Stage 2 (bank professional rewrite)
- `ppl/models/active_prototype_memory.py`:
  - Unify averaging to one **count-weighted running mean** for update and merge
    (`p ← normalize((n·p + c)/(n+1))`); drop the fixed-momentum EMA inconsistency.
  - **Order-invariant construction**: accumulate active candidates per epoch, rebuild the bank at
    epoch end via deterministic per-series agglomerative clustering down to `max_prototypes_per_series`
    (canonical sort of candidates first). Prototype = count-weighted centroid, normalized.
  - **B2 fix**: never EMA/merge across series — same-series only; skip if no same-series slot.
  - **B3 fix**: prune grace period (only prune prototypes older than N updates) or EMA-decayed support.
  - Relax `max_prototypes_per_series` (or make it proportional to per-series active count, capped by
    `max_prototypes`) so more of the 286 actives form prototypes.
- `ppl/models/active_prototype_forward.py`: move candidate collection to per-epoch accumulation +
  epoch-end rebuild; keep train-only + no-leakage guarantees.
- config: expose the new cap policy.
- Verify: bank determinism test (same candidates in any order → identical prototypes), no cross-series
  contamination, per-series counts match the relaxed cap; num_active reflects the new policy.

## Non-goals / kept correct
- No val/test leakage (train-only update, frozen bank at eval) — preserved.
- Query built from detached buffers (grads → embedder + query_proj only) — preserved.
- Eval re-applies the refinement phase at the checkpoint epoch (already correct) — preserved.
