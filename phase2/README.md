# phase2/: the production pipeline on a ranking file

Builds the submission for one ranking file by rule, from the A-side 2025 fits. It reads no earlier submission and uses no row-keyed
values. The entry point is `scripts/run_final.sh <period> [stage ...]`; the steps can also be run one by one:

```bash
bash phase2/run_pipeline.sh <period> ws np_features np_side        # workspace, features, side tables
bash phase2/run_pipeline.sh <period>                                 # base line, FORM=v99 (members must be in place: scripts/run_final.sh <period> members)
bash phase2/steps/p2_v107.sh <period>                                # chain-variant steps, TW1', KRS, KRS-NL -> submissions/final.v111.parquet
```

## Layout
| path | role |
|---|---|
| `run_pipeline.sh` | the base line (`FORM=v99`), one function per step; markers `work/<ws>/logs/<step>.done`, logs `work/<ws>/logs/<step>.log` |
| `steps/p2_v107.sh` | the chain-variant steps (neighbour clocks, Kumo member KCATLg, Kumo in the clock stages, set transformer, Kumo stacker), TW1′, KRS, KRS-NL |
| `p2ws.py` | creates the workspace `work/<ws>/`: copy of `scripts/`, link farms over `data/`, empty `submissions/` |
| `lib/guard/sitecustomize.py` | workspace guard: copy-on-write, fail-closed writes, strict reads, input ledger `logs/ledger.jsonl` |
| `periods/*.env` | one file per ranking file (`final.env` = line A and `final_bag.env` = line B of the submitted file, `public_janjul2026.env` = the public ranking file, January + July 2026, `feb2025dry.env` = February 2025 as a held-out dry run) |
| `manifest/members.tsv` | production recipe of every member family (seeds, GPU, command, role, CPU architecture) |
| `steps/p2_v99_prep.py` | installs the lever scripts from `scripts/` into a workspace and links the 2025 fold side |
| `steps/p2_unmatched.py` | unmatched rows by rule: C1 (`unm_other`) and the seed-bag blend (`unm_LIRF`, exposed) |
| `steps/p2_slot.py` | the stage files under `work/<ws>/stages/` that the stage scripts read by fixed names, built by rule inside the workspace |
| `steps/p2_unm_resid_apply.py`, `steps/p2_unm_link_apply.py` | the unmatched-row residual model and the callsign ADS-B link, by rule |
| `steps/p2_postrules.py` | standfloor stand floor and offblock_pull late-NM pull: `fit` (2025) and `apply` |
| `steps/p2_exposed_scenarios.py`, `steps/p2_e1s.py` | PX-safe rule on exposed LIRF rows (a row moves only if every label scenario agrees); E1s on exposed non-LIRF rows (`MVT` minus the aircraft's own at-stand ADS-B clock) |
| `steps/p2_compose.py`, `steps/p2_tw1_rule.py`, `steps/p2_novelty.py` | isolate composition; TW1′ ids by rule; BIG novelty mask |
| `steps/p2_final_inputs.sh`, `steps/p2_period_inputs.py` | period inputs: OPDI, side tables, ADS-B clock extraction of the ranking days, coverage check |
| `steps/p2_check.py`, `steps/p2_score.py`, `steps/p2_mkdry.py` | validation; scoring and synthetic period for the dry run |

## Rules of the workspace
- The project's `scripts/`, `data/` and `submissions/` are never written; the guard enforces this for every Python process.
- The script copies are byte-identical to the project's `scripts/` (`work/<ws>/manifest/scripts.md5`); `steps/p2_v99_prep.py` installs
  the scripts of the `FORM=v99` stages that a workspace lacks (all of them with `MODE=new`). The period dependence is read from the
  environment the drivers export: `P2_NRANK` (ranking rows, from `NRANK` of the period file), `MODE` (with `MODE=new` the `adsb_features`
  row check of the `*_assemble.py` scripts prints a count instead of asserting) and `PRU_DIR` (the PRU csv directory).
- Every input file a run read is listed in `work/<ws>/logs/ledger.jsonl`.
