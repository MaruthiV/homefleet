# Reproducing the corrected paper

Use release **v0.2.0**, protocol **causal-v3**, for the named preprint *Adaptive Reliability Reserves for Residential Battery Fleets*. The original workshop results used different information and accounting conventions and must not be mixed with this release. [Revision notes](REVISION_NOTES.md) describe the corrections.

## Environment and a first run

Python 3.12 is required. Commands below run at the repository root on macOS or Linux.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m homefleet.demo --check
python -m pytest -q
```

The example includes eight ResStock simulated homes, 28 days of preceding history, and two days of the Houston Beryl replay. It requires no network access. The two policies use identical homes and outage draws. Its weekly margin is a rate over that short episode, not a measured full-week profit. `physical_violations` and `reachable_floor_violations` must both be zero; charging recovery shortfall is a separate quantity.

`requirements-lock.txt` records the tested macOS ARM64 Python 3.12 environment and fixes versions for the paper's numerical reproduction. Wheels and low-level dependencies can differ on other platforms. `pip install -e '.[research,dev]'` is the portable development installation; use the lock for reported numbers and fitted-model compatibility. Optional Numba accelerates the same dynamic program. Tests compare it with the NumPy reference. First use includes compilation overhead.

BLAST-Lite requires a separate NumPy 1 environment:

```bash
python3.12 -m venv envs/blast
envs/blast/bin/python -m pip install -r requirements-blast-lock.txt
```

`requirements-blast.txt` specifies the smaller direct dependency set when reproducing on another platform. BLAST is used only by the aging sensitivity, not by training or the main evaluations.

## Frozen artifacts

```bash
python scripts/download_data.py --group bundles
python scripts/download_data.py --group raw
python scripts/download_data.py --group checkpoints
python scripts/download_data.py --group results
python scripts/verify_data.py
```

The manifest in `artifacts/release_manifest.json` identifies the versioned release URLs, byte counts, and SHA-256 hashes. The downloader verifies a complete archive before extracting it. `--destination /path/to/checkout` selects another checkout; run orchestration commands from that checkout. `HOMEFLEET_ROOT` overrides the runtime workspace root. Full repository orchestration requires that root to contain the checkout; installed data-loader modules can also use a data-only root.

The bundles include individual simulated load and PV series, building identifiers, observed county fractions, prices, alerts, completed-hour weather, and causal risk forecasts. They are sufficient for controller evaluation. `--group raw` also supplies the saved hazard diagnostics used by manuscript generation, as well as cached source inputs and the feeder needed for rebuilding and ancillary analyses. [The data card](DATA_CARD.md) defines every field and its provenance. Downloading upstream data afresh may yield revised source records and is a separate operation from reproducing the frozen release.

## Complete experiment grid

`configs/revision.yaml` freezes training budgets, seeds, penalties, validation selection, and main evaluation datasets. `configs/splits.yaml` specifies exact event start/end times. The manifest generator enumerates every expected controller/window result.

```bash
python scripts/plan_experiments.py
python scripts/reproduce.py baselines --workers 3
python scripts/reproduce.py train --workers 4
python scripts/reproduce.py policies --workers 4
python scripts/verify_results.py
python scripts/summarize_revision.py
```

Training produces 21 checkpoints: ten learned-reserve promise-cost runs (penalties 5 and 20, seeds 1–5), seven learned-reserve full-outage-cost runs (penalty 1, seeds 0–1; penalty 5, seeds 0–2; penalty 20, seeds 0–1), and four full-cost fixed/no-operating-reserve ablations (penalty 5, seeds 0–1). The last group still enforces physical battery limits. Each run uses 2,998,272 transitions, the largest complete rollout multiple below three million. Configuration, library versions, training progress, and source hashes are saved beside each checkpoint. Training and evaluation are CPU workloads; the full grid is substantially larger than the included example.

The jobs resume completed rows and checkpoints. CSV evaluation phases require one writer per output. Training serializes each run with a file lock and preserves a matching completed checkpoint. `--runs` restricts a training/policy queue to exact run names; use a distinct `--status-label` when launching another queue. `policies --available-only` evaluates checkpoints that have already finished while training continues; the final verifier still requires all planned outputs. Deleting a checkpoint deliberately retrains that run. Deleting a result CSV deliberately recomputes that evaluation.

The main grid contains eleven datasets: five regular-year/metro evaluations and six event-only datasets. Every learned checkpoint is evaluated with and without the alert override. The verifier requires exact coverage, matching duration/configuration, valid per-home counts, and zero physical/reachable-floor violations. It rejects legacy results.

The ten promise-cost candidates are selected using **Houston 2024 only**: maximize annualized inventory-adjusted margin minus $20 per kWh of first-five-hour unserved energy, with controller name as a deterministic tie breaker. `selected_policy.json` records the chosen checkpoint. Guarded evaluations and other years/events do not select it. Raw-cost and reserve ablations are reported as sensitivity results, not extra candidates for the headline selection.

Annual summaries use only nonoverlapping regular windows, weighted by duration. Named windows are scored separately. Promise rates pool episode/failure counts. Paired bootstrap intervals resample regular windows with seed 0 and 2,000 draws. They do not cover uncertainty in the simulated outage construction or future weather. Each window starts at 90% charge; annualized figures therefore describe episodic replay. The 2026 split has no observed outage truth and no reliability score.

`seasonal.csv` groups whole regular windows by the season of their first America/Chicago timestamp (DJF, MAM, JJA, SON). A window crossing a boundary remains in its starting cohort. These cohort rates preserve the full windows and paired outage draws; they are not exact calendar-season totals. The paper reports Houston 2024, and the released CSV retains every controller/dataset and bootstrap interval. Selection still uses only the full validation-year aggregate.

Named storms before 2024 are event holdouts within the 2019–2023 training-year range, not chronological deployment backtests. Training excludes their scored intervals but can use other later dates and prior-observation buffers after an event; no extra 28-day feature embargo is imposed. Houston 2025 and 2026 are the chronological tests after all PPO training dates.

## Stress tests, system consequences, and figures

After selection, run:

```bash
python scripts/eval_structure.py --workers 2
python scripts/degradation_table.py
python scripts/ac_check.py --policy "$(python -c 'import json; print(json.load(open("experiments/revised/selected_policy.json"))["path"])')"
python scripts/stress_guardrail.py --policy "$(python -c 'import json; print(json.load(open("experiments/revised/selected_policy.json"))["path"])')"
python scripts/benchmark_latency.py
python scripts/prop_constants.py
python scripts/build_descriptive.py
python scripts/build_paper_results.py
```

Fleet sensitivities use six fixed windows: Houston 2024 `w10`, derecho, Beryl and August heat, Houston Uri 2021, and Houston Elliott 2022. Five scenarios cover the base 100-home mix, 500 and 1,000 homes, 70% Core inverters, and 20% per-interval communication dropout. These are window sensitivities, not independent annual generalization trials. Larger fleets resample the same 300-profile pool.

The aging analysis uses 20 homes and four 2024 windows. BLAST repeats each SOC and ambient-temperature trace to one year. Cycle/throughput models annualize each window. Discontinuous repetition, ambient rather than cell temperature, and uncalibrated pack chemistry limit the interpretation to model sensitivity.

The AC audit evaluates three days at 30% and 100% battery penetration, retaining the feeder's load models. It reports phase voltages at buses reachable in the bus-level source graph, convergence and line loading, plus the count of phase nodes at unreachable buses. This reachability mask does not certify conductor continuity or phase energization. The load/PV reference uses identical placement and profiles. Existing violations are retained. The Austin synthetic feeder with Houston load shapes is not a calibrated Houston network, and controller dispatch contains no feeder feedback.

Synthetic tests vary price, outage duration, alert lead, initial SOC, false/missed alerts, dropout, load uplift and state of health. They include cash effects after an alert expires. The guardrail proposition requires sufficient continuous warning, charging availability, and adequate energy/power; it gives no alert-duty-cycle bound on total margin loss.

`build_paper_results.py` creates paper tables/figures and numerical macros from corrected CSVs, with verification before generation. It copies the compact public report into `artifacts/`. The paper contains no hand-entered experimental headline values. The same generator updates the marked README finding from the verified Houston 2024 rows, sums the 21 recorded training-loop times for the compute note, and records the README hash alongside the paper macros.

Run the latency benchmark after other compute jobs finish; contention changes wall-time measurements. `python scripts/verify_results.py --full` checks the complete main and ancillary outputs before packaging.

Ancillary learned-policy rows identify the selected checkpoint and policy code by hash. Fleet and aging commands preserve compatible reference rows and recompute learned rows when either identity changes; the full verifier rejects stale or unidentified learned results. AC, synthetic, and runtime outputs record the same provenance.

## Rebuild inputs from upstream sources

Use an empty data root for a fresh-source build. These downloads can be large and source services can change. Frozen release inputs are the reference for the paper.

```bash
python scripts/pull_eaglei.py --years 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python scripts/pull_ercot.py --what rtm_spp --years 2019 2020 2021 2022 2023 2024 2025 2026
python scripts/pull_ercot.py --what dam_spp --years 2019 2020 2021 2022 2023 2024 2025 2026
python scripts/pull_weather.py
python scripts/pull_alerts.py
python scripts/pull_resstock.py --n 300 --seed 0
python scripts/pull_feeder.py
python scripts/build_hazard.py
python scripts/build_bundles.py --metros houston dfw san_antonio --years 2019 2020 2021 2022 2023 2024 2025
python scripts/build_bundles.py --metros houston --years 2026 --end 2026-08-15
python scripts/prepare_revision.py
```

The source bundle stage assembles realized physical inputs; `prepare_revision.py` adds prior-year history, completed-hour controller weather, the causal forecast, IDs, and the version/hash manifest. Use only its `data/processed/revised` outputs with revised checkpoints. No API credentials or operator household telemetry are required.

Weather ingestion uses UTC Unix timestamps before converting to America/Chicago, retaining the 23-hour spring day and both copies of the repeated fall hour. Each cache records `utc-unix-v1`; the loaders reject legacy caches without this marker. Physical inputs are rebuilt under `data/processed/physical_utc` rather than reusing the submitted NPZ files at the processed-data root. The `causal-v3` bundles and checkpoints supersede the archived pre-weather-correction revision as well as the workshop submission.

## Build the manuscript

TeX Live with `latexmk`, pdfLaTeX and BibTeX is sufficient:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
python scripts/package_arxiv.py
```

The manuscript uses named authors and the NeurIPS preprint option. `package_arxiv.py` produces the PDF, minimal TeX source archive, and submission metadata under `release/arxiv/`, and rebuilds the source in a clean directory. Uploading to arXiv remains a separate author action.
