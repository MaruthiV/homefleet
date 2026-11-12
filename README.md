# HomeFleet-ERCOT

Code and public replay data for **Adaptive Reliability Reserves for Residential Battery Fleets**, by Maruthi Vemula and Neeraj Praneeth Gajula.

[Paper PDF](paper/main.pdf) · [Paper source](paper/main.tex) · [Reproduce the paper](docs/REPRODUCIBILITY.md) · [Data card](docs/DATA_CARD.md) · [Releases](https://github.com/MaruthiV/homefleet/releases)

HomeFleet studies how residential batteries can trade electricity while retaining energy for household outages. It compares fixed reserves, alert charging, historical backup tiers, risk-conditioned optimization, and learned power/reserve decisions. All controllers use the same battery and tariff accounting and paired outage draws.

The replay combines **observed ERCOT prices and county outage records with simulated ResStock household loads and synthetic PV**. Household interruptions are synthesized from county fractions. It contains no battery-company customer telemetry. Dispatch takes prices as given; a separate static feeder audit examines voltages and line loading.

<!-- GENERATED_HEADLINE_START -->

In the corrected Houston 2024 replay, the threshold 0.1 reserve raises simulated backup coverage from **75.0% to 86.5%** and adds **USD 10.2 in modeled margin per home-year** (paired 95% interval: USD 1.7–21.7). Coverage means serving the first five hours or until earlier restoration. The paired interval for the coverage gain includes zero, and the 2025 replay yields higher margin with lower observed coverage. These are conditional simulation comparisons.

Across the 10 promise-cost PPO runs in that replay, backup coverage ranges from **31.1% to 96.3%**; **10 of 10** have lower modeled margin than fixed 20%. [The complete results table](artifacts/summary.csv) retains the full values and intervals.

<!-- GENERATED_HEADLINE_END -->

## Run the included example

Python **3.12** is required. No data download, API key, or PyTorch installation is needed for this example.

```bash
git clone https://github.com/MaruthiV/homefleet.git
cd homefleet
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m homefleet.demo --check
```

This runs two days of the Houston Beryl replay for eight simulated homes, comparing a 20% fixed reserve with a hazard-threshold reserve. The package includes the preceding history needed by causal forecasts. Output reports modeled margin, five-hour backup promise rate, unserved energy, and constraint checks, followed by `Reference results match.` Higher margin and promise rate are preferable; lower unserved energy is preferable. The example's weekly margin is a short-episode rate.

## Reproduce the paper

The corrected named preprint uses **v0.2.0 / causal-v3**. Its [revision notes](docs/REVISION_NOTES.md) explain substantive changes from the workshop submission, including causal information, charging behavior, metric accounting, and the guardrail claim.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/download_data.py --group bundles
python scripts/download_data.py --group raw
python scripts/download_data.py --group checkpoints
python scripts/download_data.py --group results
python scripts/verify_data.py
python scripts/verify_results.py --full
python scripts/summarize_revision.py
python scripts/build_paper_results.py
python -m pytest -q
```

Downloads are checked against the release's [SHA-256 manifest](artifacts/release_manifest.json). Bundles include individual simulated household time series and building IDs. Results and checkpoints can be inspected without retraining. The [reproducibility guide](docs/REPRODUCIBILITY.md) gives the complete training/evaluation grid, exact seed selection, source-data rebuild, sensitivity analyses, and PDF build. It also describes the tested environment and separate BLAST-Lite setup.

To rerun the main experiments from frozen inputs:

```bash
python scripts/reproduce.py baselines --workers 3
python scripts/reproduce.py train --workers 4
python scripts/reproduce.py policies --workers 4
```

These are resumable CPU sweeps: 21 training runs and both guarded and unguarded evaluations. They take substantially longer than the included example. Configuration and progress are recorded under `experiments/revised/`. Policy selection uses Houston 2024 only; named event windows and chronological tests do not select the checkpoint.

<!-- GENERATED_COMPUTE_START -->

Reference training used an ARM64 CPU workstation with 11 logical processors and 18 GiB RAM, up to 6 concurrent training workers, and 1 PyTorch thread per run. Recorded training-loop wall times sum to **15.3 hours across all 21 runs**, excluding startup. This sum differs from elapsed time when runs execute in parallel. The checkpoint archive retains each run's configuration and training log; inference/evaluation timings are reported separately.

<!-- GENERATED_COMPUTE_END -->

## What the benchmark measures

- **Economics:** modeled gross margin with an explicit terminal inventory adjustment, plus seasonal comparisons, cash-only and daily tail metrics. It excludes capital, hedging, subscriptions, ancillary services, and degradation charges.
- **Backup:** the fraction of simulated outage episodes served through their first five hours or earlier restoration, plus unserved energy and per-home outcomes.
- **Transfer and failure:** chronological and metro changes, named storms, fleet size/composition/dropout, and synthetic joint price/outage events.
- **System consequences:** static AC feeder violations and sensitivity to battery-aging model choice.

Annualized results combine nonoverlapping episodes that each start at 90% charge. They are replay rates rather than continuous-year company profits. County risk forecasts are not household outage probabilities. The 2026 split has prices through August 15 but no observed outage truth, so it supports market comparisons only. An alert override requires sufficient warning and available charging; it is not an unconditional reliability guarantee. See the paper and data card for assumptions.

## Repository map

| Path | Contents |
| --- | --- |
| `src/homefleet/data/` | ERCOT, EAGLE-I, ResStock, weather, alert loaders and assembly |
| `src/homefleet/sim/`, `env.py` | Battery, tariff, PV and household outage replay |
| `src/homefleet/control/` | Forecasts, independent-home dynamic programs and priority allocation |
| `src/homefleet/rl/` | PPO training and evaluation |
| `configs/` | Hardware, tariff, event splits and frozen revision grid |
| `scripts/` | Downloads, sweeps, audits, aggregation and manuscript builds |
| `artifacts/` | Compact reports, experiment coverage and release checksums |
| `paper/` | Named preprint source, generated figures/tables and PDF |
| `tests/` | Causality, accounting, feasibility, solver parity and aggregation checks |

## Citation and reuse

```bibtex
@misc{vemula2026homefleet,
  title = {Adaptive Reliability Reserves for Residential Battery Fleets},
  author = {Vemula, Maruthi and Gajula, Neeraj Praneeth},
  year = {2026},
  howpublished = {Preprint and research code, version 0.2.0},
  url = {https://github.com/MaruthiV/homefleet}
}
```

[CITATION.cff](CITATION.cff) supplies machine-readable software citation. An arXiv identifier will be added after submission; none is assigned here.

Authored code is [MIT licensed](LICENSE). Third-party inputs and software retain their own terms; see [NOTICE](NOTICE) and source links in the data card. Please report reproducibility issues through [GitHub Issues](https://github.com/MaruthiV/homefleet/issues), including the release, command, Python version, and relevant error output.
