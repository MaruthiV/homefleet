# Changes from the workshop submission

The original submitted source, PDF, configurations, and outputs are preserved locally under `archives/submitted-2026-08-31`, with a file-hash manifest. The corrected release uses protocol `causal-v3`. Submitted outputs are not pooled with corrected outputs.

## Scientific corrections

- Historical weather is requested as UTC Unix timestamps and converted to America/Chicago afterward. The previous ISO-clock ingestion used a fixed API offset as though it followed daylight saving time, shifting winter observations by one hour. Protocol `causal-v3` rebuilds weather, synthetic PV, county-risk models, all replay bundles, and all trained policies; earlier corrected-input results remain archived and are not relabelled.
- Slot forecasts initialize from preceding history rather than from the evaluation episode. Reserve quantiles exclude incomplete and future target-year energy windows.
- Weather features use completed hours. The outage-risk model fits 2014–2019 and calibrates on 2020, replacing future-weather proxy inputs and post-Uri calibration.
- PPO observations respect day-ahead price publication times. The same operating minimum is used for learned and risk-adaptive optimization controllers. Optimization targets include the physical minimum when translating usable backup energy into a state-of-charge requirement.
- The hazard-threshold target retains the shared 10% operating minimum even when its historical backup-need estimate is near zero. The final replay revision is `reserve-minimum-v3`; a regression test covers the former 5% target in this case.
- Raising a reserve now forces feasible charging toward it. Unattainable instantaneous targets are reported as recovery shortfall, separately from physical or reachable-floor violations.
- The alert override acts on the affected county's homes. It does not clamp unaffected counties when another county has an alert.
- Training and scoring use the same flicker-merged five-hour promise window. Retail revenue excludes unserved demand, and terminal inventory is valued consistently in training and reporting.
- Regular evaluation windows partition the year. Named stress events are evaluated separately and do not receive extra weight in annualized comparisons.
- Regular partitions are identified by numeric `wNN` labels. A named event such as `wind_mar_2025` is excluded from annual summaries even though its name begins with `w`; the year-total check and regression test prevent that duplication.
- Fixed penalties are identified as Lagrangian relaxations, not per-home budget enforcement. The optimization shortfall cost is described as a surrogate. The hazard-threshold rule is not presented as a proved chance constraint.
- The rolling discretized lookahead controller is no longer called a proved global upper bound.
- The throughput aging proxy now normalizes charge plus discharge energy by twice rated capacity, following the source model. All reported replays use the corrected expression directly; preliminary outputs are archived separately.

## Guardrail correction

The alert-duty-cycle bound on total profit loss is removed. An alert changes stored energy and may change later decisions after it ends. A conditional energy-and-power sufficiency proposition states connection, availability, continuous lead, usable capacity, and power requirements. Charging from the physical minimum takes up to four hours across the included presets, not the previously stated 3.3 hours.

The alert analysis distinguishes an alert active at onset from any alert in the preceding day. It includes every evaluated county rather than the first county of each metro. Synthetic tests include short and missing warnings, false alerts, low initial energy, cold-load uplift, reduced state of health, and communication dropout. Complete-episode economics include behavior after an alert ends.

## Reproducibility and presentation

The ignore rule that hid the source data loaders is anchored to the raw data directory. The package includes a small individual-home example and expected numerical outputs, geographic helper snapshots, and bundled configurations. The real-time price downloader, prerequisite metadata handling, paper reproduction entry points, and artifact manifests are part of the public release. Unimplemented solver stubs are removed.

The public paper uses named authors and the NeurIPS preprint style, with a standard layout and public GitHub link. The author block follows the authors' published [arXiv:2607.24667](https://arxiv.org/html/2607.24667v1): Maruthi Vemula (UNC Chapel Hill) followed by Neeraj Praneeth Gajula (NYU).

The AC table generator's percent-sign escaping is corrected. Figures identify units, reference policies, uncertainty, and the preferable direction. The data card distinguishes observed county/market records from simulated homes and PV. The paper's interpretation is conditional on the replay model and does not claim validation of grid dynamics or operator deployment.
