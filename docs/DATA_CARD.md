# HomeFleet ERCOT data card

HomeFleet is a replay benchmark assembled from observed market and outage data and simulated household profiles. It contains no Base Power customer telemetry. The release identifies each sampled ResStock building and includes individual load and PV time series, county fields, forecasts, and reproducible controller inputs.

## Sources and transformations

| Input | Source and version | Transformation |
| --- | --- | --- |
| Real-time settlement prices | [ERCOT MIS](https://www.ercot.com/mp/data-products), yearly RTMLZHBSPP archives, report 13061, 2019–2026 | Parse delivery date, hour, interval, and repeated-hour flag; remove duplicate settlement-point rows; retain USD/MWh. |
| Day-ahead settlement prices | ERCOT yearly DAMLZHBSPP archives, report 13060 | Hourly prices carried forward to quarter-hour delivery intervals. Access is restricted by publication time in the controller. |
| County interruptions | [ORNL EAGLE-I v4](https://doi.org/10.6084/m9.figshare.24237376.v4), 2014–2025, CC BY 4.0 | Texas county customers out divided by the fixed MCC modeled-customer denominator; fractions clipped to [0,1]. |
| Home loads | [NREL ResStock 2024 AMY2018 release 2](https://data.openei.org/submissions/4520) | Sample single-family homes in six counties; convert interval-ending EST-without-DST timestamps to interval-start America/Chicago and kWh/quarter-hour to kW. Reuse the 2018 local-calendar profile across years. |
| Weather | [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api), downloaded through August 15, 2026, [CC BY 4.0](https://open-meteo.com/en/licence) | Hourly county-centroid model/reanalysis weather used as a proxy for observations. Revised controller features use completed hours only. |
| Alerts | [Iowa Environmental Mesonet NWS VTEC archive](https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml) | County and zone products mapped to FIPS using the included zone-county snapshot. Active intervals are reconstructed from archived issue/expiry fields. |
| PV | HomeFleet synthesis from irradiance, temperature, and a fixed seed | Approximately 30% adoption, 5–9 kW DC, temperature derating, 14% losses, 96% inverter efficiency. These are simulated outputs. |
| Feeder | [NREL SMART-DS](https://data.openei.org/submissions/2981), Austin P1U, p1uhs0_1247--p1udt2142 | Static AC feeder evaluation with simulated home net demands substituted for load powers. No feeder feedback is used in dispatch. |

The helper snapshots `county_centroids.json` and `zone_county_map.json` retain the exact geographic mappings used by the source build. They are public geographic metadata, not household coordinates. The warning crosswalk is a single April 2026 snapshot applied to all historical years; historical zone-boundary changes can affect warning matching.

## Geographic coverage

Each pool contains 300 simulated homes. Houston uses Harris County (48201), Dallas–Fort Worth uses Dallas (48113), Tarrant (48439), Collin (48085), and Denton (48121), and San Antonio uses Bexar (48029). Larger fleets resample these pools with replacement. They do not add distinct source buildings.

## Bundle schema

Let T be the number of quarter-hour timestamps, N=300 homes, C the number of counties, and K=5 forecast horizons.

| Field | Shape | Units and meaning |
| --- | --- | --- |
| `index` | T | UTC nanoseconds; converted to America/Chicago for decisions. |
| `price_rt`, `price_da` | T | USD/MWh; realized settlement and day-ahead delivery prices. |
| `load_kw`, `pv_kw` | T × N | Simulated gross demand and PV power, kW. |
| `home_id`, `home_fips`, `home_kw_dc` | N | ResStock identifier, county FIPS, and synthesized PV rating. |
| `fips_list` | C | County order for all county-level arrays. |
| `frac` | T × C | Observed county outage fraction, with the missing-data convention below. |
| `alert` | T × C | Public qualifying alert active at that time. |
| `weather` | T × C × 7 | Completed-hour temperature, wind speed, gust, precipitation, irradiance, dew point, and pressure. Column names are in `weather_cols`. |
| `hazard` | T × C × K | Calibrated probability of county fraction reaching 1% within the next horizon. |
| `hazard_h` | K | Horizons in hours: 1, 3, 6, 12, 24. |
| `evaluation_start` | scalar | First index eligible for regular-year evaluation after preceding history. |
| `protocol` | scalar | `causal-v3` for corrected data. |

Revised bundles include the previous year's final 28 days when available. The prefix supplies history; it is excluded from the target year's summaries. The 2019 data has no 2018 price prefix, so regular evaluation begins after 28 days. The 2026 bundle ends on August 15 and has no observed outage truth. Its target-year outage field is a zero-filled simulator placeholder; the preceding-history prefix retains 2025 observations. The 2026 reliability score is unavailable.

Weather is requested as UTC Unix seconds and converted to America/Chicago afterward, retaining daylight-saving transitions. Temperature and dew point are in degrees Celsius, wind speed and gusts in km/h, precipitation in mm over the preceding hour, shortwave radiation in W/m² averaged over the preceding hour, and surface pressure in hPa. The physical PV synthesis interpolates hourly weather to quarter-hours; controller weather features use the completed-hour lag described below. Each corrected cache carries the `utc-unix-v1` clock marker.

## Controller information

The outage-risk models fit 2014–2019 and calibrate on 2020. Weather and reported outage features are delayed to completed hours. Future realized weather is not treated as a forecast. Published day-ahead prices and trailing 28-day slot medians form price forecasts. Reserve requirements use complete historical energy windows strictly before episode start, never the full target-year load profile.

These timing rules define the replay's information set; they do not verify the publication vintage of every input. Open-Meteo historical weather combines model/reanalysis products rather than providing a contemporaneous station feed. The [IEM archive documentation](https://mesonet.agron.iastate.edu/info/datasets/vtec.html) also explains that event records retain their last known state and may be corrected. The benchmark uses the frozen retrospective records with stated lags, not a replay of every as-issued operational message. Operational performance under actual reporting delays and revisions remains untested.

Individual home interruptions are synthesized from county fractions with one uniform draw per home and county event. County-event probabilities are not household failure probabilities. EAGLE-I missing observations are filled with zero in replay; this can understate outages during reporting gaps. The fixed MCC denominator does not resolve changes in coverage. The benchmark therefore supports comparisons conditional on a specified simulator and recorded inputs, not estimates of actual provider customer reliability.

## Versioning and reuse

The bundle manifest records file hashes, home IDs, timestamp ranges, and hazard-model settings. Exact paper inputs and checkpoint versions are separate from scripts that download potentially updated upstream archives. Third-party datasets retain their source-specific terms and attribution; the repository's MIT license applies to authored code. See [NOTICE](../NOTICE).
