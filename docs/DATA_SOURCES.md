# Data sources

This repository contains code only: no raw data, no per-row derived tables, no out-of-fold vectors, no predictions, no trained models.
Each dataset below is downloaded by the script in its row into `data/`, which is git-ignored.

## 1. Competition data (not redistributed)
| file | content | md5 |
|---|---|---|
| `training_*.parquet` | 2025 movements with flight-plan (`_flt`) and movement (`_mvt`) columns | md5 per file from the organisers' download |
| `ranking.parquet`, `submitting.parquet` | public period, Jan + Jul 2026 | same |
| `final_ranking.parquet`, `final_submitting.parquet` | final blind period, Jan + Feb + Jun + Jul 2026, re-issued 8 Oct 2026 11:04 UTC | `1ce0c6ee9ea15b8c3d0cd282591d4e0a` / `2c1f407296e217fd7d242c9d51ba27ad` (`docs/REPRODUCE.md` §8) |

Registered teams get the files from the OpenSky Network MinIO store `s3.opensky-network.org` (`scripts/download_data.py`; team
credentials from `PRC_ACCESS_KEY` / `PRC_SECRET_KEY` or the pyopensky `secret.conf`, neither part of this repository). The challenge
states no licence for them. Asked about reuse (Discord, 26 Sep 2026), the organiser replied "ask the owner for the license": `_flt`
columns come from the EUROCONTROL Network Manager, `_mvt` columns from the airports. Nothing from these files is published here.

## 2. External open data used by the final model
| # | dataset | used for | URL | licence / terms | pulled | script |
|---|---|---|---|---|---|---|
| 1 | **adsb.lol globe_history** 2025 + 2026 (readsb `trace_full` per airframe), release tag `planes-readsb-prod-0` (fallback `-staging-0`), cut to boxes around the 10 airports | ground clocks per departure (tables adsb, adsb2, adsb3, adsb5, adsb6, adsbL), stand positions, the unmatched-row callsign link | <https://github.com/adsblol/globe_history_2025>, <https://github.com/adsblol/globe_history_2026>, docs <https://www.adsb.lol/docs/open-data/historical/> | **ODbL 1.0**; feeder contributions CC0 | 19–21 Sep 2026 (2025, Jan/Jul 2026); 8 Oct 2026 (Feb/Jun 2026) | `adsb_all.sh` → `adsb_pull.py`, `adsb_extract{,2,3}.py`, `adsb_segment_extract.py`, `adsb_pushstart_extract.py`, `adsb_longstop_extract.py`, `build_adsb*_features.py`, `build_stand_pos.py` |
| 2 | **EUROCONTROL PRU performance data**, 10 airports, 2018 – Aug 2026: daily all / ATC pre-departure delay, airport traffic, ATFM slot adherence, arrival ATFM delay by cause | daily stacker inputs (`xpr`), daily calibration (B) | <https://ansperformance.eu/data/> (csv <https://www.eurocontrol.int/performance/data/download/csv/>) | copy allowed with EUROCONTROL named as source, non-commercial, no modification of the documents (<https://ansperformance.eu/about/disclaimer/>). Only aggregates used, nothing redistributed | 8, 24, 25 Sep; 8 Oct 2026 (daily to 31 Aug 2026) | `fetch_pru.py` → `pru_build.py`; `pru_daily_calib.py` |
| 3 | **OPDI** v0.0.2 (EUROCONTROL Open Performance Data Initiative): flight list and flight events, Jan 2022 – Jul 2026 | groups `opdi`, `opdir` | <https://www.opdi.aero/>, <https://www.eurocontrol.int/performance/data/download/OPDI/v002/> | EUROCONTROL open data: attribution, non-commercial | 7 Sep; 8 Oct 2026 (Feb–Jun 2026) | `fetch_opdi.py` → `build_opdi_features.py` |
| 4 | **Iowa Environmental Mesonet ASOS/METAR**, hourly, 10 airports | group `wx` | <https://mesonet.agron.iastate.edu/request/download.phtml> | free public data, attribution requested | 5 Sep 2026; extended to 8 Oct 2026 (identical on the overlap) | `download_metar.py`, `build_metar.py`, `metar_features.py` |
| 5 | **OpenFlights airlines.dat** (IATA ↔ ICAO designators) | fallback designator map for the callsign link (28 of 13,160 links) | <https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat> | **ODbL 1.0**, © OpenFlights.org contributors | 27 Sep 2026, md5 `ba4e6c7f794489d7ef6075b7dba6c90f` | `unm_link.py` |

## 3. Tested, not used by the final model
| # | dataset | status | URL | licence | script |
|---|---|---|---|---|---|
| 6 | **EUROCONTROL Taxi-Out Additional Time**, monthly per airport (`taxi_out_additional_time_{2025,2026}.csv`) | allowed for the final phase by organiser ruling (8 Oct 2026, including the scored months). Tested as a per-airport-month level shift: fold gate **failed** (4-fold mean ΔMSE +20.7 s²). Not used  | <https://www.eurocontrol.int/performance/data/download/csv/> | PRU terms as row 2 | analysis only |
| 7 | **OpenStreetMap** (taxiways, runways, stands via Overpass) | tested as the `geo` feature group; `scripts/build_features.py` adds it only when `data/external/stand_runway_distance.parquet` exists, and no final member or Kumo feature list reads it | <https://overpass-api.de/> | **ODbL 1.0**, © OpenStreetMap contributors | not included (not used by the final model) |
| 8 | **OurAirports** runway thresholds | as row 7 | <https://ourairports.com/data/> | public domain | not included (not used by the final model) |
| 9 | **EUROCONTROL Aviation Intelligence Unit**, daily airport punctuality | tested as calibration and stacker inputs (aiu_calib, AIU_PUNCTUALITY); not in the final line | <https://www.eurocontrol.int/Economics/DailyPunctuality-Airports.html> | copy allowed with the EUROCONTROL notice and disclaimer; no modification | not included (not used by the final model) |
| 10 | **X-Plane Scenery Gateway** `apt.dat` (stands, taxi network) | tested (xplane_stands); not in the final line | <https://gateway.x-plane.com/> | GNU GPL v2+ | not included (not used by the final model) |
| 11 | adsb.lol alternate release tags (`-staging-0tmp`, `-prod-0tmp`) for 22 missing or empty 2025 days | gate failed; the final model keeps the gaps | as row 1 | ODbL 1.0 | – |
| 12 | PRC Data Challenge 2024 flight list (README and a 2 MB sample) | read for analysis; no model uses it. Its trajectory files are OpenSky state vectors and were not touched | <https://doi.org/10.4121/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4> | CC BY 4.0 | – |

## 4. Organiser rulings relied on
| date (2026) | ruling |
|---|---|
| 1 Sep | proprietary or paid models behind an API: not allowed |
| 3 and 11 Sep | OpenSky state vectors / trajectories and the OpenSky historical database (Trino) are not external open data for this challenge |
| 6 Sep | OPDI allowed |
| 9 Sep | "open source data are allowed if declared in the documentation"; artefact rows are scored as exported |
| 17–19 Sep | adsb.lol is open data; extracting off-block times from open trajectory data is allowed |
| 24–25 Sep | any openly published data under an open licence is allowed (covers the PRU aggregates) |
| 1–2 Oct | data or model without an explicit open licence is not open |
| 4 Oct | OPDI confirmed as open data for prize eligibility |
| 8 Oct | EUROCONTROL monthly taxi-out file: "still ok" for the final phase (row 6) |



## 5. Model weights
NVIDIA **Kumo Tabular** (`nvidia/Kumo-Tabular`, tag v1.0.0, `medium/regressor.pt` sha256 `daa6a174…fb725`), licence **OpenMDW-1.1**.
Downloaded by the user, run locally, not redistributed (`docs/licences.md`). No proprietary model and no model behind an API is used.

## 6. Attribution
- Contains information from **adsb.lol**, made available under the **Open Database License (ODbL) 1.0**.
- Source: **EUROCONTROL** Performance Review Unit (ansperformance.eu) and the Open Performance Data Initiative (OPDI).
- Weather data: **Iowa Environmental Mesonet**, Iowa State University.
- Airline designators: **OpenFlights.org** contributors, ODbL 1.0.
- Map data (tested only): © **OpenStreetMap** contributors, ODbL 1.0; runway data: **OurAirports** (public domain).

**ODbL share-alike.** We publish no derived database. The per-flight clock tables (adsb.lol), the designator map (OpenFlights) and the
geometry (OSM) stay local; only the code that builds them is here. Anyone who publishes those tables must release them under the ODbL.

## 7. Not used
OpenSky Network state vectors and historical database; commercial or registration-gated sources (DDR2, BADA, ADRR, paid APIs);
proprietary pre-trained models behind an API.
