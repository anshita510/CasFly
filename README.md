# CasFly

CasFly is for real-time tracing of probable precursor event chains across fragmented edge data sources (for example: wearables, bedside monitors, and device-side health logs) without centralizing all raw data.

This repository provides a production-oriented implementation of that workflow:
1. `sdk/` to build TPHGs, run LaVE chain expansion, and serve tracing over HTTP with FastAPI.
2. `evaluation/` and `run_results_reproduction.sh` to reproduce paper-style LaVE vs PCMCI+ analyses.

Typical use:
1. Integrate `sdk/` in your application to trace precursor chains for incoming events in real time.
2. Run the evaluation scripts to validate behavior and compare against baseline methods.

## Folder Structure

- `sdk/` : installable SDK and HTTP service layer
- `evaluation/compare_pcmci_lave_tau.py` : tau-sweep comparison pipeline
- `evaluation/evaluate_predictive_and_hits.py` : predictive + reverse-hits analyses
- `evaluation/requirements-results-repro.txt` : Python deps for reproduction
- `run_results_reproduction.sh` : one-command reproduction launcher
- `RESULTS_REPRODUCTION.md` : detailed reproduction notes

## Quick Start (SDK + API)

```bash
cd sdk
python3 -m pip install -e .
python3 -m pip install -r service/requirements.txt
uvicorn casfly_service.app:app --app-dir service --host 0.0.0.0 --port 8000
```

## Reproduce Results

Expected local inputs:
- `evaluation/device_partitions_patientwise/`  
  Folder of per-device event logs. It must contain `Device*/Patient_*.csv` files used to train/evaluate PCMCI+ and compare against LaVE.
- `evaluation/all_merged_metrics_log.csv`  
  Consolidated LaVE execution log (CSV) containing chain/hop metrics; used to extract normalized LaVE edges for comparison.

Run:

```bash
python3 -m pip install -r evaluation/requirements-results-repro.txt
bash run_results_reproduction.sh
```

Outputs are written under `evaluation/artifacts/repro_paper` by default.

## Notes

- This bundle intentionally excludes large/raw datasets and generated artifacts.
- The paper evaluation section is Synthea-based.

## Citation

You can use GitHub's **Cite this repository** button (powered by `CITATION.cff`) or use:

```bibtex
@software{gupta_misra_casfly_repo_2026,
  title   = {CasFly: SDK, Service, and Reproducibility Pipeline},
  author  = {Gupta, Anshita and Misra, Sudip},
  year    = {2026},
  url     = {https://github.com/anshita510/CasFly},
  version = {v0.1.0}
}
```

Paper citation:

```bibtex
@ARTICLE{11250683,
  author={Gupta, Anshita and Misra, Sudip},
  journal={IEEE Internet of Things Journal},
  title={CasFly: Causal Chain Tracing Across Fragmented Edge Data for IoT Healthcare},
  year={2026},
  volume={13},
  number={3},
  pages={4578-4586},
  keywords={Medical services;Internet of Things;Probabilistic logic;Viterbi algorithm;Heuristic algorithms;Protocols;Heart rate;Real-time systems;Wearable Health Monitoring Systems;Object recognition;Causal chain tracing;decentralized protocols;edge computing;fragmented data;IoT healthcare;probabilistic graphs},
  doi={10.1109/JIOT.2025.3633292}
}
```

IEEE Xplore: `https://ieeexplore.ieee.org/document/11250683/`
