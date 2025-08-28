# FER Reproduction: SGD + λ Symmetry Extension

This repo contains a mini-reproduction of the paper  
*Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis*.

## Contents
- `src/` – core source files (train_sgd.py, process_pb.py, cppn.py, etc.)
- `scripts/` – run scripts (`run_all.sh`, metrics scripts)
- `picbreeder_genomes/` – genome input (skull.zip)
- `data/` – key outputs (metrics CSV, LaTeX table, scatter plots)
- `docs/` – report PDF (optional)

## Setup

```bash
git clone https://github.com/rukundob451/fer-repro.git
cd fer-repro
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
