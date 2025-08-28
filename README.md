# FER Reproduction + λ Symmetry Extension

Reproduction of the SGD baseline from *Questioning Representational Optimism in Deep Learning* using CPPNs on the PicBreeder “skull”, plus a mirror-consistency regularizer (λ). We sweep λ ∈ {0.0, 0.05, 0.2, 0.5} and report reconstruction (MSE) vs. symmetry.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
