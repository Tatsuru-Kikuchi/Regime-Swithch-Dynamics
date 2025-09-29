# Regime Switch Dynamics: A Framework for Boundary Detection in Treatment Effects

**Author**: Tatsuru Kikuchi  
**Repository**: https://github.com/Tatsuru-Kikuchi/Regime-Switch-Dynamics  
**Status**: Active Development  
**Last Updated**: 2025

## Overview

This repository develops theoretical and empirical methods for detecting **regime boundaries** in treatment effects. We address a fundamental question in econometrics: when do treatment effects transition between regimes—from local to systemic, temporary to persistent, or isolated to contagious?

### Core Research Questions

1. **When** do treatment effects cross from one regime to another?
2. **Where** in space/networks do these boundaries occur?
3. **How** do regime transitions happen (gradual vs. sudden)?
4. **Why** do some treatments trigger regime changes while others don't?

### Key Innovation

We develop **Boundary Crossing Probability (BCP)** indicators that extend traditional event studies and difference-in-differences methods by explicitly modeling regime transitions. This allows researchers to:

- Detect when local effects become systemic
- Identify spatial/temporal boundaries in treatment propagation
- Quantify spillover effects across units
- Improve policy targeting and timing

## Project Status: Critical Assessment

**Important Notice**: This project is in early development. The theoretical connection between spatial spillovers and dynamic treatment effects is **hypothesis to be tested**, not established fact. We are proceeding with empirical validation first, theoretical elaboration second.

### Current Priorities

1. ✅ **Phase 1**: Test empirically whether boundary connections exist (Weeks 1-4)
2. ⏳ **Phase 2**: Develop validated methodology (Weeks 5-8)
3. 🔜 **Phase 3**: Apply to empirical questions (Weeks 9-12)

## Repository Structure

```
Regime-Switch-Dynamics/
├── theory/                              # Theoretical framework
│   ├── boundary_crossing_theory.py      # Core BCP implementation
│   ├── boundary_testing.py              # Empirical validation tests
│   ├── spatial_boundaries.py            # Spatial spillover detection
│   ├── temporal_boundaries.py           # Dynamic regime detection
│   └── monte_carlo_validation.py        # Statistical properties
├── empirical/                           # Empirical applications
│   ├── ai_investment/                   # Primary application
│   │   ├── boundary_analysis.py         
│   │   ├── executive_heterogeneity.py   
│   │   └── industry_transformation.py   
│   └── validation_studies/              # Method comparison
├── data/                                # Data management
│   ├── simulated/                       # Monte Carlo data
│   └── processed/                       # Cleaned empirical data
├── notebooks/                           # Jupyter notebooks
│   ├── 01_theoretical_introduction.ipynb
│   ├── 02_empirical_validation.ipynb
│   └── 03_ai_investment_application.ipynb
├── tests/                               # Unit and integration tests
├── docs/                                # Documentation
│   ├── methodology_guide.md
│   └── api_reference.md
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/Tatsuru-Kikuchi/Regime-Switch-Dynamics.git
cd Regime-Switch-Dynamics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/
```

## Quick Start

### 1. Test Boundary Connections (Start Here)

Before developing unified boundary theory, test whether spatial and temporal boundaries are empirically connected:

```python
from theory.boundary_testing import BoundaryConnectionTester
import pandas as pd

# Load your data
data = pd.read_csv('data/processed/your_panel_data.csv')

# Initialize tester
tester = BoundaryConnectionTester()

# Test if boundaries are connected
results = tester.test_spatial_temporal_independence(
    data,
    spatial_boundary_var='spatial_distance',
    temporal_boundary_var='time_since_treatment',
    outcome_var='outcome'
)

print(results['recommendation'])
# Output: "Strong evidence for boundary connection" or 
#         "Little evidence for connection - treat boundaries separately"
```

### 2. Calculate Boundary Crossing Probabilities

If empirical tests support connection, calculate joint probabilities:

```python
from theory.boundary_crossing_theory import (
    BoundaryCrossingProbability, 
    BoundaryParameters
)

# Set parameters
params = BoundaryParameters(
    spatial_decay=0.05,
    temporal_persistence=2.0,
    intensity_threshold=1.2
)

# Initialize model
bcp_model = BoundaryCrossingProbability(params)

# Calculate crossing probabilities
crossing_probs = bcp_model.joint_crossing_probability(
    distances=data['spatial_distance'].values,
    time_since_treatment=data['time_since_treatment'].values,
    treatment_intensity=data['treatment_intensity'].values
)

# Add to dataframe
data['boundary_crossing_prob'] = crossing_probs
```

### 3. Boundary-Aware Difference-in-Differences

Extend traditional DiD to account for boundary crossing:

```python
from theory.boundary_crossing_theory import BoundaryAwareDiD

# Initialize estimator
did_estimator = BoundaryAwareDiD(bcp_model)

# Estimate treatment effects
results = did_estimator.estimate_boundary_aware_did(
    panel_data=data,
    outcome_var='productivity',
    treatment_var='ai_investment',
    unit_id_col='firm_id',
    time_col='year'
)

print(f"Standard DiD: {results['standard_did']['coefficient']:.3f}")
print(f"Boundary-adjusted: {results['boundary_adjusted_coefficient']:.3f}")
print(f"Adjustment: {results['boundary_adjustment_percent']:.1f}%")
```

## Research Strategy

### Phase 1: Empirical Validation (Current Priority)

**Goal**: Determine empirically whether boundary unification is justified

**Tasks**:
1. Test independence of spatial and temporal boundaries
2. Compare multiplicative vs. additive effect structures
3. Validate out-of-sample prediction performance
4. Conduct robustness checks

**Decision Point**: Proceed with unified framework only if empirical evidence supports it

### Phase 2: Methodology Development

**If boundaries ARE connected**:
- Develop joint boundary crossing probability theory
- Create unified event study framework
- Establish identification conditions

**If boundaries are INDEPENDENT**:
- Develop separate spatial and temporal indicators
- Focus on stronger empirical mechanism
- Study sequential boundary crossing

### Phase 3: Empirical Applications

**Primary Application**: AI Investment and Firm Productivity
- Research question: When do AI investments trigger persistent productivity gains?
- Data: Japanese firm-level data (2015-2023)
- Key variables: AI investment amounts, CEO characteristics, productivity measures

**Secondary Applications** (choose one):
- Urban aging and concentration in Japan
- Financial crisis propagation

## Key Components

### 1. Boundary Crossing Probability (BCP)

Core theoretical quantity measuring likelihood of regime transition:

```
P(Boundary Cross | X, D, t) = P_spatial × P_temporal × P_intensity
```

**Components**:
- `P_spatial`: Spatial spillover probability
- `P_temporal`: Dynamic persistence probability  
- `P_intensity`: Treatment intensity effect

### 2. New Event Study Indicators

**Spillover Boundary Indicator (SBI)**:
```
SBI_it = 1{P(Spatial Cross) > threshold}
```

**Regime Transition Probability (RTP)**:
```
RTP_it = Γ(t; α, β) × sigmoid(intensity - threshold)
```

**Dynamic Treatment Intensity (DTI)**:
```
DTI_it = treatment_intensity × P(Boundary Cross)
```

### 3. Boundary-Aware DiD

Decompose treatment effects by boundary crossing status:

```
ATE = ATE_no_crossing × (1 - BCP) + ATE_crossing × BCP
```

## Expected Contributions

### Theoretical
- First framework connecting spatial spillovers and dynamic treatment effects
- New class of boundary-aware treatment effect estimators
- Testable theory of regime transitions in treatment effects

### Empirical
- Extension of AI investment-productivity analysis with causal mechanisms
- Identification of optimal investment thresholds and timing
- Executive heterogeneity in technology adoption effectiveness

### Policy
- Early warning indicators for regime transitions
- Improved targeting of interventions accounting for spillovers
- Timing recommendations for policy implementation

## Warnings and Limitations

### Current Limitations
1. **Untested theoretical connections**: Spatial-temporal boundary unity requires empirical validation
2. **Identification challenges**: Boundary crossing may be endogenous
3. **Small sample concerns**: Method requires substantial data for boundary detection
4. **Computational intensity**: Monte Carlo validation is time-consuming

### Research Integrity
We are committed to:
- Testing assumptions empirically before theoretical elaboration
- Reporting null results if boundaries are not empirically connected
- Comparing with traditional methods to establish practical significance
- Transparency about limitations and alternative interpretations

## Contributing

This repository is part of ongoing dissertation research. External contributions are welcome but please contact the author first to discuss coordination.

### Development Priorities
1. Complete empirical validation tests
2. Apply to AI investment data
3. Monte Carlo validation of statistical properties
4. Documentation and testing improvements

## Citation

If you use this code, please cite:

```bibtex
@misc{kikuchi2025regime,
  author = {Kikuchi, Tatsuru},
  title = {Regime Switch Dynamics: A Framework for Boundary Detection in Treatment Effects},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Tatsuru-Kikuchi/Regime-Switch-Dynamics}
}
```

## Related Papers

1. Kikuchi, T. (2024). "AI Investment and Firm Productivity: How Executive Demographics Drive Technology Adoption and Performance in Japanese Enterprises." arXiv:2508.03757

2. Kikuchi, T. (2025). "Stochastic Boundaries in Spatial General Equilibrium: A Diffusion-Based Approach to Causal Inference with Spillover Effects." Working Paper.

## Contact

**Tatsuru Kikuchi**  
Email: [your email]  
Repository: https://github.com/Tatsuru-Kikuchi/Regime-Switch-Dynamics

## License

MIT License - See LICENSE file for details

## Acknowledgments

This research benefits from discussions with [advisors/colleagues]. All errors are my own.

---

**Last Updated**: January 2025  
**Next Milestone**: Complete Phase 1 empirical validation by [date]
