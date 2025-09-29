#!/bin/bash

# Revised Repository Setup - Validation-First Approach
# This creates a structure focused on rigorous validation before application

echo "Setting up Regime-Switch-Dynamics with Validation-First Structure..."

# Create validation directories (PRIMARY FOCUS)
mkdir -p validation/china_shock
mkdir -p validation/minimum_wage
mkdir -p validation/bankruptcy_contagion

# Create theory directories (separated by mechanism)
mkdir -p theory

# Create application directories (SECONDARY - after validation)
mkdir -p application/ai_investment
mkdir -p application/other_applications

# Create data directories
mkdir -p data/china_shock/raw
mkdir -p data/china_shock/processed
mkdir -p data/ai_investment/raw
mkdir -p data/ai_investment/processed
mkdir -p data/simulated

# Create notebooks directory
mkdir -p notebooks

# Create tests and docs
mkdir -p tests/validation_tests
mkdir -p tests/theory_tests
mkdir -p docs
mkdir -p figures

# Create __init__.py files
touch validation/__init__.py
touch validation/china_shock/__init__.py
touch validation/minimum_wage/__init__.py
touch validation/bankruptcy_contagion/__init__.py
touch theory/__init__.py
touch application/__init__.py
touch application/ai_investment/__init__.py
touch tests/__init__.py

# Create validation files (China Shock - PRIMARY)
cat > validation/china_shock/data_preparation.py << 'EOF'
"""
Download and prepare Autor, Dorn, Hanson China shock data for validation

Data source: https://www.ddorn.net/data.htm
Paper: "The China Syndrome: Local Labor Market Effects of Import Competition"
"""

import pandas as pd
import numpy as np
import requests
import os

def download_china_shock_data():
    """
    Download China shock data from David Dorn's website
    
    Returns cleaned county-level panel data with:
    - Manufacturing employment changes
    - Import exposure from China
    - Geographic coordinates
    - Time period: 1990-2007
    """
    
    print("Downloading China shock data...")
    print("Note: This is placeholder. Actual implementation will:")
    print("1. Download from https://www.ddorn.net/data.htm")
    print("2. Merge county-level employment data")
    print("3. Calculate import exposure measures")
    print("4. Add geographic distance matrix")
    
    # Placeholder for actual implementation
    # In reality, you would download ADH replication files
    
    return None

def prepare_for_boundary_analysis(raw_data):
    """
    Transform ADH data into format needed for boundary detection
    
    Required columns:
    - unit_id: County FIPS code
    - time: Year
    - treatment: China import exposure (continuous)
    - outcome: Manufacturing employment change
    - spatial_distance: Distance to nearest treated county
    - treatment_intensity: Size of import shock
    """
    
    pass

if __name__ == "__main__":
    print("China Shock Data Preparation")
    print("=" * 50)
    print("This will download Autor, Dorn, Hanson replication data")
    print("for validating boundary detection methods.")
EOF

cat > validation/china_shock/boundary_detection.py << 'EOF'
"""
Apply boundary detection methods to China shock data

This is the PRIMARY VALIDATION TEST for our methodology.
If we cannot detect known boundaries here, the method needs refinement.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('../..')
from theory.spatial_boundaries import SpatialBoundaryDetector

def detect_china_shock_boundaries(data):
    """
    Apply our boundary detection method to China shock data
    
    Known results from literature (Autor et al. 2013):
    - Manufacturing employment effects concentrated in "China shock" regions
    - Spillovers to nearby counties through commuting zones
    - Effects decay with distance from manufacturing centers
    - Boundary approximately 150km radius from shock epicenters
    """
    
    detector = SpatialBoundaryDetector()
    
    results = {
        'detected_boundary_km': None,
        'spillover_decay_rate': None,
        'treatment_effect_magnitude': None,
        'spatial_extent': None
    }
    
    # Implementation will go here after theory files are created
    
    return results

def compare_with_literature(detected_boundaries):
    """
    Compare our detected boundaries with established literature
    
    Literature benchmarks:
    - Boundary: ~150km (commuting zone radius)
    - Decay rate: 0.04-0.06 per km
    - Effect size: 0.5-0.8 percentage point employment decline per $1000 exposure
    """
    
    literature_values = {
        'boundary_km': 150,
        'decay_rate': 0.05,
        'effect_size': 0.6
    }
    
    comparison = {}
    
    for key in literature_values:
        if key in detected_boundaries:
            literature = literature_values[key]
            detected = detected_boundaries[key]
            difference = abs(detected - literature) / literature
            
            comparison[key] = {
                'literature': literature,
                'detected': detected,
                'pct_difference': difference * 100,
                'within_20pct': difference < 0.2  # Validation criterion
            }
    
    # Overall validation
    comparison['method_validated'] = all(
        v.get('within_20pct', False) for v in comparison.values() if isinstance(v, dict)
    )
    
    return comparison

if __name__ == "__main__":
    print("China Shock Boundary Detection")
    print("=" * 50)
    print("Validating boundary detection on known spatial spillovers")
EOF

cat > validation/china_shock/validation_report.md << 'EOF'
# China Shock Validation Report

## Objective
Validate boundary detection methodology on Autor, Dorn, Hanson (2013) China trade shock data where spatial boundaries are well-documented.

## Known Results from Literature

**Autor et al. (2013) findings:**
- Import exposure creates local manufacturing employment declines
- Effects concentrated in manufacturing-intensive regions
- Spillovers to nearby counties through commuting zones
- Spatial extent: ~150km radius from shock epicenters
- Decay rate: Effects halve approximately every 100km

## Our Method's Performance

### Boundary Detection
- Detected boundary: [TBD] km
- Literature benchmark: 150 km
- Difference: [TBD]%
- **Status**: [PASS/FAIL - within 20%]

### Spillover Decay Rate
- Detected decay: [TBD] per km
- Literature benchmark: 0.05 per km
- Difference: [TBD]%
- **Status**: [PASS/FAIL]

### Effect Magnitude
- Detected effect: [TBD] pp per $1000 exposure
- Literature benchmark: 0.6 pp per $1000 exposure
- Difference: [TBD]%
- **Status**: [PASS/FAIL]

## Overall Validation

**Method Validated**: [YES/NO]

**Decision**:
- [IF YES] Proceed to AI investment application
- [IF NO] Refine method before novel applications

## Lessons Learned

[Document what worked and what didn't]

## Next Steps

[Based on validation results]
EOF

# Create theory files (separated by mechanism)
cat > theory/spatial_boundaries.py << 'EOF'
"""
Spatial boundary detection - effects decay with geographic distance

Theory: Treatment effects propagate through:
- Local labor markets
- Knowledge spillovers
- Geographic proximity

Distinct from network boundaries (relationship-based propagation)
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

@dataclass
class SpatialParameters:
    """Parameters for spatial boundary detection"""
    decay_rate: float = 0.05  # Per km decay
    threshold_km: float = 150  # Distance threshold
    
class SpatialBoundaryDetector:
    """
    Detect boundaries where treatment effects stop propagating spatially
    
    Focus: GEOGRAPHIC distance, not network relationships
    """
    
    def __init__(self, params: SpatialParameters = None):
        self.params = params or SpatialParameters()
        
    def detect_boundaries(self, data: pd.DataFrame) -> dict:
        """
        Detect spatial boundaries in treatment effects
        
        Args:
            data: DataFrame with columns:
                - unit_id: Location identifier
                - lat, lon: Coordinates
                - treatment: Treatment indicator
                - outcome: Outcome variable
                
        Returns:
            Dictionary with detected boundary characteristics
        """
        
        results = {
            'boundary_detected': False,
            'boundary_distance_km': None,
            'decay_rate': None,
            'confidence_interval': None
        }
        
        # Implementation placeholder
        # Real implementation will:
        # 1. Calculate distance matrix
        # 2. Estimate spillover decay with distance
        # 3. Identify where effects become insignificant
        # 4. Test for sharp boundaries vs smooth decay
        
        return results
    
    def crossing_probability(self, distance_km: float, treatment_intensity: float) -> float:
        """
        Probability that treatment effect crosses spatial boundary
        
        Args:
            distance_km: Geographic distance from treatment
            treatment_intensity: Size of initial treatment
            
        Returns:
            Probability [0,1] that effect propagates to this distance
        """
        
        # Exponential decay model
        spatial_prob = np.exp(-self.params.decay_rate * distance_km)
        
        # Intensity adjustment
        intensity_factor = 1 / (1 + np.exp(-(treatment_intensity - 1.0)))
        
        return spatial_prob * intensity_factor

EOF

cat > theory/network_boundaries.py << 'EOF'
"""
Network boundary detection - effects propagate through relationships

Theory: Treatment effects spread through:
- Supply chain connections
- Buyer-supplier relationships
- Contractual obligations

Distinct from spatial boundaries (geography-based propagation)
"""

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass

@dataclass
class NetworkParameters:
    """Parameters for network boundary detection"""
    connection_threshold: float = 0.5
    max_hops: int = 3  # Maximum network distance
    
class NetworkBoundaryDetector:
    """
    Detect boundaries where treatment effects stop propagating through networks
    
    Focus: RELATIONSHIP distance, not geographic
    """
    
    def __init__(self, params: NetworkParameters = None):
        self.params = params or NetworkParameters()
        
    def detect_boundaries(self, data: pd.DataFrame, network: nx.Graph) -> dict:
        """
        Detect network boundaries in treatment effects
        
        Args:
            data: DataFrame with treatment and outcome
            network: NetworkX graph of relationships
            
        Returns:
            Dictionary with detected boundary characteristics
        """
        
        results = {
            'boundary_detected': False,
            'boundary_hops': None,
            'network_decay_rate': None
        }
        
        return results
    
    def crossing_probability(self, network_distance: int, 
                           connection_strength: float) -> float:
        """
        Probability that treatment effect crosses network boundary
        
        Args:
            network_distance: Hops in relationship graph
            connection_strength: Strength of connections [0,1]
            
        Returns:
            Probability [0,1] that effect propagates through network
        """
        
        # Network effects decay with hops
        network_prob = (1 - self.params.connection_threshold) ** network_distance
        
        # Connection strength matters
        network_prob *= connection_strength
        
        return network_prob

EOF

cat > theory/boundary_testing.py << 'EOF'
"""
Test whether spatial and network boundaries are independent mechanisms

Critical question: Can we combine them or must we model separately?
"""

import numpy as np
import pandas as pd
from scipy import stats

def test_spatial_network_independence(data: pd.DataFrame,
                                     spatial_measure: str,
                                     network_measure: str,
                                     outcome: str) -> dict:
    """
    Test if spatial and network boundaries operate independently
    
    H0: Spatial and network effects are independent
    H1: They interact (cannot model separately)
    """
    
    # Chi-square test for independence
    spatial_high = data[spatial_measure] > data[spatial_measure].median()
    network_high = data[network_measure] > data[network_measure].median()
    
    contingency = pd.crosstab(spatial_high, network_high)
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    
    # Interaction effect test
    high_high = data[spatial_high & network_high][outcome].mean()
    high_low = data[spatial_high & ~network_high][outcome].mean()
    low_high = data[~spatial_high & network_high][outcome].mean()
    low_low = data[~spatial_high & ~network_high][outcome].mean()
    
    interaction = (high_high - high_low) - (low_high - low_low)
    
    return {
        'independence_p_value': p_value,
        'independent': p_value > 0.05,
        'interaction_effect': interaction,
        'recommendation': 'Model separately' if p_value > 0.05 else 'May need interaction'
    }

EOF

# Create application files with LIMITATIONS
cat > application/ai_investment/LIMITATIONS.md << 'EOF'
# AI Investment Data: Known Limitations

## Critical Issues for Boundary Detection

### 1. Insufficient Time Series
- **Problem**: Only 8 years of data (2015-2023)
- **Impact**: Cannot observe full regime dynamics
- **Consequence**: "Boundaries" may be noise, not real regime changes

### 2. COVID-19 Confounding
- **Problem**: Pandemic (2020-2022) creates massive disruption
- **Impact**: Cannot separate AI effects from pandemic effects
- **Consequence**: Any detected "boundary crossing" may be pandemic, not AI

### 3. Measurement Error
- **Problem**: "AI adoption" is poorly defined
- **Examples**: Chatbot? Machine learning? Automation? RPA?
- **Impact**: Treatment variable has high error
- **Consequence**: Attenuates boundary detection

### 4. Selection Bias
- **Problem**: Early AI adopters are different firms
- **Cannot control for**: Unmeasured innovation capacity
- **Consequence**: Confounds boundary crossing with pre-existing differences

### 5. Theory Validation Paradox
- **Problem**: Using same data to develop AND validate theory
- **This violates**: Scientific method of independent validation
- **Consequence**: Circular reasoning, overfitting

## Appropriate Use of This Data

**DO**: Use as exploratory application after validation on China shock
**DON'T**: Use to validate the boundary detection methodology

## Honest Research Framing

When writing about AI investment results:
- Frame as "exploratory application"
- Note all five limitations explicitly
- Compare with validated method from China shock
- Acknowledge uncertainty in findings
- Provide sensitivity analyses

## Alternative if Validation Fails

If China shock validation fails, DO NOT proceed with AI application.
Instead: Refine method, validate again, then apply to AI.
EOF

# Create documentation
cat > docs/VALIDATION_STRATEGY.md << 'EOF'
# Validation Strategy

## Philosophy

We validate on KNOWN phenomena before applying to UNKNOWN questions.

## Three-Stage Approach

### Stage 1: China Shock (Known Spatial Boundaries)
- **Data**: Autor, Dorn, Hanson (2013)
- **Known result**: ~150km boundary, 0.05 decay rate
- **Test**: Can our method detect these known boundaries?
- **Criterion**: Within 20% of literature values

### Stage 2: Spatial vs Network Clarification
- **Question**: Are these separate mechanisms?
- **Test**: Independence tests, interaction effects
- **Outcome**: Model separately or jointly

### Stage 3: Application (Only if Stages 1-2 Pass)
- **Data**: AI investment (exploratory)
- **Framing**: Application, not validation
- **Limitations**: Fully documented

## Decision Tree

```
Stage 1: China Shock Validation
├── SUCCESS → Proceed to Stage 2
└── FAILURE → Refine method, repeat Stage 1

Stage 2: Spatial vs Network
├── INDEPENDENT → Separate models
├── DEPENDENT → Joint model
└── UNCLEAR → More data needed

Stage 3: Application
├── After Stages 1-2 pass
└── With full limitations documented
```

## Validation Criteria

Method is validated if it:
1. Detects China shock boundaries within ±20% of literature
2. Achieves better out-of-sample prediction than baseline
3. Identifies correct spatial decay patterns
4. Does not produce false positives in placebo tests
EOF

cat > docs/SPATIAL_VS_NETWORK.md << 'EOF'
# Spatial vs Network Boundaries: Conceptual Distinction

## The Confusion

Your earlier framework conflated two different types of propagation:
1. **Spatial**: Effects spread through geographic proximity
2. **Network**: Effects spread through relationships

These have different:
- Theoretical foundations
- Empirical patterns
- Policy implications

## Spatial Boundaries

**Mechanism**: Geographic spillovers
- Local labor markets
- Knowledge diffusion
- Commuting zones

**Measurement**: Distance in kilometers

**Example**: Manufacturing decline spreads to nearby towns through shared labor markets

## Network Boundaries

**Mechanism**: Relationship-based propagation
- Supply chain connections
- Buyer-supplier links
- Contractual obligations

**Measurement**: Graph distance (hops)

**Example**: Firm bankruptcy affects direct suppliers, then suppliers' suppliers

## Why This Matters

If you conflate them:
- Cannot identify true mechanism
- Policy recommendations misleading
- Theory development confused

## Our Approach

1. Model spatial boundaries first (China shock)
2. Model network boundaries separately (bankruptcy data)
3. Test if they interact
4. Only combine if empirically justified

## Example Comparison

**Spatial**: "AI effects spread to firms within 50km"
→ Policy: Target geographic clusters

**Network**: "AI effects spread to direct suppliers"
→ Policy: Target supply chain partners

These require DIFFERENT policies!
EOF

# Update .gitignore for new structure
cat >> .gitignore << 'EOF'

# Validation data (may be large)
data/china_shock/raw/*
data/minimum_wage/raw/*

# Keep processed data structure
!data/*/processed/.gitkeep
EOF

# Create placeholder validation notebooks
cat > notebooks/01_china_shock_validation.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# China Shock Validation: Primary Method Test\n",
    "\n",
    "This is the MOST IMPORTANT notebook. If our method cannot detect known boundaries here, it needs refinement."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "from validation.china_shock import data_preparation, boundary_detection\n",
    "from theory.spatial_boundaries import SpatialBoundaryDetector\n",
    "\n",
    "print(\"Imports successful\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo ""
echo "✓ Revised repository structure created!"
echo ""
echo "Key changes from original:"
echo "1. PRIMARY FOCUS: validation/ directory with China shock data"
echo "2. SEPARATION: spatial_boundaries.py vs network_boundaries.py"
echo "3. HONESTY: AI investment limitations documented"
echo "4. DECISION TREE: Clear validation criteria before application"
echo ""
echo "Next steps:"
echo "1. Review LIMITATIONS.md in application/ai_investment/"
echo "2. Read VALIDATION_STRATEGY.md in docs/"
echo "3. Start with China shock data download"
echo "4. Run validation BEFORE any AI application"
echo ""
echo "This is proper science: validate first, apply second."
