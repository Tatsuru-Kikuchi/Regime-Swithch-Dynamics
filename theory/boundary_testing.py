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

