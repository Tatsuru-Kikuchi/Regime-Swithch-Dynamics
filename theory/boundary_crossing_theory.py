"""
Core Boundary Crossing Theory for Regime Switch Dynamics

This module implements the theoretical foundation connecting:
1. Spillover boundaries (spatial/network propagation)
2. Dynamic treatment effects (temporal regime changes)
3. New empirical indicators for event studies

Author: Tatsuru Kikuchi
Repository: https://github.com/Tatsuru-Kikuchi/Regime-Swithch-Dynamics
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Callable, Union
import warnings

@dataclass
class BoundaryParameters:
    """Core parameters governing boundary crossing dynamics"""
    spatial_decay: float = 0.1          # Rate of spatial effect decay
    temporal_persistence: float = 0.8   # AR(1) coefficient for persistence  
    intensity_threshold: float = 1.5    # Minimum treatment intensity for boundary crossing
    network_amplification: float = 1.3  # Network effect multiplier
    volatility_scaling: float = 0.5     # Boundary uncertainty parameter

class BoundaryCrossingProbability:
    """
    Core class for Boundary Crossing Probability (BCP) calculation
    
    The BCP unifies spatial spillover boundaries and temporal regime boundaries
    through a joint probability framework:
    
    P(Boundary Cross | X, D, t) = P_spatial × P_temporal × P_intensity
    
    This enables detection of when/where/how treatment effects transition
    between regimes (local → spillover, temporary → persistent, etc.)
    """
    
    def __init__(self, params: BoundaryParameters):
        self.params = params
        self.fitted_parameters = {}
        
    def spatial_crossing_probability(self, 
                                   distances: np.ndarray,
                                   network_centrality: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Probability that treatment effect crosses spatial boundary
        
        P_spatial = exp(-λ * distance) × (1 + γ * centrality)
        
        Args:
            distances: Spatial/economic distances to other units
            network_centrality: Network centrality measures (optional)
            
        Returns:
            Spatial boundary crossing probabilities [0,1]
        """
        
        # Base spatial decay
        spatial_prob = np.exp(-self.params.spatial_decay * distances)
        
        # Network amplification
        if network_centrality is not None:
            network_effect = 1 + self.params.network_amplification * network_centrality
            spatial_prob *= network_effect
            
        # Ensure probabilities in [0,1]
        return np.clip(spatial_prob, 0, 1)
    
    def temporal_crossing_probability(self,
                                    time_since_treatment: np.ndarray,
                                    treatment_intensity: np.ndarray,
                                    volatility: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Probability that treatment effect crosses temporal boundary (becomes persistent)
        
        P_temporal = Γ(t; α, β) × sigmoid(intensity - threshold)
        
        Args:
            time_since_treatment: Time elapsed since treatment start
            treatment_intensity: Magnitude of initial treatment
            volatility: Treatment effect volatility (optional)
            
        Returns:
            Temporal boundary crossing probabilities [0,1]
        """
        
        # Gamma distribution for temporal evolution (most effects appear with delay)
        alpha, beta = 2.0, self.params.temporal_persistence
        temporal_component = stats.gamma.cdf(time_since_treatment, a=alpha, scale=beta)
        
        # Intensity threshold effect (sigmoid to ensure smooth transition)
        intensity_component = 1 / (1 + np.exp(-(treatment_intensity - self.params.intensity_threshold)))
        
        # Volatility adjustment (higher volatility → lower crossing probability)
        if volatility is not None:
            volatility_adjustment = np.exp(-self.params.volatility_scaling * volatility)
            temporal_component *= volatility_adjustment
            
        return np.clip(temporal_component * intensity_component, 0, 1)
    
    def joint_crossing_probability(self,
                                 distances: np.ndarray,
                                 time_since_treatment: np.ndarray, 
                                 treatment_intensity: np.ndarray,
                                 network_centrality: Optional[np.ndarray] = None,
                                 volatility: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Joint boundary crossing probability combining spatial and temporal components
        
        P(Boundary Cross) = P_spatial × P_temporal
        
        This is the core theoretical quantity that unifies spillover boundaries
        and dynamic treatment effects.
        """
        
        # Calculate component probabilities
        p_spatial = self.spatial_crossing_probability(distances, network_centrality)
        p_temporal = self.temporal_crossing_probability(time_since_treatment, 
                                                      treatment_intensity, volatility)
        
        # Joint probability (independence assumption - can be relaxed)
        joint_prob = p_spatial * p_temporal
        
        return joint_prob
    
    def estimate_parameters(self,
                          panel_data: pd.DataFrame,
                          outcome_var: str,
                          treatment_var: str,
                          distance_var: str,
                          time_var: str,
                          method: str = 'mle') -> Dict:
        """
        Estimate boundary parameters from data using maximum likelihood
        
        The likelihood function is constructed based on observed regime transitions:
        - High boundary crossing probability → larger treatment effects
        - Low boundary crossing probability → smaller/temporary effects
        """
        
        def neg_log_likelihood(params_vector):
            """Negative log-likelihood for parameter estimation"""
            
            # Unpack parameters
            spatial_decay, temporal_persistence, intensity_threshold = params_vector
            
            # Update parameters temporarily
            temp_params = BoundaryParameters(
                spatial_decay=spatial_decay,
                temporal_persistence=temporal_persistence, 
                intensity_threshold=intensity_threshold,
                network_amplification=self.params.network_amplification,
                volatility_scaling=self.params.volatility_scaling
            )
            
            # Calculate boundary crossing probabilities
            bcp = self.joint_crossing_probability(
                panel_data[distance_var].values,
                panel_data[time_var].values,
                panel_data[treatment_var].values
            )
            
            # Likelihood: effects should be larger when BCP is high
            effects = panel_data[outcome_var].values
            
            # Model: E[Y | BCP] = β₀ + β₁ × BCP
            # Higher BCP → higher expected outcome
            expected_effects = np.mean(effects) + bcp * np.std(effects)
            
            # Gaussian likelihood
            residuals = effects - expected_effects
            log_likelihood = -0.5 * np.sum(residuals**2) / np.var(residuals)
            log_likelihood -= 0.5 * len(effects) * np.log(2 * np.pi * np.var(residuals))
            
            return -log_likelihood
        
        # Optimize parameters
        initial_params = [self.params.spatial_decay, 
                         self.params.temporal_persistence,
                         self.params.intensity_threshold]
        
        bounds = [(0.01, 1.0),    # spatial_decay
                 (0.1, 5.0),     # temporal_persistence  
                 (0.1, 10.0)]    # intensity_threshold
        
        try:
            result = optimize.minimize(neg_log_likelihood, initial_params, 
                                     bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                self.fitted_parameters = {
                    'spatial_decay': result.x[0],
                    'temporal_persistence': result.x[1], 
                    'intensity_threshold': result.x[2],
                    'log_likelihood': -result.fun,
                    'convergence': True
                }
                
                # Update parameters
                self.params.spatial_decay = result.x[0]
                self.params.temporal_persistence = result.x[1]
                self.params.intensity_threshold = result.x[2]
                
            else:
                self.fitted_parameters = {'convergence': False, 'message': result.message}
                
        except Exception as e:
            self.fitted_parameters = {'convergence': False, 'error': str(e)}
            
        return self.fitted_parameters

class SpilloverBoundaryIndicators:
    """
    New empirical indicators for boundary crossing in event studies
    
    These indicators extend traditional event study methodology by providing
    quantitative measures of when/where/how regime changes occur.
    """
    
    def __init__(self, bcp_model: BoundaryCrossingProbability):
        self.bcp_model = bcp_model
        
    def spillover_boundary_indicator(self, panel_data: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Binary indicator for spillover boundary crossing
        
        SBI_it = 1{P(Spatial Boundary Cross) > threshold}
        
        This indicator identifies units/times where treatment effects
        are likely to propagate to other units.
        """
        
        spatial_probs = self.bcp_model.spatial_crossing_probability(
            panel_data['spatial_distance'].values,
            panel_data.get('network_centrality', None)
        )
        
        return pd.Series(spatial_probs > threshold, index=panel_data.index, name='SBI')
    
    def regime_transition_probability(self, panel_data: pd.DataFrame) -> pd.Series:
        """
        Continuous probability of regime transition
        
        RTP_it = P(Temporal Boundary Cross | treatment_intensity, time_since_treatment)
        
        This measures how likely treatment effects are to become persistent.
        """
        
        temporal_probs = self.bcp_model.temporal_crossing_probability(
            panel_data['time_since_treatment'].values,
            panel_data['treatment_intensity'].values,
            panel_data.get('volatility', None)
        )
        
        return pd.Series(temporal_probs, index=panel_data.index, name='RTP')
    
    def dynamic_treatment_intensity(self, panel_data: pd.DataFrame) -> pd.Series:
        """
        Treatment intensity adjusted for boundary crossing probability
        
        DTI_it = treatment_intensity × P(Boundary Cross)
        
        This provides a more accurate measure of "effective" treatment intensity
        accounting for regime dynamics.
        """
        
        joint_probs = self.bcp_model.joint_crossing_probability(
            panel_data['spatial_distance'].values,
            panel_data['time_since_treatment'].values,
            panel_data['treatment_intensity'].values,
            panel_data.get('network_centrality', None),
            panel_data.get('volatility', None)
        )
        
        dynamic_intensity = panel_data['treatment_intensity'] * joint_probs
        return pd.Series(dynamic_intensity, index=panel_data.index, name='DTI')
    
    def boundary_crossing_sequence(self, 
                                  panel_data: pd.DataFrame,
                                  unit_id_col: str,
                                  time_col: str) -> pd.DataFrame:
        """
        Track boundary crossing sequence for each unit over time
        
        Returns panel with boundary crossing indicators and their evolution.
        """
        
        results = []
        
        for unit_id in panel_data[unit_id_col].unique():
            unit_data = panel_data[panel_data[unit_id_col] == unit_id].sort_values(time_col)
            
            # Calculate indicators over time for this unit
            sbi = self.spillover_boundary_indicator(unit_data)
            rtp = self.regime_transition_probability(unit_data)
            dti = self.dynamic_treatment_intensity(unit_data)
            
            # Track cumulative boundary crossings
            cumulative_spatial = sbi.cumsum()
            cumulative_temporal = (rtp > 0.5).cumsum()  # Binary version of RTP
            
            unit_results = unit_data.copy()
            unit_results['SBI'] = sbi
            unit_results['RTP'] = rtp  
            unit_results['DTI'] = dti
            unit_results['cumulative_spatial_crossings'] = cumulative_spatial
            unit_results['cumulative_temporal_crossings'] = cumulative_temporal
            unit_results['total_boundary_crossings'] = cumulative_spatial + cumulative_temporal
            
            results.append(unit_results)
        
        return pd.concat(results, ignore_index=True)

class BoundaryAwareDiD:
    """
    Difference-in-Differences estimation with boundary crossing adjustments
    
    Extends traditional DiD by:
    1. Separating effects that cross boundaries from those that don't
    2. Adjusting for spillover effects between treated and control units
    3. Modeling dynamic treatment effect evolution
    """
    
    def __init__(self, bcp_model: BoundaryCrossingProbability):
        self.bcp_model = bcp_model
        self.results = {}
        
    def estimate_boundary_aware_did(self,
                                  panel_data: pd.DataFrame,
                                  outcome_var: str,
                                  treatment_var: str,
                                  unit_id_col: str,
                                  time_col: str,
                                  cluster_se: bool = True) -> Dict:
        """
        Estimate treatment effects accounting for boundary crossing
        
        The model decomposes total effects:
        ATE = ATE_no_crossing + ATE_boundary_crossing
        """
        
        # Calculate boundary crossing indicators
        indicators = SpilloverBoundaryIndicators(self.bcp_model)
        enhanced_data = indicators.boundary_crossing_sequence(panel_data, unit_id_col, time_col)
        
        # Separate observations by boundary crossing status
        crossing_data = enhanced_data[enhanced_data['SBI'] | (enhanced_data['RTP'] > 0.5)]
        no_crossing_data = enhanced_data[~(enhanced_data['SBI'] | (enhanced_data['RTP'] > 0.5))]
        
        results = {
            'total_observations': len(enhanced_data),
            'crossing_observations': len(crossing_data),
            'no_crossing_observations': len(no_crossing_data),
            'boundary_crossing_rate': len(crossing_data) / len(enhanced_data) if len(enhanced_data) > 0 else 0
        }
        
        # Estimate effects for each regime
        if len(crossing_data) > 10:  # Minimum observations for estimation
            crossing_effect = self._estimate_did_subsample(crossing_data, outcome_var, treatment_var)
            results['boundary_crossing_effect'] = crossing_effect
        else:
            results['boundary_crossing_effect'] = {'coefficient': np.nan, 'std_error': np.nan}
            
        if len(no_crossing_data) > 10:
            no_crossing_effect = self._estimate_did_subsample(no_crossing_data, outcome_var, treatment_var)  
            results['no_crossing_effect'] = no_crossing_effect
        else:
            results['no_crossing_effect'] = {'coefficient': np.nan, 'std_error': np.nan}
        
        # Overall boundary-adjusted effect
        if not (np.isnan(results['boundary_crossing_effect']['coefficient']) or 
                np.isnan(results['no_crossing_effect']['coefficient'])):
            
            # Weighted average by observation counts
            w_crossing = len(crossing_data) / len(enhanced_data)
            w_no_crossing = len(no_crossing_data) / len(enhanced_data)
            
            boundary_adjusted_coeff = (w_crossing * results['boundary_crossing_effect']['coefficient'] +
                                     w_no_crossing * results['no_crossing_effect']['coefficient'])
            
            results['boundary_adjusted_coefficient'] = boundary_adjusted_coeff
            results['spillover_premium'] = (results['boundary_crossing_effect']['coefficient'] - 
                                          results['no_crossing_effect']['coefficient'])
        
        # Compare with standard DiD
        standard_did = self._estimate_did_subsample(enhanced_data, outcome_var, treatment_var)
        results['standard_did'] = standard_did
        
        if 'boundary_adjusted_coefficient' in results:
            results['boundary_adjustment_magnitude'] = abs(results['boundary_adjusted_coefficient'] - 
                                                         standard_did['coefficient'])
            results['boundary_adjustment_percent'] = (results['boundary_adjustment_magnitude'] / 
                                                    abs(standard_did['coefficient']) * 100 
                                                    if standard_did['coefficient'] != 0 else np.nan)
        
        self.results = results
        return results
    
    def _estimate_did_subsample(self, data: pd.DataFrame, outcome_var: str, treatment_var: str) -> Dict:
        """Simplified DiD estimation for subsample"""
        
        # Simple before-after comparison for treated vs control
        # This would be replaced with proper econometric estimation in practice
        
        treated = data[data[treatment_var] == 1]
        control = data[data[treatment_var] == 0]
        
        if len(treated) == 0 or len(control) == 0:
            return {'coefficient': np.nan, 'std_error': np.nan, 'observations': len(data)}
        
        # Simple difference in means (placeholder for proper DiD)
        treatment_effect = treated[outcome_var].mean() - control[outcome_var].mean()
        pooled_std = np.sqrt((treated[outcome_var].var() / len(treated)) + 
                           (control[outcome_var].var() / len(control)))
        
        return {
            'coefficient': treatment_effect,
            'std_error': pooled_std,
            'observations': len(data),
            't_statistic': treatment_effect / pooled_std if pooled_std > 0 else np.nan
        }

# Example usage and testing
def demonstrate_boundary_framework():
    """Demonstrate the boundary crossing framework with simulated data"""
    
    print("Boundary Crossing Framework for Regime Switch Dynamics")
    print("=" * 60)
    
    # Initialize parameters and model
    params = BoundaryParameters(
        spatial_decay=0.05,
        temporal_persistence=2.0,
        intensity_threshold=1.2,
        network_amplification=1.5,
        volatility_scaling=0.3
    )
    
    bcp_model = BoundaryCrossingProbability(params)
    
    # Simulate panel data
    np.random.seed(42)
    n_units, n_periods = 100, 20
    
    panel_data = []
    for i in range(n_units):
        for t in range(n_periods):
            treatment_start = np.random.randint(5, 15) if np.random.random() < 0.3 else np.inf
            is_treated = 1 if t >= treatment_start else 0
            time_since = max(0, t - treatment_start) if treatment_start != np.inf else 0
            
            panel_data.append({
                'unit_id': i,
                'time': t,
                'treatment': is_treated,
                'treatment_intensity': np.random.lognormal(0.5, 0.3) if is_treated else 0,
                'spatial_distance': np.random.exponential(20),
                'network_centrality': np.random.beta(2, 8),
                'time_since_treatment': time_since,
                'outcome': 1 + 0.4 * is_treated + np.random.normal(0, 0.2),
                'volatility': np.random.exponential(0.1)
            })
    
    df = pd.DataFrame(panel_data)
    
    # Demonstrate boundary crossing probabilities
    treated_data = df[df['treatment'] == 1]
    
    if len(treated_data) > 0:
        joint_probs = bcp_model.joint_crossing_probability(
            treated_data['spatial_distance'].values,
            treated_data['time_since_treatment'].values,
            treated_data['treatment_intensity'].values,
            treated_data['network_centrality'].values
        )
        
        print(f"Boundary Crossing Probability Statistics:")
        print(f"  Mean: {np.mean(joint_probs):.3f}")
        print(f"  Std:  {np.std(joint_probs):.3f}")
        print(f"  High crossing rate (>0.6): {np.mean(joint_probs > 0.6):.2%}")
    
    # Demonstrate boundary indicators
    indicators = SpilloverBoundaryIndicators(bcp_model)
    enhanced_df = indicators.boundary_crossing_sequence(df, 'unit_id', 'time')
    
    print(f"\nBoundary Indicators:")
    print(f"  Spillover boundary crossings: {enhanced_df['SBI'].sum()}")
    print(f"  High regime transition prob (>0.5): {(enhanced_df['RTP'] > 0.5).sum()}")
    print(f"  Average dynamic treatment intensity: {enhanced_df['DTI'].mean():.3f}")
    
    # Demonstrate boundary-aware DiD
    did_estimator = BoundaryAwareDiD(bcp_model)
    did_results = did_estimator.estimate_boundary_aware_did(
        df, 'outcome', 'treatment', 'unit_id', 'time'
    )
    
    print(f"\nBoundary-Aware DiD Results:")
    print(f"  Standard DiD coefficient: {did_results['standard_did']['coefficient']:.3f}")
    if 'boundary_adjusted_coefficient' in did_results:
        print(f"  Boundary-adjusted coefficient: {did_results['boundary_adjusted_coefficient']:.3f}")
        print(f"  Boundary adjustment: {did_results['boundary_adjustment_percent']:.1f}%")
    print(f"  Boundary crossing rate: {did_results['boundary_crossing_rate']:.2%}")

if __name__ == "__main__":
    demonstrate_boundary_framework()
