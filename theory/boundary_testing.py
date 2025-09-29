"""
Robust Empirical Strategy for Testing Boundary Connections

This implementation addresses the conceptual challenges in connecting
spillover boundaries and dynamic treatment effects through rigorous
empirical testing rather than theoretical assertion.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import warnings
from typing import Dict, List, Tuple, Optional

class BoundaryConnectionTester:
    """
    Test whether spillover boundaries and dynamic treatment boundaries
    are empirically connected, rather than assuming theoretical unity
    """
    
    def __init__(self):
        self.test_results = {}
        self.validation_metrics = {}
    
    def test_spatial_temporal_independence(self, 
                                         panel_data: pd.DataFrame,
                                         spatial_boundary_var: str,
                                         temporal_boundary_var: str,
                                         outcome_var: str) -> Dict:
        """
        Test H0: Spatial and temporal boundaries are independent
        
        If they are truly unified, we should reject independence.
        If independence holds, the theoretical unification is questionable.
        """
        
        # Create contingency table
        spatial_high = panel_data[spatial_boundary_var] > panel_data[spatial_boundary_var].median()
        temporal_high = panel_data[temporal_boundary_var] > panel_data[temporal_boundary_var].median()
        
        contingency_table = pd.crosstab(spatial_high, temporal_high)
        
        # Chi-square test for independence
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Correlation test
        correlation = panel_data[spatial_boundary_var].corr(panel_data[temporal_boundary_var])
        
        # Effect on outcomes test
        outcome_by_boundaries = panel_data.groupby([spatial_high, temporal_high])[outcome_var].mean()
        
        # Test for interaction effect (evidence of connection)
        high_high = outcome_by_boundaries.loc[(True, True)] if (True, True) in outcome_by_boundaries else np.nan
        high_low = outcome_by_boundaries.loc[(True, False)] if (True, False) in outcome_by_boundaries else np.nan
        low_high = outcome_by_boundaries.loc[(False, True)] if (False, True) in outcome_by_boundaries else np.nan
        low_low = outcome_by_boundaries.loc[(False, False)] if (False, False) in outcome_by_boundaries else np.nan
        
        # Interaction effect: (HH - HL) - (LH - LL)
        if not (np.isnan([high_high, high_low, low_high, low_low]).any()):
            interaction_effect = (high_high - high_low) - (low_high - low_low)
        else:
            interaction_effect = np.nan
        
        return {
            'independence_test': {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'reject_independence': p_value < 0.05,
                'interpretation': 'Connected' if p_value < 0.05 else 'Independent'
            },
            'correlation': correlation,
            'interaction_effect': interaction_effect,
            'contingency_table': contingency_table.to_dict(),
            'recommendation': self._interpret_independence_test(p_value, correlation, interaction_effect)
        }
    
    def _interpret_independence_test(self, p_value: float, correlation: float, interaction: float) -> str:
        """Provide interpretation of independence test results"""
        
        if p_value < 0.05 and abs(correlation) > 0.2 and abs(interaction) > 0.1:
            return "Strong evidence for boundary connection - proceed with unified framework"
        elif p_value < 0.05 and abs(correlation) > 0.1:
            return "Moderate evidence for connection - develop conditional unified framework"
        elif p_value >= 0.05 and abs(correlation) < 0.1:
            return "Little evidence for connection - treat boundaries separately"
        else:
            return "Mixed evidence - requires deeper theoretical development"
    
    def test_multiplicative_vs_additive_effects(self,
                                              panel_data: pd.DataFrame,
                                              spatial_boundary_indicator: str,
                                              temporal_boundary_indicator: str,
                                              outcome_var: str) -> Dict:
        """
        Test whether boundary crossing effects are multiplicative or additive
        
        Multiplicative: ATE(both) = ATE(spatial) × ATE(temporal)
        Additive: ATE(both) = ATE(spatial) + ATE(temporal)
        """
        
        # Create boundary crossing groups
        data = panel_data.copy()
        data['spatial_cross'] = data[spatial_boundary_indicator] == 1
        data['temporal_cross'] = data[temporal_boundary_indicator] == 1
        data['both_cross'] = data['spatial_cross'] & data['temporal_cross']
        data['neither_cross'] = ~data['spatial_cross'] & ~data['temporal_cross']
        
        # Calculate effects for each group
        baseline = data[data['neither_cross']][outcome_var].mean()
        
        effects = {}
        for group in ['spatial_cross', 'temporal_cross', 'both_cross']:
            group_data = data[data[group]]
            if len(group_data) > 0:
                effects[group] = group_data[outcome_var].mean() - baseline
            else:
                effects[group] = np.nan
        
        # Test multiplicative vs additive
        if not np.isnan(list(effects.values())).any():
            predicted_additive = effects['spatial_cross'] + effects['temporal_cross']
            predicted_multiplicative = effects['spatial_cross'] * effects['temporal_cross']
            observed_joint = effects['both_cross']
            
            additive_error = abs(observed_joint - predicted_additive)
            multiplicative_error = abs(observed_joint - predicted_multiplicative)
            
            # Which model fits better?
            better_fit = 'additive' if additive_error < multiplicative_error else 'multiplicative'
            
            return {
                'effects': effects,
                'predicted_additive': predicted_additive,
                'predicted_multiplicative': predicted_multiplicative,
                'observed_joint': observed_joint,
                'additive_error': additive_error,
                'multiplicative_error': multiplicative_error,
                'better_fit': better_fit,
                'effect_magnitude': abs(additive_error - multiplicative_error),
                'theoretical_implication': self._interpret_effect_structure(better_fit, effects)
            }
        else:
            return {'error': 'Insufficient data for all boundary combinations'}
    
    def _interpret_effect_structure(self, better_fit: str, effects: Dict) -> str:
        """Interpret the structure of boundary effects"""
        
        if better_fit == 'multiplicative':
            return ("Effects compound multiplicatively - supports unified boundary theory. "
                   "Spatial and temporal boundaries amplify each other.")
        else:
            return ("Effects are additive - suggests boundaries operate independently. "
                   "Question unified framework, consider separate mechanisms.")
    
    def validate_boundary_predictions(self,
                                    panel_data: pd.DataFrame,
                                    boundary_model,
                                    outcome_var: str,
                                    test_size: float = 0.3) -> Dict:
        """
        Out-of-sample validation: Do boundary crossing probabilities predict outcomes?
        """
        
        # Split data
        n_train = int(len(panel_data) * (1 - test_size))
        train_data = panel_data.iloc[:n_train]
        test_data = panel_data.iloc[n_train:]
        
        # Fit boundary model on training data (simplified example)
        X_train = train_data[['treatment_intensity', 'spatial_distance', 'time_since_treatment']].fillna(0)
        y_train = train_data[outcome_var].fillna(0)
        
        X_test = test_data[['treatment_intensity', 'spatial_distance', 'time_since_treatment']].fillna(0)
        y_test = test_data[outcome_var].fillna(0)
        
        # Baseline model: traditional variables only
        baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_r2 = baseline_model.score(X_test, y_test)
        
        # Enhanced model: include boundary crossing probabilities
        # (This would use your actual boundary model)
        # For demonstration, create mock boundary probabilities
        np.random.seed(42)
        train_data_enhanced = X_train.copy()
        test_data_enhanced = X_test.copy()
        
        # Mock boundary crossing probabilities (replace with actual model)
        train_data_enhanced['spatial_boundary_prob'] = np.random.beta(2, 3, len(X_train))
        train_data_enhanced['temporal_boundary_prob'] = np.random.beta(3, 2, len(X_train))
        train_data_enhanced['joint_boundary_prob'] = (
            train_data_enhanced['spatial_boundary_prob'] * 
            train_data_enhanced['temporal_boundary_prob']
        )
        
        test_data_enhanced['spatial_boundary_prob'] = np.random.beta(2, 3, len(X_test))
        test_data_enhanced['temporal_boundary_prob'] = np.random.beta(3, 2, len(X_test))
        test_data_enhanced['joint_boundary_prob'] = (
            test_data_enhanced['spatial_boundary_prob'] * 
            test_data_enhanced['temporal_boundary_prob']
        )
        
        # Enhanced model with boundary features
        enhanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
        enhanced_model.fit(train_data_enhanced, y_train)
        enhanced_pred = enhanced_model.predict(test_data_enhanced)
        enhanced_r2 = enhanced_model.score(test_data_enhanced, y_test)
        
        # Improvement from boundary features
        r2_improvement = enhanced_r2 - baseline_r2
        
        # Feature importance
        feature_names = list(train_data_enhanced.columns)
        feature_importance = dict(zip(feature_names, enhanced_model.feature_importances_))
        
        return {
            'baseline_r2': baseline_r2,
            'enhanced_r2': enhanced_r2,
            'r2_improvement': r2_improvement,
            'percentage_improvement': (r2_improvement / abs(baseline_r2) * 100) if baseline_r2 != 0 else np.nan,
            'feature_importance': feature_importance,
            'validation_status': 'Boundary features helpful' if r2_improvement > 0.01 else 'Limited improvement',
            'recommendation': self._interpret_validation_results(r2_improvement, feature_importance)
        }
    
    def _interpret_validation_results(self, improvement: float, importance: Dict) -> str:
        """Interpret validation results"""
        
        boundary_importance = (
            importance.get('spatial_boundary_prob', 0) + 
            importance.get('temporal_boundary_prob', 0) + 
            importance.get('joint_boundary_prob', 0)
        )
        
        if improvement > 0.05 and boundary_importance > 0.2:
            return "Strong validation - boundary features substantially improve prediction"
        elif improvement > 0.01 and boundary_importance > 0.1:
            return "Moderate validation - boundary features provide some improvement"
        else:
            return "Weak validation - reconsider boundary framework utility"
    
    def robustness_checks(self,
                         panel_data: pd.DataFrame,
                         boundary_results: Dict,
                         n_bootstrap: int = 1000) -> Dict:
        """
        Robustness checks for boundary analysis
        """
        
        robustness = {}
        
        # Bootstrap confidence intervals
        if 'interaction_effect' in boundary_results:
            bootstrap_effects = []
            n_obs = len(panel_data)
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                boot_sample = panel_data.sample(n=n_obs, replace=True)
                
                # Recalculate interaction effect
                # (Simplified - would use full boundary calculation)
                try:
                    spatial_high = boot_sample['spatial_distance'] < boot_sample['spatial_distance'].median()
                    temporal_high = boot_sample['time_since_treatment'] > boot_sample['time_since_treatment'].median()
                    
                    outcome_groups = boot_sample.groupby([spatial_high, temporal_high])['outcome'].mean()
                    
                    if len(outcome_groups) == 4:
                        hh = outcome_groups.loc[(True, True)]
                        hl = outcome_groups.loc[(True, False)]
                        lh = outcome_groups.loc[(False, True)]
                        ll = outcome_groups.loc[(False, False)]
                        
                        interaction = (hh - hl) - (lh - ll)
                        bootstrap_effects.append(interaction)
                except:
                    continue
            
            if bootstrap_effects:
                robustness['bootstrap_interaction'] = {
                    'mean': np.mean(bootstrap_effects),
                    'std': np.std(bootstrap_effects),
                    'ci_lower': np.percentile(bootstrap_effects, 2.5),
                    'ci_upper': np.percentile(bootstrap_effects, 97.5),
                    'significant': not (np.percentile(bootstrap_effects, 2.5) <= 0 <= np.percentile(bootstrap_effects, 97.5))
                }
        
        # Alternative specifications
        robustness['alternative_thresholds'] = {}
        for threshold in [0.25, 0.5, 0.75]:
            # Test boundary definitions at different thresholds
            # (Implementation would vary based on specific boundary definitions)
            robustness['alternative_thresholds'][f'threshold_{threshold}'] = {
                'threshold_value': threshold,
                'placeholder': 'Would test boundary sensitivity to threshold choice'
            }
        
        return robustness

def run_empirical_validation_example():
    """
    Example of how to use the robust empirical strategy
    """
    
    print("Robust Empirical Strategy for Boundary Analysis")
    print("=" * 50)
    
    # Simulate data for testing
    np.random.seed(42)
    n_obs = 1000
    
    data = pd.DataFrame({
        'treatment_intensity': np.random.lognormal(0, 0.5, n_obs),
        'spatial_distance': np.random.exponential(20, n_obs),
        'time_since_treatment': np.random.gamma(2, 2, n_obs),
        'spatial_boundary_indicator': np.random.binomial(1, 0.3, n_obs),
        'temporal_boundary_indicator': np.random.binomial(1, 0.4, n_obs),
    })
    
    # Create outcome with interaction effect
    data['outcome'] = (1 + 
                      0.3 * data['spatial_boundary_indicator'] +
                      0.4 * data['temporal_boundary_indicator'] +
                      0.2 * data['spatial_boundary_indicator'] * data['temporal_boundary_indicator'] +
                      np.random.normal(0, 0.3, n_obs))
    
    # Initialize tester
    tester = BoundaryConnectionTester()
    
    # Test 1: Independence
    independence_test = tester.test_spatial_temporal_independence(
        data, 'spatial_distance', 'time_since_treatment', 'outcome'
    )
    print(f"Independence Test Results:")
    print(f"  Recommendation: {independence_test['recommendation']}")
    print(f"  Chi-square p-value: {independence_test['independence_test']['p_value']:.4f}")
    print(f"  Correlation: {independence_test['correlation']:.3f}")
    
    # Test 2: Effect structure
    effect_structure = tester.test_multiplicative_vs_additive_effects(
        data, 'spatial_boundary_indicator', 'temporal_boundary_indicator', 'outcome'
    )
    
    if 'better_fit' in effect_structure:
        print(f"\nEffect Structure Test:")
        print(f"  Better fit: {effect_structure['better_fit']}")
        print(f"  Theoretical implication: {effect_structure['theoretical_implication']}")
    
    print("\nValidation complete. Use these results to refine theoretical framework.")

if __name__ == "__main__":
    run_empirical_validation_example()
