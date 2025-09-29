"""Monte Carlo validation"""
import numpy as np
import pandas as pd

def generate_simulated_panel(n_units=100, n_periods=20, seed=42):
    np.random.seed(seed)
    data = []
    
    for unit in range(n_units):
        treatment_time = np.random.randint(5, 15) if np.random.random() < 0.3 else np.inf
        
        for t in range(n_periods):
            is_treated = 1 if t >= treatment_time else 0
            time_since = max(0, t - treatment_time) if treatment_time != np.inf else 0
            
            data.append({
                'unit_id': unit,
                'time': t,
                'treatment': is_treated,
                'time_since_treatment': time_since,
                'treatment_intensity': np.random.lognormal(0, 0.5) if is_treated else 0,
                'spatial_distance': np.random.exponential(20),
                'outcome': 1 + 0.3 * is_treated + np.random.normal(0, 0.2)
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    data = generate_simulated_panel()
    data.to_csv('data/simulated/test_panel.csv', index=False)
    print(f"Generated {len(data)} observations")
