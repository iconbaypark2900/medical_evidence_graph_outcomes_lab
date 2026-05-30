"""
Test the Cox regression with a minimal example
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# Create minimal test data 
n = 100
test_data = pd.DataFrame({
    'duration': np.random.exponential(1, n),
    'event': np.random.binomial(1, 0.5, n),
    'var1': np.random.normal(0, 1, n), 
    'var2': np.random.binomial(1, 0.5, n)
})

print("Test data:")
print(test_data.head())
print("\nData types:")
print(test_data.dtypes)

# Create Cox model 
try:
    cph = CoxPHFitter()
    cph.fit(test_data, duration_col='duration', event_col='event')
    print("\nCox model fitted successfully!")
    print("Hazard ratios:", cph.hazard_ratios_)
except Exception as e:
    print(f"\nError in Cox regression: {e}")
    print(f"Error type: {type(e)}")