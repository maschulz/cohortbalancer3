# CohortBalancer3

Statistical matching for causal inference from observational data.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python CI](https://github.com/maschulz/cohortbalancer3/actions/workflows/ci.yml/badge.svg)](https://github.com/maschulz/cohortbalancer3/actions/workflows/ci.yml)

## Installation

Via pip:
```bash
% FIXME: This is not working yet.
pip install cohortbalancer3
```

From source:
```bash
git clone https://github.com/maschulz/cohortbalancer3.git
cd cohortbalancer3
pip install -e .
```

## Core Concepts

CohortBalancer3 implements three essential components:

1. **MatcherConfig**: Defines matching parameters
2. **Matcher**: Executes the matching algorithm
3. **MatchResults**: Contains matched data and diagnostics

## Tutorial

### Basic Matching

```python
import pandas as pd
from cohortbalancer3 import Matcher, MatcherConfig

# 1. Prepare your data with treatment/control indicators and covariates
data = pd.DataFrame({
    'treatment': [1, 1, 1, 0, 0, 0, 0, 0],
    'age': [45, 55, 35, 65, 40, 52, 38, 60],
    'bmi': [28, 32, 24, 30, 22, 29, 26, 31],
    'outcome': [120, 142, 118, 145, 110, 125, 115, 135]
})

# 2. Configure the matcher
config = MatcherConfig(
    treatment_col='treatment',  # Treatment indicator (1=treated, 0=control)
    covariates=['age', 'bmi'],  # Variables to balance
    outcomes=['outcome']        # Optional: for treatment effect estimation
)

# 3. Perform matching
matcher = Matcher(data, config)
matcher.match()
results = matcher.get_results()

# 4. Examine results
matched_data = results.matched_data
print(f"Original data: {len(data)} observations")
print(f"Matched data: {len(matched_data)} observations")

# 5. Assess balance
balance_stats = results.balance_statistics
print(f"Mean standardized difference before: {balance_stats['smd_before'].mean():.4f}")
print(f"Mean standardized difference after: {balance_stats['smd_after'].mean():.4f}")

# 6. Estimate treatment effects
effects = results.effect_estimates
print(f"Estimated effect: {effects['effect'].values[0]:.4f}")
print(f"95% CI: [{effects['ci_lower'].values[0]:.4f}, {effects['ci_upper'].values[0]:.4f}]")
print(f"p-value: {effects['p_value'].values[0]:.4f}")
```

### Creating a Report

```python
from cohortbalancer3 import create_report

report_path = create_report(
    results,
    method_name="Basic Matching Example",
    output_dir="./output",
    report_filename="matching_report.html"
)
print(f"Report saved to: {report_path}")
```

### Visualization

```python
from cohortbalancer3.visualization import plot_balance, plot_propensity_distributions

# Plot standardized mean differences
balance_plot = plot_balance(results)
balance_plot.savefig("balance.png")

# Plot propensity score distributions
prop_plot = plot_propensity_distributions(results)
prop_plot.savefig("propensity.png")
```

## Configuration Options

```python
# Complete configuration example
config = MatcherConfig(
    # Required parameters
    treatment_col='treatment',       # Column indicating treatment (1) or control (0)
    covariates=['age', 'bmi', 'sex'], # Variables to balance
    
    # Matching method
    match_method='greedy',           # 'greedy' (faster) or 'optimal' (better quality)
    distance_method='propensity',    # 'propensity', 'mahalanobis', or 'euclidean'
    
    # Matching constraints
    caliper=0.2,                     # Maximum allowed distance (or 'auto')
    exact_match_cols=['sex'],        # Variables requiring exact matches
    ratio=1.0,                       # Controls per treatment (1.0 = 1:1 matching)
    replace=False,                   # Allow reuse of control units
    
    # Propensity score settings
    estimate_propensity=True,        # Estimate propensity scores
    propensity_col=None,             # Pre-computed propensity score column name
    
    # Treatment effect estimation
    outcomes=['outcome1', 'outcome2'], # Outcome variables for effect estimation
    estimand='ate',                  # 'ate', 'att', or 'atc'
    
    # Other settings
    random_state=42,                 # For reproducibility
    verbose=True                     # Enable detailed logging
)
```

## Working with Results

```python
# Access matched data
matched_data = results.matched_data

# Examine balance statistics
balance_stats = results.balance_statistics

# Get treatment effect estimates
effects = results.effect_estimates

# Get propensity scores
propensity = results.propensity_scores

# Get match pairs information
pairs = results.get_match_pairs()
```

## Visualization Functions

```python
from cohortbalancer3.visualization import (
    plot_balance,                    # Balance before/after matching
    plot_propensity_distributions,   # Propensity score distributions
    plot_matched_pairs_distance,     # Match quality histogram
    plot_covariate_distributions,    # Distribution of covariates
    plot_treatment_effects,          # Forest plot of effects
    plot_matching_summary            # Sample size flow diagram
)
```

## Matching Methods

### Greedy Matching

Sequentially matches treatment units to their nearest control units. Fast but may not find globally optimal matches.

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi'],
    match_method='greedy',
    distance_method='propensity'
)
```

### Optimal Matching

Uses network flow algorithms to find globally optimal matches. Provides better balance but is slower.

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi'],
    match_method='optimal',
    distance_method='mahalanobis'
)
```

## Distance Metrics

### Propensity Score

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    distance_method='propensity',
    estimate_propensity=True
)
```

### Mahalanobis Distance

Accounts for correlation between variables:

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    distance_method='mahalanobis'
)
```

### Euclidean Distance

Simple distance useful when variables are standardized:

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    distance_method='euclidean',
    standardize=True
)
```

## Interpreting Balance

Standardized mean difference (SMD) is the key balance metric:
- SMD < 0.1: Good balance
- SMD 0.1-0.2: Moderate imbalance
- SMD > 0.2: Substantial imbalance

```python
# Calculate mean SMD across all covariates
mean_smd_before = results.balance_statistics['smd_before'].mean()
mean_smd_after = results.balance_statistics['smd_after'].mean()

print(f"Balance improvement: {(1 - mean_smd_after/mean_smd_before)*100:.1f}%")
```

## Troubleshooting

### Common Issues

1. **No matches found**: Try relaxing caliper or using different distance metric
   ```python
   config = MatcherConfig(..., caliper='auto')
   ```

2. **Poor balance after matching**: Try different matching method
   ```python
   config = MatcherConfig(..., match_method='optimal')
   ```

3. **Treatment effect not significant**: Check balance, increase sample size, review model
   ```python
   # Examine individual covariates
   print(results.balance_statistics.sort_values('smd_after', ascending=False))
   ```

4. **Error with propensity scores**: Use Mahalanobis distance or pre-compute propensity
   ```python
   config = MatcherConfig(..., distance_method='mahalanobis')
   ```

## Complete Example

```python
import pandas as pd
import numpy as np
from cohortbalancer3 import Matcher, MatcherConfig, create_report
from cohortbalancer3.visualization import plot_balance, plot_treatment_effects

# Generate synthetic data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'treatment': np.random.binomial(1, 0.3, n),
    'age': np.random.normal(50, 10, n),
    'bmi': np.random.normal(25, 5, n),
    'bp': np.random.normal(120, 15, n),
    'sex': np.random.binomial(1, 0.5, n)
})

# Make treatment more likely for older patients with higher BMI
p = 1 / (1 + np.exp(-(data['age']/10 + data['bmi']/5 - 10)))
data['treatment'] = np.random.binomial(1, p)

# Create outcome with treatment effect
data['outcome'] = (
    5 + 0.1*data['age'] + 0.2*data['bmi'] + 
    0.1*data['bp'] + 2*data['treatment'] + 
    np.random.normal(0, 1, n)
)

# Configure matching
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp', 'sex'],
    outcomes=['outcome'],
    match_method='optimal',
    distance_method='propensity',
    exact_match_cols=['sex'],
    caliper='auto',
    random_state=42
)

# Perform matching
matcher = Matcher(data, config)
matcher.match()
results = matcher.get_results()

# Examine results
print("\nBalance Statistics:")
balance = results.balance_statistics
print(f"Mean SMD before: {balance['smd_before'].mean():.4f}")
print(f"Mean SMD after: {balance['smd_after'].mean():.4f}")

print("\nTreatment Effect:")
effects = results.effect_estimates
print(f"Effect: {effects['effect'].values[0]:.4f}")
print(f"95% CI: [{effects['ci_lower'].values[0]:.4f}, {effects['ci_upper'].values[0]:.4f}]")
print(f"p-value: {effects['p_value'].values[0]:.4f}")

# Visualize results
plot_balance(results).savefig("balance.png")
plot_treatment_effects(results).savefig("effects.png")

# Create report
create_report(
    results,
    method_name="Optimal Matching Example",
    output_dir="./output",
    report_filename="matching_report.html"
)
```
