# CohortBalancer3 API Guide

CohortBalancer3 is a Python package for propensity score matching and treatment effect estimation. This guide explains the simplified API.

## Core Components

The CohortBalancer3 API consists of three main components:

1. **MatcherConfig**: A single configuration class with flattened parameters
2. **Matcher**: Main class for performing matching operations
3. **MatchResults**: Container for all matching results

Additionally, visualization functions are provided in a separate module.

## Configuration (MatcherConfig)

The `MatcherConfig` class contains all settings needed for matching, propensity score estimation, balance assessment, and treatment effect estimation:

```python
from cohortbalancer3 import MatcherConfig

# Create a configuration
config = MatcherConfig(
    # Core parameters
    treatment_col="treatment",
    covariates=["age", "sex", "bmi"],
    
    # Matching parameters
    match_method="greedy",  # "greedy", "optimal"
    distance_method="mahalanobis",  # "euclidean", "mahalanobis", "propensity", "logit"
    exact_match_cols=["sex"],
    ratio=2.0,  # 1:2 matching
    caliper="auto",  # Automatically calculate optimal caliper
    caliper_scale=0.2,  # Scale factor for automatic caliper (default: 0.2)
    
    # Propensity parameters
    estimate_propensity=True,
    propensity_model="logistic",
    
    # Outcome parameters
    outcomes=["y1", "y2", "y3"],
    effect_method="regression_adjustment"
)
```

### Configuration Parameters

#### Core Parameters
- `treatment_col`: Name of column containing treatment indicator (0/1)
- `covariates`: List of covariate column names

#### Matching Parameters
- `match_method`: Matching algorithm ("greedy", "optimal")
- `distance_method`: Method for calculating distances
- `exact_match_cols`: Columns to match exactly on
- `standardize`: Whether to standardize covariates
- `caliper`: Maximum allowed distance for a match (numeric value, "auto", or None)
- `caliper_scale`: Scale factor for automatic caliper calculation (default: 0.2)
- `replace`: Whether to allow replacement in matching
- `ratio`: Matching ratio (e.g., 2.0 for 1:2 matching)
- `random_state`: Random seed for reproducibility
- `weights`: Optional dictionary of variable weights

### Automatic Caliper Calculation

CohortBalancer3 supports automatic caliper calculation based on the distance method:

- For propensity score matching (`"propensity"` or `"logit"`): 
  - `0.2 × standard deviation of the logit of propensity scores` (based on Austin, 2011)
  
- For Mahalanobis distance matching:
  - `90th percentile of the distance distribution`
  
- For other distance methods:
  - `median of the distance distribution`

To use automatic caliper calculation, set `caliper="auto"` in your MatcherConfig:

```python
config = MatcherConfig(
    # Other parameters...
    caliper="auto",
    caliper_scale=0.2,  # Use 0.2 for standard recommendation, adjust as needed
    # More parameters...
)
```

The `caliper_scale` parameter allows you to adjust the strictness of the caliper (lower values = stricter matching).

#### Propensity Parameters
- `estimate_propensity`: Whether to estimate propensity scores
- `propensity_col`: Name of existing propensity score column
- `propensity_model`: Model type for estimation
- `model_params`: Parameters for propensity model
- `logit_transform`: Whether to apply logit transformation
- `common_support_trimming`: Whether to trim units outside common support
- `trim_threshold`: Threshold for trimming

#### Balance Parameters
- `calculate_balance`: Whether to calculate balance statistics
- `max_standardized_diff`: Maximum acceptable standardized difference

#### Outcome Parameters
- `outcomes`: List of outcome column names
- `estimand`: Type of estimand ("ate", "att", "atc")
- `effect_method`: Method for effect estimation
- `adjustment_covariates`: Covariates for regression adjustment
- `bootstrap_iterations`: Number of bootstrap iterations
- `confidence_level`: Confidence level for intervals

## Matcher

The `Matcher` class performs the matching and analysis:

```python
from cohortbalancer3 import Matcher

# Create matcher
matcher = Matcher(data=df, config=config)

# Perform matching and get results
results = matcher.match().get_results()

# Save results to files
matcher.save_results(directory="output_directory")
```

## MatchResults

The `MatchResults` class contains all results from the matching process:

```python
# Get summary statistics
match_summary = results.get_match_summary()
balance_summary = results.get_balance_summary()
effect_summary = results.get_effect_summary()
match_pairs = results.get_match_pairs()
```

### Available Result Data

- `original_data`: Original DataFrame
- `matched_data`: DataFrame with matched units
- `treatment_indices`, `control_indices`: Indices of treatment and control units
- `match_pairs`: Dictionary mapping treatment indices to control indices
- `propensity_scores`: Estimated propensity scores
- `balance_statistics`: DataFrame with balance statistics
- `effect_estimates`: DataFrame with treatment effect estimates

## Visualization Functions

Visualization functions are provided separately:

```python
from cohortbalancer3.visualization import (
    plot_balance, plot_love_plot, plot_propensity_distributions, 
    plot_treatment_effects, plot_matching_summary
)

# Create balance plot
balance_fig = plot_balance(results, max_vars=10)
balance_fig.savefig("balance_plot.png")

# Create propensity distributions plot
propensity_fig = plot_propensity_distributions(results)
propensity_fig.savefig("propensity_plot.png")

# Create treatment effects plot
effects_fig = plot_treatment_effects(results)
effects_fig.savefig("effects_plot.png")

# Create matching summary plot
summary_fig = plot_matching_summary(results)
summary_fig.savefig("summary_plot.png")
```

## Complete Example

```python
import pandas as pd
from cohortbalancer3 import Matcher, MatcherConfig
from cohortbalancer3.visualization import plot_balance, plot_treatment_effects

# Load data
df = pd.read_csv("data.csv")

# Create configuration
config = MatcherConfig(
    treatment_col="treatment",
    covariates=["age", "sex", "bmi", "blood_pressure"],
    match_method="optimal",
    distance_method="mahalanobis",
    exact_match_cols=["sex"],
    estimate_propensity=True,
    outcomes=["outcome1", "outcome2"]
)

# Perform matching
matcher = Matcher(data=df, config=config)
results = matcher.match().get_results()

# Analyze results
balance_df = results.get_balance_summary()
effect_df = results.get_effect_summary()

# Create visualizations
plot_balance(results).savefig("balance.png")
plot_treatment_effects(results).savefig("effects.png")
```

## Common Use Cases

### Simple Matching
```python
config = MatcherConfig(
    treatment_col="treatment",
    covariates=["age", "sex", "bmi"],
    match_method="greedy"
)
```

### Exact Matching on Key Variables
```python
config = MatcherConfig(
    treatment_col="treatment",
    covariates=["age", "sex", "bmi"],
    exact_match_cols=["sex", "diagnosis"]
)
```

### Propensity Score Matching
```python
config = MatcherConfig(
    treatment_col="treatment",
    covariates=["age", "sex", "bmi"],
    estimate_propensity=True,
    match_method="greedy",
    distance_method="propensity"
)
```

### Matching with Custom Weights
```python
config = MatcherConfig(
    treatment_col="treatment",
    covariates=["age", "sex", "bmi", "blood_pressure"],
    weights={"age": 2.0, "blood_pressure": 1.5}
)
```

### Treatment Effect Estimation
```python
config = MatcherConfig(
    treatment_col="treatment",
    covariates=["age", "sex", "bmi"],
    outcomes=["y1", "y2"],
    effect_method="regression_adjustment",
    adjustment_covariates=["age", "sex"]
)
``` 