# CohortBalancer3: Statistical Matching for Causal Inference

[![Tests](https://github.com/maschulz/cohortbalancer3/actions/workflows/tests.yml/badge.svg)](https://github.com/maschulz/cohortbalancer3/actions/workflows/tests.yml)
[![Build](https://github.com/maschulz/cohortbalancer3/actions/workflows/build.yml/badge.svg)](https://github.com/maschulz/cohortbalancer3/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/maschulz/cohortbalancer3/branch/main/graph/badge.svg)](https://codecov.io/gh/maschulz/cohortbalancer3)

## Installation

```bash
pip install "git+https://github.com/maschulz/cohortbalancer3.git#egg=cohortbalancer3[viz]"
```

Or for development:

```bash
git clone https://github.com/maschulz/cohortbalancer3.git
cd cohortbalancer3
pip install -e .[dev,viz]
```

## Core Concepts

CohortBalancer3 implements three primary components:

1. **MatcherConfig**: Defines matching parameters and constraints
2. **Matcher**: Executes the matching algorithm and computes statistics
3. **MatchResults**: Contains matched data, diagnostics, and effect estimates

## Quick Start

```python
import pandas as pd
from cohortbalancer3 import Matcher, MatcherConfig

# Prepare data with treatment indicators and covariates
data = pd.DataFrame({
    'treatment': [1, 1, 1, 0, 0, 0, 0, 0],
    'age': [45, 55, 35, 65, 40, 52, 38, 60],
    'bmi': [28, 32, 24, 30, 22, 29, 26, 31],
    'outcome': [120, 142, 118, 145, 110, 125, 115, 135]
})

# Configure matcher
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi'],
    outcomes=['outcome']
)

# Perform matching
matcher = Matcher(data, config)
matcher.match()
results = matcher.get_results()

# Examine results
print(f"Original data: {len(data)} observations")
print(f"Matched data: {len(results.matched_data)} observations")
print(f"Mean SMD before: {results.balance_statistics['smd_before'].mean():.4f}")
print(f"Mean SMD after: {results.balance_statistics['smd_after'].mean():.4f}")
print(f"Estimated effect: {results.effect_estimates['effect'].values[0]:.4f}")
```

## Matching Methods

### Greedy Matching

Greedy matching sequentially pairs treatment units with their nearest available control units. It is computationally efficient but may not find the globally optimal solution.

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi'],
    match_method='greedy',
    distance_method='euclidean'
)
```

### Optimal Matching

Optimal matching uses the Hungarian algorithm to find the global matching that minimizes the total distance across all pairs. It produces better balance but is computationally more intensive.

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi'],
    match_method='optimal',
    distance_method='mahalanobis'
)
```

### Fast Greedy Matching (for Large Datasets)

For very large datasets where creating a full distance matrix is not feasible due to memory or computational constraints (e.g., >10,000 units in either group), `fast_greedy` provides a memory-efficient alternative.

This method avoids computing the N x M distance matrix. Instead, it iterates through each treatment unit, uses a propensity score caliper to select a small pool of candidate control units, and only then computes distances for this small subset.

This approach requires propensity scores for candidate selection, and a caliper is highly recommended. By default, it uses a multivariate distance (like Mahalanobis) on the candidate pool, but it can be configured to use a "pure" propensity score distance as well.

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    match_method='fast_greedy',
    # Fast greedy requires propensity scores for candidate selection (auto-estimated if not provided)
    # A caliper is also required. Using an auto-caliper is recommended.
    caliper_method='mahalanobis', # Caliper on Mahalanobis distance for candidates
    caliper_value='auto',
    caliper_scale=0.05, # Use a p-value of 0.05 for the Chi-squared threshold
    # The primary distance metric is still used for the final selection within the caliper
    distance_method='mahalanobis'
)
```

The `fast_prefilter_caliper_scale` parameter in `MatcherConfig` provides an additional layer of control. It tunes the size of the initial candidate pool for each treatment unit, which can impact both performance and the quality of the final matches. A smaller value creates a more restrictive filter, leading to faster execution but potentially excluding some good matches, while a larger value is more inclusive but computationally heavier.

## Distance Metrics

### Euclidean Distance

Calculates the direct Euclidean distance between points in the covariate space:

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    distance_method='euclidean',
    standardize=True  # Recommended when using Euclidean distance
)
```

### Mahalanobis Distance

Accounts for covariance between variables, reducing the influence of correlated covariates:

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    distance_method='mahalanobis'
)
```

### Propensity Score Distance

Computes distances based on the propensity score (probability of treatment):

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    distance_method='propensity',
)
```

## Matching Constraints

### Exact Matching

Forces matches to have identical values for specified categorical variables:

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'sex'],
    exact_match_cols=['sex']
)
```

### Caliper Matching

Restricts matches to pairs within a maximum distance threshold, which is critical for avoiding poor matches. The caliper is defined by three parameters: `caliper_method`, `caliper_value`, and `caliper_scale`.

*   `caliper_method`: The metric to use for the caliper (e.g., `'propensity'`, `'mahalanobis'`, or a specific covariate name).
*   `caliper_value`: The threshold. This can be a specific number or `'auto'` for automatic calculation.
*   `caliper_scale`: A tuning parameter for `'auto'` calipers. Its meaning depends on the `caliper_method`.

#### Fixed Numeric Caliper

Provide a specific number to `caliper_value`. This is the simplest approach.

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi'],
    caliper_method='mahalanobis',
    caliper_value=4.0  # Max Mahalanobis distance of 4.0
)
```

#### Automatic Caliper Calculation

Set `caliper_value='auto'` to let the system determine a reasonable threshold. This is the recommended approach.

*   **For Propensity Scores (`caliper_method='propensity'`)**: `caliper_scale` is a multiplier for the standard deviation of the logit-transformed propensity scores. The default of `0.2` is a widely accepted rule of thumb.

    ```python
    config = MatcherConfig(
        treatment_col='treatment',
        covariates=['age', 'bmi'],
        caliper_method='propensity',
        caliper_value='auto',
        caliper_scale=0.2  # Sets caliper to 0.2 standard deviations
    )
    ```

*   **For Mahalanobis/Euclidean Distance**: `caliper_scale` is interpreted as a **p-value** for a Chi-squared test to identify multivariate outliers. A `caliper_scale` of `0.05` would discard matches that are statistically different at the 5% significance level.

    ```python
    config = MatcherConfig(
        treatment_col='treatment',
        covariates=['age', 'bmi'],
        distance_method='mahalanobis',
        caliper_method='mahalanobis',
        caliper_value='auto',
        caliper_scale=0.05  # Use a 5% significance level for the caliper
    )
    ```

### Ratio Matching

Controls how many control units are matched to each treatment unit:

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi'],
    ratio=2.0  # 2 control units per treatment unit (1:2 matching)
)
```

### Matching with Replacement

Allows control units to be matched to multiple treatment units:

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi'],
    replace=True  # Allow reuse of control units
)
```

## Propensity Score Estimation

By default, propensity scores are automatically estimated when required by your configuration:
- using `distance_method` in {`propensity`, `logit`}
- `match_method='fast_greedy'`
- applying calipers with `caliper_method` in {`propensity`, `logit`}

Set `estimate_propensity=True` when you want scores even if they’re not strictly required (e.g., diagnostics, trimming), or when you want to control the model and its hyperparameters. Provide `propensity_col` to use pre-computed scores.

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp', 'sex'],
    distance_method='propensity',
    estimate_propensity=True,
    propensity_model='logistic',  # 'logistic', 'random_forest', or 'xgboost'
    common_support_trimming=True,  # Remove units outside of common support
    trim_threshold=0.05  # Trimming threshold for common support
)
```

Using pre-computed propensity scores:

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    distance_method='propensity',
    estimate_propensity=False,
    propensity_col='propensity_score'  # Column with pre-computed scores
)
```

## Treatment Effect Estimation

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    outcomes=['outcome1', 'outcome2'],
    estimand='ate',  # 'ate', 'att', or 'atc'
    effect_method='mean_difference',  # or 'regression_adjustment'
    bootstrap_iterations=1000,
    confidence_level=0.95
)
```

## Balance Assessment

```python
# Standardized mean difference (SMD) is the key balance metric:
# SMD < 0.1: Good balance
# SMD 0.1-0.2: Moderate imbalance
# SMD > 0.2: Substantial imbalance

# Calculate mean SMD across all covariates
mean_smd_before = results.balance_statistics['smd_before'].mean()
mean_smd_after = results.balance_statistics['smd_after'].mean()

# Assess balance for individual covariates
balance_df = results.balance_statistics
print(balance_df.sort_values('smd_after', ascending=False))

# Get Rubin's rule statistics (% of variables with SMD < 0.25 and variance ratio between 0.5-2)
rubin_stats = results.rubin_statistics
print(f"Variables satisfying Rubin's rules: {rubin_stats['pct_both_good']:.1f}%")

# Get overall balance index (0-100 scale)
balance_index = results.balance_index['balance_index']
print(f"Balance index: {balance_index:.1f}/100")
```

## Visualization

```python
from cohortbalancer3.visualization import (
    plot_balance,
    plot_propensity_comparison,
    plot_matched_pairs_distance,
    plot_covariate_distributions,
    plot_treatment_effects
)

# Plot standardized mean differences
balance_plot = plot_balance(results)
balance_plot.savefig("balance.png")

# Plot propensity score distributions
prop_plot = plot_propensity_comparison(results)
prop_plot.savefig("propensity.png")

# Plot distribution of match distances
dist_plot = plot_matched_pairs_distance(results)
dist_plot.savefig("match_distances.png")

# Plot distributions of top imbalanced covariates
cov_plot = plot_covariate_distributions(results, max_vars=8)
cov_plot.savefig("covariate_distributions.png")

# Forest plot of treatment effects
effect_plot = plot_treatment_effects(results)
effect_plot.savefig("effects.png")
```

## Generating Reports

```python
from cohortbalancer3 import create_report

# Generate HTML report with visualizations
report_path = create_report(
    results,
    method_name="Optimal Matching Analysis",
    output_dir="./output",
    report_filename="matching_report.html",
    export_tables_to_csv=True,
    dpi=300
)
print(f"Report saved to: {report_path}")
```

Alternatively, use the method on the Matcher instance:

```python
matcher.create_report(
    method_name="Greedy Matching Analysis",
    output_dir="./output"
)
```

## Configuration Options Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `treatment_col` | str | - | Treatment indicator column (1=treatment, 0=control) |
| `covariates` | list[str] | - | List of covariate column names to balance |
| `match_method` | str | "greedy" | Matching algorithm: "greedy", "optimal", or "fast_greedy" |
| `distance_method` | str | "euclidean" | Distance metric: "euclidean", "mahalanobis", "propensity", "logit" |
| `exact_match_cols` | list[str] | [] | Columns requiring exact matching |
| `standardize` | bool | True | Whether to standardize covariates before distance calculation |
| `caliper_method` | str \| None | "propensity" | Metric for the caliper ('propensity', 'logit', 'mahalanobis', a covariate name, or None). |
| `caliper_value` | float \| str \| None | "auto" | Threshold for the caliper. Can be a numeric value, 'auto', or None. |
| `caliper_scale` | float | 0.2 | Scaling factor for 'auto' caliper. For propensity/logit, it's SDs of logit(ps). For Mahalanobis/Euclidean, it's the p-value for the Chi-squared threshold. |
| `fast_prefilter_caliper_scale` | float | 0.5 | For `fast_greedy`, scales the SD of logit(propensity) to set the initial candidate search caliper. |
| `replace` | bool | False | Whether to allow reuse of control units |
| `ratio` | float | 1.0 | Matching ratio (controls per treatment unit) |
| `random_state` | int \| None | None | Random seed for reproducibility |
| `weights` | dict[str, float] \| None | None | Covariate weights for distance calculation |
| `estimate_propensity` | bool | False | Whether to estimate propensity scores |
| `propensity_col` | str \| None | None | Pre-computed propensity score column |
| `common_support_trimming` | bool | False | Remove units outside common propensity support |
| `trim_threshold` | float | 0.05 | Threshold for common support trimming |
| `propensity_model` | str | "logistic" | Model for propensity estimation: "logistic", "random_forest", "xgboost", "custom" |
| `model_params` | dict | {} | Parameters for propensity model |
| `cv_folds` | int | 5 | Cross-validation folds for propensity estimation |
| `calculate_balance` | bool | True | Whether to calculate balance statistics |
| `max_standardized_diff` | float | 0.1 | Threshold for acceptable standardized difference |
| `outcomes` | list[str] | [] | Outcome variables for effect estimation |
| `estimand` | str | "ate" | Estimand type: "ate", "att", or "atc" |
| `effect_method` | str | "mean_difference" | Effect estimation method: "mean_difference" or "regression_adjustment" |
| `adjustment_covariates` | list[str] \| None | None | Covariates for regression adjustment |
| `bootstrap_iterations` | int | 1000 | Bootstrap iterations for confidence intervals |
| `confidence_level` | float | 0.95 | Confidence level for effect estimation |

## MatchResults Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `original_data` | pd.DataFrame | The dataset before matching |
| `matched_data` | pd.DataFrame | The dataset after matching |
| `pairs` | list[tuple] | List of matched pairs as (treatment_id, control_id) |
| `match_groups` | dict | Dictionary mapping treatment IDs to lists of control IDs |
| `match_distances` | list[float] | Distances for each matched pair |
| `distance_matrix` | np.ndarray | Full distance matrix between treatment and control |
| `propensity_scores` | np.ndarray | Estimated propensity scores |
| `propensity_model` | object | Fitted propensity score model |
| `propensity_metrics` | dict | Metrics of propensity model performance |
| `balance_statistics` | pd.DataFrame | Balance statistics for each covariate |
| `rubin_statistics` | dict | Balance statistics based on Rubin's rules |
| `balance_index` | dict | Summary of overall balance improvement |
| `effect_estimates` | pd.DataFrame | Treatment effect estimates for outcomes |
| `config` | MatcherConfig | The configuration used for matching |

## Useful Methods on MatchResults

```python
# Get matching summary statistics
summary = results.get_match_summary()
print(f"Matched {summary['n_treatment_matched']} treatment and {summary['n_control_matched']} control units")

# Get detailed balance statistics
balance = results.get_balance_summary()

# Get treatment effect estimates
effects = results.get_effect_summary()

# Get match pairs as a DataFrame
pairs_df = results.get_match_pairs()

# Get match groups (for ratio matching)
groups_df = results.get_match_groups()
```

## Complete Example

```python
import numpy as np
import pandas as pd
from cohortbalancer3 import Matcher, MatcherConfig, create_report

# Generate synthetic data with confounding
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'age': np.random.normal(50, 10, n),
    'bmi': np.random.normal(25, 5, n),
    'bp': np.random.normal(120, 15, n),
    'sex': np.random.binomial(1, 0.5, n)
})

# Make treatment more likely for older patients with higher BMI
p = 1 / (1 + np.exp(-(data['age']/10 + data['bmi']/5 - 10)))
data['treatment'] = np.random.binomial(1, p, n)

# Create outcome with treatment effect
data['outcome'] = (
    5 + 0.1*data['age'] + 0.2*data['bmi'] + 
    0.1*data['bp'] + 2*data['treatment'] + 
    np.random.normal(0, 1, n)
)

# Configure matching with optimal matching and Mahalanobis distance
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp', 'sex'],
    outcomes=['outcome'],
    match_method='optimal',
    distance_method='mahalanobis',
    exact_match_cols=['sex'],
    caliper_method='mahalanobis',
    caliper_value='auto',
    caliper_scale=0.05, # Use p=0.05 for Chi-squared threshold
    random_state=42
)

# Perform matching
matcher = Matcher(data, config)
matcher.match()
results = matcher.get_results()

# Examine balance before and after matching
print(f"Mean SMD before: {results.balance_statistics['smd_before'].mean():.4f}")
print(f"Mean SMD after: {results.balance_statistics['smd_after'].mean():.4f}")

# Examine treatment effect
effects = results.effect_estimates
print(f"Effect: {effects['effect'].values[0]:.4f}")
print(f"95% CI: [{effects['ci_lower'].values[0]:.4f}, {effects['ci_upper'].values[0]:.4f}]")
print(f"p-value: {effects['p_value'].values[0]:.4f}")

# Generate report
create_report(
    results,
    method_name="Optimal Matching Example",
    output_dir="./output",
    report_filename="matching_report.html"
)
```

## Troubleshooting

### No matches found
- Try relaxing the caliper constraint (`caliper_scale` > 0.1 or provide a numeric `caliper_value`).
- Use a different distance metric.
- Check if `exact_match_cols` is too restrictive.

```python
config = MatcherConfig(..., caliper_method='mahalanobis', caliper_value='auto', caliper_scale=0.5)
```

### Poor balance after matching
- Try `optimal` matching instead of `greedy`.
- Use `mahalanobis` distance if covariates are correlated.
- Ensure propensity score model is well-specified if using propensity-based matching (e.g., add interaction/polynomial terms).

```python
config = MatcherConfig(..., match_method='optimal', distance_method='mahalanobis')
```

### Computational performance issues
- For large datasets, use `match_method='fast_greedy'`. It is significantly faster and more memory-efficient.
- If not using `fast_greedy`, consider standard `greedy` matching, as it is faster than `optimal`.
- Consider sampling the control group if it's very large.
- Turn off bootstrapping for faster treatment effect estimates (`bootstrap_iterations=0`).

```python
config = MatcherConfig(
    ..., 
    match_method='fast_greedy',
    caliper_method='propensity',
    caliper_value='auto'
)
```

### Memory issues with distance matrix
- For large datasets, use `match_method='fast_greedy'`. This method is designed to avoid creating the full distance matrix, which is the primary source of memory errors.
- This requires propensity scores (auto-estimated if not provided) and setting a caliper (`caliper_method` and `caliper_value` must be set).

## Advanced Features

### Weighted Distance Calculation

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    weights={'age': 2.0, 'bmi': 1.0, 'bp': 0.5}  # Weight age more heavily
)
```

### Regression Adjustment

```python
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    outcomes=['outcome'],
    effect_method='regression_adjustment',
    adjustment_covariates=['age', 'bmi', 'bp']
)
```

### Different Estimands

```python
# Average Treatment Effect on the Treated (ATT)
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    outcomes=['outcome'],
    estimand='att'  # Focus on effect for treated population
)

# Average Treatment Effect on the Controls (ATC)
config = MatcherConfig(
    treatment_col='treatment',
    covariates=['age', 'bmi', 'bp'],
    outcomes=['outcome'],
    estimand='atc'  # Focus on effect for control population
)
```
