#!/usr/bin/env python
"""Basic Matching Example for CohortBalancer3

This example demonstrates the core functionality of CohortBalancer3:
1. Creating synthetic data
2. Configuring matching parameters
3. Performing greedy matching
4. Assessing balance and treatment effects
5. Generating a visual report

The example is self-contained and can be run directly.
"""

import logging
import os

import numpy as np
import pandas as pd

# Import from cohortbalancer3
from cohortbalancer3 import Matcher, MatcherConfig, configure_logging, create_report

# Set up logging - you can set to DEBUG for more detailed output
configure_logging(level=logging.INFO)


def generate_synthetic_data(n_samples=1000, random_state=42):
    """Generate synthetic data with a treatment effect."""
    np.random.seed(random_state)

    # Create covariates with different distributions for treatment/control
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)

    # Calculate propensity score (probability of treatment)
    # More likely to get treatment if x1 and x2 are high
    propensity = 1 / (1 + np.exp(-(0.5 * x1 + 0.8 * x2 - 0.1 * x3 + 0.1)))

    # Assign treatment based on propensity
    treatment = (np.random.random(n_samples) < propensity).astype(int)

    # Create an outcome with a treatment effect
    # The treatment increases the outcome by 2 units on average
    outcome = (
        1.2 * x1
        + 0.8 * x2
        - 0.5 * x3
        + 2 * treatment
        + np.random.normal(0, 1, n_samples)
    )

    # Create a categorical variable for exact matching demonstration
    category = np.random.choice(["A", "B", "C"], n_samples)

    # Create a DataFrame
    data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "treatment": treatment,
            "outcome": outcome,
            "category": category,
            "propensity": propensity,
        }
    )

    print(f"Generated data with {n_samples} samples")
    print(f"Treatment group: {data['treatment'].sum()} units")
    print(f"Control group: {n_samples - data['treatment'].sum()} units")

    return data


def main():
    """Run the complete matching workflow."""
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=1000)

    # Calculate true average treatment effect (for comparison)
    true_ate = 2.0  # We built the data with this effect

    # Calculate naive estimate (simple difference in means)
    naive_estimate = (
        data[data["treatment"] == 1]["outcome"].mean()
        - data[data["treatment"] == 0]["outcome"].mean()
    )

    print(f"True treatment effect: {true_ate:.4f}")
    print(f"Naive estimate (before matching): {naive_estimate:.4f}")
    print("This naive estimate is biased due to confounding factors")

    # Configure the matcher
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["x1", "x2", "x3"],
        match_method="greedy",  # Using greedy matching
        distance_method="euclidean",
        standardize=True,
        caliper="auto",  # Automatically determine appropriate caliper
        exact_match_cols=[],  # Could use ['category'] for exact matching
        replace=False,
        ratio=1.0,  # 1:1 matching
        random_state=42,
        # Include the outcome for treatment effect estimation
        outcomes=["outcome"],
        estimand="ate",
        effect_method="mean_difference",
    )

    # Create and run the matcher
    matcher = Matcher(data, config)
    matcher.match()
    results = matcher.get_results()

    # Display matching summary
    print("\nMatching Summary:")
    n_matched = len(results.matched_data) // 2  # Since we're doing 1:1 matching
    print(f"Matched pairs: {n_matched}")

    # Display balance statistics
    if results.balance_statistics is not None:
        balance_df = results.balance_statistics
        print("\nBalance Statistics:")
        print(
            balance_df[
                [
                    "variable",
                    "smd_before",
                    "smd_after",
                    "var_ratio_before",
                    "var_ratio_after",
                ]
            ]
            .sort_values("smd_before", ascending=False)
            .head()
        )

        # Overall balance
        mean_smd_before = balance_df["smd_before"].mean()
        mean_smd_after = balance_df["smd_after"].mean()
        print(
            f"\nMean standardized difference - Before: {mean_smd_before:.4f}, After: {mean_smd_after:.4f}"
        )

    # Display treatment effect estimates
    if results.effect_estimates is not None:
        effect_df = results.effect_estimates
        print("\nTreatment Effect Estimates:")
        print(effect_df[["outcome", "effect", "ci_lower", "ci_upper", "p_value"]])

        # Compare with true effect
        estimated_ate = effect_df[effect_df["outcome"] == "outcome"]["effect"].values[0]
        print(f"\nTrue ATE: {true_ate:.4f}")
        print(f"Naive estimate (before matching): {naive_estimate:.4f}")
        print(f"Matched estimate (after matching): {estimated_ate:.4f}")

    # Generate a visual report
    output_dir = os.path.join("examples", "output")
    # If we're already in the examples directory, adjust the path
    if os.path.basename(os.getcwd()) == "examples":
        output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    # For more reliable path handling
    absolute_output_dir = os.path.abspath(output_dir)
    print(f"Saving report files to: {absolute_output_dir}")

    report_path = create_report(
        results,
        method_name="Greedy Matching Example",
        output_dir=absolute_output_dir,
        report_filename="basic_matching_report.html",
        # Set dpi higher for better quality images
        dpi=150,
    )

    print(f"\nGenerated report at: {report_path}")
    print(
        "Open this HTML file in a browser to see the complete visualization of results"
    )
    print(
        "Tip: If images aren't showing, check that the image files exist in the same directory as the HTML file"
    )


if __name__ == "__main__":
    main()
