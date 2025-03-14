#!/usr/bin/env python
"""
Example script demonstrating the reporting functionality in CohortBalancer3.

This script generates synthetic data, performs matching using various methods,
and creates HTML reports of the matching results using the built-in reporting tools.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from cohortbalancer3.datatypes import MatcherConfig
from cohortbalancer3.matcher import Matcher
from cohortbalancer3.reporting import create_report


# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)


def generate_synthetic_data(n_samples=500):
    """Generate synthetic data for matching example."""
    # Features
    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    systolic_bp = np.random.normal(120, 15, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    
    # Binary features
    sex = np.random.binomial(1, 0.5, n_samples)  # 0=female, 1=male
    smoker = np.random.binomial(1, 0.3, n_samples)
    
    # Categorical feature (converted to dummies)
    blood_type = np.random.choice(['A', 'B', 'AB', 'O'], size=n_samples, p=[0.4, 0.1, 0.1, 0.4])
    
    # Treatment assignment (more likely for older, male, smokers with high bp)
    logit = -2 + 0.03 * age + 0.5 * sex + 0.8 * smoker + 0.02 * systolic_bp
    prob_treatment = 1 / (1 + np.exp(-logit))
    treatment = np.random.binomial(1, prob_treatment)
    
    # Outcome with treatment effect = 5.0 (higher is worse)
    outcome = 50 + 5.0 * treatment + 0.1 * age + 5 * smoker + 0.05 * systolic_bp + np.random.normal(0, 5, n_samples)
    
    # Secondary outcome - more affected by age
    outcome2 = 20 + 3.0 * treatment + 0.2 * age + 0.1 * bmi + np.random.normal(0, 3, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'cholesterol': cholesterol,
        'smoker': smoker,
        'blood_type': blood_type,
        'treatment': treatment,
        'outcome': outcome,
        'outcome2': outcome2
    })
    
    # Create dummy variables for blood type and convert to int
    blood_type_dummies = pd.get_dummies(df['blood_type'], prefix='blood')
    # Convert boolean columns to integers (0/1)
    blood_type_dummies = blood_type_dummies.astype(int)
    df = pd.concat([df, blood_type_dummies], axis=1)
    
    return df


def main():
    """Main function to run the example."""
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=1000)
    
    # Save original data
    data.to_csv(os.path.join(OUTPUT_DIR, "synthetic_data.csv"), index=False)
    print(f"Synthetic data saved to {os.path.join(OUTPUT_DIR, 'synthetic_data.csv')}")
    
    # Method 1: Using the Matcher class to generate a report
    print("\n1. Using Matcher.create_report() for a greedy propensity score match...")
    
    # Define covariates for matching
    covariates = [
        'age', 'sex', 'bmi', 'systolic_bp', 'cholesterol', 'smoker', 
        'blood_A', 'blood_B', 'blood_AB', 'blood_O'
    ]
    
    # Create matching configuration
    config_greedy = MatcherConfig(
        treatment_col='treatment',
        covariates=covariates,
        outcomes=['outcome', 'outcome2'],
        match_method='greedy',
        distance_method='propensity',
        standardize=True,
        caliper='auto',
        exact_match_cols=['sex'],
        estimate_propensity=True,
        propensity_col=None,
        random_state=42,
        calculate_balance=True,
        ratio=1.0
    )
    
    # Initialize and perform matching
    matcher_greedy = Matcher(data=data, config=config_greedy)
    matcher_greedy.match()
    
    # Generate a report directly from the matcher
    greedy_report_path = matcher_greedy.create_report(
        method_name="Greedy Matching with Propensity Scores",
        output_dir=OUTPUT_DIR,
        report_filename="greedy_propensity_report.html"
    )
    print(f"Report created at: {greedy_report_path}")
    
    # Method 2: Using the create_report function directly
    print("\n2. Using create_report() directly for an optimal Mahalanobis match...")
    
    # Create a configuration for optimal matching with Mahalanobis distance
    config_optimal = MatcherConfig(
        treatment_col='treatment',
        covariates=covariates,
        outcomes=['outcome', 'outcome2'],
        match_method='optimal',
        distance_method='mahalanobis',
        standardize=True,
        caliper='auto',
        exact_match_cols=['sex'],
        estimate_propensity=False,
        random_state=42,
        calculate_balance=True,
        ratio=1.0
    )
    
    # Initialize and perform matching
    matcher_optimal = Matcher(data=data, config=config_optimal)
    matcher_optimal.match()
    results_optimal = matcher_optimal.get_results()
    
    # Generate a report using the create_report function directly
    from cohortbalancer3 import create_report
    
    optimal_report_path = create_report(
        results=results_optimal,
        method_name="Optimal Matching with Mahalanobis Distance",
        output_dir=OUTPUT_DIR,
        report_filename="optimal_mahalanobis_report.html"
    )
    print(f"Report created at: {optimal_report_path}")
    
    print("\nAll reports generated successfully!")


if __name__ == "__main__":
    main() 