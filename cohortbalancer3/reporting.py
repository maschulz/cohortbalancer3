"""
Reporting functionality for CohortBalancer3.

This module provides functions for generating professional HTML reports and
exporting matching results in various formats. It leverages the visualization
module to create informative plots and the reporting_templates module for
structuring HTML reports.
"""

import os
import tempfile
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Dict

import matplotlib.pyplot as plt
import pandas as pd

from cohortbalancer3.reporting_templates import (
    get_covariate_distributions_section,
    get_empty_column,
    get_matched_pairs_scatter_column,
    get_propensity_section,
    get_report_css,
    get_report_template,
    get_treatment_effects_section,
)
from cohortbalancer3.utils.logging import get_logger
from cohortbalancer3.visualization import (
    plot_balance,
    plot_covariate_distributions,
    plot_match_groups,
    plot_matched_pairs_distance,
    plot_matched_pairs_scatter,
    plot_matching_summary,
    plot_propensity_calibration,
    plot_propensity_comparison,
    plot_treatment_effects,
)

# Set up logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from cohortbalancer3.datatypes import MatchResults


def create_visualizations(
    results: "MatchResults",
    output_dir: str,
    prefix: str = "",
    dpi: int = 300,
    max_vars_balance: int = 15,
    max_vars_dist: int = 8,
) -> Dict[str, str]:
    """Create visualizations for matching results and save them to disk.

    Args:
        results: MatchResults object containing matching results
        output_dir: Directory where visualizations will be saved
        prefix: Prefix to add to filenames
        dpi: DPI for saved images
        max_vars_balance: Maximum number of variables to show in balance plot
        max_vars_dist: Maximum number of variables to show in distribution plots

    Returns:
        Dictionary mapping visualization names to file paths
    """
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        prefix = f"{prefix}_"

    # Create dictionary to store image filenames for the HTML report
    image_paths = {}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use results.original_data and results.matched_data instead of derived datasets

    # 1. Balance plot
    fig_balance = plot_balance(results, max_vars=max_vars_balance, figsize=(12, 8))
    balance_filename = f"{prefix}balance_plot_{timestamp}.png"
    fig_balance.savefig(os.path.join(output_dir, balance_filename), dpi=dpi)
    plt.close(fig_balance)
    image_paths["balance"] = os.path.join(output_dir, balance_filename)

    # Additional visualizations for many-to-one matching
    if results.config.ratio > 1.0:
        # Create a special visualization showing matching groups
        fig_groups = plot_match_groups(results, figsize=(12, 8))
        groups_filename = f"{prefix}match_groups_{timestamp}.png"
        fig_groups.savefig(os.path.join(output_dir, groups_filename), dpi=dpi)
        plt.close(fig_groups)
        image_paths["match_groups"] = os.path.join(output_dir, groups_filename)

    # Propensity scores visualization (if available)
    if results.propensity_scores is not None:
        try:
            # 2. Propensity calibration plot
            fig_calibration = plot_propensity_calibration(results)
            calibration_filename = f"{prefix}propensity_calibration_{timestamp}.png"
            fig_calibration.savefig(
                os.path.join(output_dir, calibration_filename), dpi=dpi
            )
            plt.close(fig_calibration)
            image_paths["calibration"] = os.path.join(output_dir, calibration_filename)

            # 3. Propensity score comparison (before vs after matching)
            fig_ps_comp = plot_propensity_comparison(results, figsize=(12, 6))
            ps_comp_filename = f"{prefix}propensity_comparison_{timestamp}.png"
            fig_ps_comp.savefig(os.path.join(output_dir, ps_comp_filename), dpi=dpi)
            plt.close(fig_ps_comp)
            image_paths["propensity_comparison"] = os.path.join(
                output_dir, ps_comp_filename
            )
        except Exception as e:
            warnings.warn(f"Could not create propensity score visualizations: {str(e)}")

    # 4. Treatment effects (if outcomes were provided)
    if results.effect_estimates is not None:
        try:
            fig_effects = plot_treatment_effects(results)
            effects_filename = f"{prefix}treatment_effects_{timestamp}.png"
            fig_effects.savefig(os.path.join(output_dir, effects_filename), dpi=dpi)
            plt.close(fig_effects)
            image_paths["treatment"] = os.path.join(output_dir, effects_filename)
        except Exception as e:
            warnings.warn(f"Could not create treatment effect visualizations: {str(e)}")

    # 5. Matched pairs distance
    try:
        fig_distances = plot_matched_pairs_distance(results)
        distances_filename = f"{prefix}matched_distances_{timestamp}.png"
        fig_distances.savefig(os.path.join(output_dir, distances_filename), dpi=dpi)
        plt.close(fig_distances)
        image_paths["distances"] = os.path.join(output_dir, distances_filename)
    except Exception as e:
        warnings.warn(
            f"Could not create matched pairs distance visualization: {str(e)}"
        )

    # 6. Matching summary
    try:
        fig_summary = plot_matching_summary(results)
        summary_filename = f"{prefix}matching_summary_{timestamp}.png"
        fig_summary.savefig(os.path.join(output_dir, summary_filename), dpi=dpi)
        plt.close(fig_summary)
        image_paths["summary"] = os.path.join(output_dir, summary_filename)
    except Exception as e:
        warnings.warn(f"Could not create matching summary visualization: {str(e)}")

    # 7. Covariate distributions before and after matching
    try:
        fig_cov_dist = plot_covariate_distributions(
            results, max_vars=max_vars_dist, figsize=(14, 16)
        )
        cov_dist_filename = f"{prefix}covariate_distributions_{timestamp}.png"
        fig_cov_dist.savefig(os.path.join(output_dir, cov_dist_filename), dpi=dpi)
        plt.close(fig_cov_dist)
        image_paths["covariate_distributions"] = os.path.join(
            output_dir, cov_dist_filename
        )
    except Exception as e:
        warnings.warn(
            f"Could not create covariate distributions visualization: {str(e)}"
        )

    # 8. Matched pairs scatter plot
    if results.balance_statistics is not None:
        try:
            sorted_vars = results.balance_statistics.sort_values(
                "smd_before", ascending=False
            )
            # Get top 2 covariates that are in the data (not treatment or outcome)
            covariates = results.config.covariates
            top_vars = [var for var in sorted_vars["variable"] if var in covariates]

            if len(top_vars) >= 2:
                x_var, y_var = top_vars[0], top_vars[1]
                fig_scatter = plot_matched_pairs_scatter(
                    results, x_var=x_var, y_var=y_var, figsize=(10, 10)
                )
                scatter_filename = f"{prefix}matched_pairs_scatter_{timestamp}.png"
                fig_scatter.savefig(os.path.join(output_dir, scatter_filename), dpi=dpi)
                plt.close(fig_scatter)
                image_paths["matched_pairs_scatter"] = os.path.join(
                    output_dir, scatter_filename
                )
        except Exception as e:
            warnings.warn(
                f"Could not create matched pairs scatter visualization: {str(e)}"
            )

    return image_paths


def export_tables(
    results: "MatchResults", output_dir: str, prefix: str = ""
) -> Dict[str, str]:
    """Export tables from matching results to CSV files.

    Args:
        results: MatchResults object containing matching results
        output_dir: Directory where tables will be saved
        prefix: Prefix to add to filenames

    Returns:
        Dictionary mapping table names to file paths
    """
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        prefix = f"{prefix}_"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store file paths
    table_paths = {}

    # 1. Matched data
    matched_data_path = os.path.join(
        output_dir, f"{prefix}matched_data_{timestamp}.csv"
    )
    results.matched_data.to_csv(matched_data_path, index=True)
    table_paths["matched_data"] = matched_data_path

    # 2. Balance statistics
    if results.balance_statistics is not None:
        balance_path = os.path.join(
            output_dir, f"{prefix}balance_statistics_{timestamp}.csv"
        )
        results.balance_statistics.to_csv(balance_path, index=False)
        table_paths["balance_statistics"] = balance_path

    # 3. Effect estimates
    if results.effect_estimates is not None:
        effects_path = os.path.join(
            output_dir, f"{prefix}effect_estimates_{timestamp}.csv"
        )
        results.effect_estimates.to_csv(effects_path, index=False)
        table_paths["effect_estimates"] = effects_path

    # 4. Match pairs (convert to DataFrame first)
    match_pairs_df = results.get_match_pairs()
    match_pairs_path = os.path.join(output_dir, f"{prefix}match_pairs_{timestamp}.csv")
    match_pairs_df.to_csv(match_pairs_path, index=False)
    table_paths["match_pairs"] = match_pairs_path

    # 5. Rubin statistics (if available)
    if results.rubin_statistics is not None:
        rubin_path = os.path.join(
            output_dir, f"{prefix}rubin_statistics_{timestamp}.csv"
        )
        pd.DataFrame([results.rubin_statistics]).to_csv(rubin_path, index=False)
        table_paths["rubin_statistics"] = rubin_path

    # 6. Balance index (if available)
    if results.balance_index is not None:
        balance_index_path = os.path.join(
            output_dir, f"{prefix}balance_index_{timestamp}.csv"
        )
        pd.DataFrame([results.balance_index]).to_csv(balance_index_path, index=False)
        table_paths["balance_index"] = balance_index_path

    return table_paths


def generate_html_report(
    results: "MatchResults",
    method_name: str,
    image_paths: Dict[str, str],
    output_dir: str,
    filename: str = "matching_report.html",
) -> str:
    """Generate a professional HTML report summarizing the matching results.

    Args:
        results: MatchResults object containing matching results
        method_name: Name of the matching method used (e.g., "Greedy Matching with Propensity Scores")
        image_paths: Dictionary of image paths from create_visualizations function
        output_dir: Directory where the report will be saved
        filename: Output filename for the HTML report

    Returns:
        Path to the generated HTML report
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert absolute image paths to relative paths for the HTML
    relative_image_paths = {}
    for key, path in image_paths.items():
        if os.path.isabs(path):
            # Extract just the filename for use in HTML
            relative_image_paths[key] = os.path.basename(path)
        else:
            relative_image_paths[key] = path

    # Get basic statistics
    n_total = len(results.original_data)
    n_matched = len(results.matched_data)
    n_treat_orig = (results.original_data[results.config.treatment_col] == 1).sum()
    n_control_orig = (results.original_data[results.config.treatment_col] == 0).sum()
    n_treat_matched = (results.matched_data[results.config.treatment_col] == 1).sum()
    n_control_matched = (results.matched_data[results.config.treatment_col] == 0).sum()
    matching_ratio = n_control_matched / n_treat_matched if n_treat_matched > 0 else 0.0

    # Extract treatment effects
    effect_html = ""
    effect_table_html = ""
    if results.effect_estimates is not None:
        effect_table_html = results.effect_estimates.to_html(
            float_format="%.4f", classes="table table-bordered table-striped table-sm"
        )

    # Prepare balance statistics summary
    balance_html = ""
    if results.balance_statistics is not None:
        # Get top 10 covariates with highest SMD before matching
        top_covariates = results.balance_statistics.sort_values(
            "smd_before", ascending=False
        ).head(10)

        balance_html = top_covariates.to_html(
            float_format="%.4f", classes="table table-bordered table-striped table-sm"
        )

        # Compute overall balance metrics
        mean_smd_before = results.balance_statistics["smd_before"].mean()
        mean_smd_after = results.balance_statistics["smd_after"].mean()
        max_smd_before = results.balance_statistics["smd_before"].max()
        max_smd_after = results.balance_statistics["smd_after"].max()
        prop_balanced_before = (
            results.balance_statistics["smd_before"] < 0.1
        ).mean() * 100
        prop_balanced_after = (
            results.balance_statistics["smd_after"] < 0.1
        ).mean() * 100
    else:
        # Fallback values if balance statistics are not available
        mean_smd_before = mean_smd_after = max_smd_before = max_smd_after = float("nan")
        prop_balanced_before = prop_balanced_after = float("nan")

    # Calculate improvement metrics
    if mean_smd_before > 0:
        smd_improvement = (1 - mean_smd_after / mean_smd_before) * 100
    else:
        smd_improvement = 0.0

    if max_smd_before > 0:
        max_smd_improvement = (1 - max_smd_after / max_smd_before) * 100
    else:
        max_smd_improvement = 0.0

    balance_diff = prop_balanced_after - prop_balanced_before

    # Get configuration details
    config = results.config
    distance_method = config.distance_method
    caliper = config.caliper
    exact_match_cols = config.exact_match_cols if config.exact_match_cols else []
    ratio = config.ratio

    # Prepare text with conditional statements pre-processed
    if caliper == "auto":
        caliper_text = " with an automatically selected caliper"
    else:
        caliper_text = f" with a caliper of {caliper}"

    if exact_match_cols:
        exact_match_text = (
            f" Exact matching was enforced on: {', '.join(exact_match_cols)}."
        )
    else:
        exact_match_text = ""

    # Determine balance quality text
    if prop_balanced_after > 90:
        balance_quality = "excellent"
    elif prop_balanced_after > 75:
        balance_quality = "good"
    elif prop_balanced_after > 50:
        balance_quality = "moderate"
    else:
        balance_quality = "poor"

    # Format exact_match_cols for display
    exact_match_cols_str = ", ".join(exact_match_cols) if exact_match_cols else "None"

    # Determine section and figure numbers based on which content is available
    has_propensity = "propensity_comparison" in relative_image_paths
    has_treatment = "treatment" in relative_image_paths

    # Section numbering
    propensity_section_number = 3
    matching_quality_section_number = 4 if has_propensity else 3
    treatment_section_number = 5 if has_propensity else 4
    covariate_section_number = (
        6
        if has_propensity and has_treatment
        else 5
        if has_propensity or has_treatment
        else 4
    )

    # Figure numbering
    distance_figure_number = 6 if has_propensity else 4
    treatment_figure_number = 7 if has_propensity else 5
    covariate_figure_number = (
        9
        if has_propensity and has_treatment
        else 7
        if has_propensity or has_treatment
        else 5
    )

    # Get CSS style
    css_style = get_report_css()

    # Prepare template components
    if "matched_pairs_scatter" in relative_image_paths:
        scatter_template = get_matched_pairs_scatter_column()
        matched_pairs_scatter_col = scatter_template.format(
            scatter_img=relative_image_paths.get("matched_pairs_scatter", "")
        )
    else:
        matched_pairs_scatter_col = get_empty_column()

    # Prepare propensity section if needed
    if has_propensity:
        propensity_template = get_propensity_section()
        propensity_section = propensity_template.format(
            propensity_comparison_img=relative_image_paths.get(
                "propensity_comparison", ""
            ),
            calibration_img=relative_image_paths.get("calibration", ""),
        )
    else:
        propensity_section = ""

    # Prepare second scatter plot section if needed
    if (
        "matched_pairs_scatter" in relative_image_paths
        and "scatter_template" in locals()
    ):
        # Only add this if it wasn't already included in the balance section
        extra_scatter_plot = ""
    else:
        extra_scatter_plot = get_empty_column()

    # Prepare treatment effects section if needed
    if has_treatment:
        treatment_template = get_treatment_effects_section()
        treatment_effects_section = treatment_template.format(
            treatment_img=relative_image_paths.get("treatment", ""),
            section_number=treatment_section_number,
            figure_number=treatment_figure_number,
            effect_table_html=effect_table_html,
        )
    else:
        treatment_effects_section = ""

    # Prepare covariate distributions section if needed
    if "covariate_distributions" in relative_image_paths:
        covariate_template = get_covariate_distributions_section()
        covariate_distributions_section = covariate_template.format(
            covariate_distributions_img=relative_image_paths.get(
                "covariate_distributions", ""
            ),
            section_number=covariate_section_number,
            figure_number=covariate_figure_number,
        )
    else:
        covariate_distributions_section = ""

    # Get HTML template and format with data
    html_template = get_report_template()
    html_content = html_template.format(
        # Basic info
        css_style=css_style,
        method_name=method_name,
        timestamp=timestamp,
        # Statistics
        n_total=n_total,
        n_matched=n_matched,
        n_treat_orig=n_treat_orig,
        n_control_orig=n_control_orig,
        n_treat_matched=n_treat_matched,
        n_control_matched=n_control_matched,
        matching_ratio=matching_ratio if isinstance(matching_ratio, float) else 0,
        # Balance metrics
        mean_smd_before=mean_smd_before,
        mean_smd_after=mean_smd_after,
        max_smd_before=max_smd_before,
        max_smd_after=max_smd_after,
        prop_balanced_before=prop_balanced_before,
        prop_balanced_after=prop_balanced_after,
        smd_improvement=smd_improvement,
        max_smd_improvement=max_smd_improvement,
        balance_diff=balance_diff,
        # Pre-processed conditional text
        caliper_text=caliper_text,
        exact_match_text=exact_match_text,
        balance_quality=balance_quality,
        # Configuration
        distance_method=distance_method,
        caliper=caliper,
        ratio=ratio,
        exact_match_cols_str=exact_match_cols_str,
        # Images
        summary_img=relative_image_paths.get("summary", ""),
        balance_img=relative_image_paths.get("balance", ""),
        distances_img=relative_image_paths.get("distances", ""),
        # HTML components
        balance_html=balance_html,
        matched_pairs_scatter_col=matched_pairs_scatter_col,
        propensity_section=propensity_section,
        propensity_section_number=propensity_section_number,
        distance_figure_number=distance_figure_number,
        extra_scatter_plot=extra_scatter_plot,
        treatment_effects_section=treatment_effects_section,
        covariate_distributions_section=covariate_distributions_section,
    )

    # Write HTML file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def create_report(
    results: "MatchResults",
    method_name: str = None,
    output_dir: str = None,
    prefix: str = "",
    report_filename: str = "matching_report.html",
    export_tables_to_csv: bool = True,
    dpi: int = 300,
    max_vars_balance: int = 15,
    max_vars_dist: int = 8,
) -> str:
    """Create a comprehensive HTML report and optionally export data tables.

    Args:
        results: MatchResults object containing matching results
        method_name: Description of the matching method for the report title
        output_dir: Directory to save report files, temporary dir used if None
        prefix: Prefix for all output filenames
        report_filename: Filename for the HTML report
        export_tables_to_csv: Whether to export data tables to CSV files
        dpi: Resolution for output figures
        max_vars_balance: Maximum number of variables to show in balance plots
        max_vars_dist: Maximum number of variables to show in distribution plots

    Returns:
        Path to the generated HTML report
    """
    from cohortbalancer3.datatypes import MatchResults

    logger.info("Creating matching results report")

    if not isinstance(results, MatchResults):
        raise TypeError("Results must be a MatchResults object")

    # Generate default method name if not provided
    if method_name is None:
        config = results.config
        method_parts = []

        # Add matching method
        method_parts.append(config.match_method.capitalize())

        # Add distance method if not the default for the matching method
        if (
            config.distance_method != "euclidean"
            and config.match_method != "propensity"
        ):
            method_parts.append(config.distance_method.capitalize())

        # Add replacement info if enabled
        if config.replace:
            method_parts.append("with replacement")

        # Add ratio if not 1:1
        if config.ratio != 1.0:
            method_parts.append(f"{config.ratio:.1f}:1 ratio")

        method_name = " ".join(method_parts)

    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
        logger.info(
            f"No output directory specified, using temporary directory: {output_dir}"
        )
    else:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving report files to: {output_dir}")

    # Create visualizations
    logger.debug("Generating visualizations")
    image_paths = create_visualizations(
        results,
        output_dir,
        prefix=prefix,
        dpi=dpi,
        max_vars_balance=max_vars_balance,
        max_vars_dist=max_vars_dist,
    )

    # Export tables if requested
    if export_tables_to_csv:
        logger.debug("Exporting data tables to CSV")
        table_paths = export_tables(results, output_dir, prefix=prefix)

    # Generate the HTML report
    logger.debug("Generating HTML report")
    report_path = generate_html_report(
        results, method_name, image_paths, output_dir, filename=report_filename
    )

    logger.info(f"Report created successfully: {report_path}")
    return report_path
