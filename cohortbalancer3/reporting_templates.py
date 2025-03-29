"""HTML templates and styling for CohortBalancer3 reports.

This module contains templates and styling for generating HTML reports from matching results.
These templates are used by the reporting module to create professional, publication-quality
reports summarizing propensity score matching analyses.
"""


def get_report_css() -> str:
    """Get CSS styling for HTML reports.

    Returns:
        str: CSS styling as a string

    """
    return """
        body { 
            padding: 40px 0; 
            font-family: 'Crimson Pro', 'Times New Roman', serif;
            line-height: 1.6;
            color: #212529;
            background-color: #fcfcfc;
        }
        .container {
            max-width: 1200px;
            background-color: white;
            padding: 2rem 3rem;
            box-shadow: 0 0 15px rgba(0,0,0,0.05);
            border-radius: 5px;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Lato', Arial, sans-serif;
            font-weight: 700;
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.7em;
        }
        h1 { 
            font-size: 2.2rem; 
            text-align: center;
            margin-bottom: 0.3em;
            border-bottom: 1px solid #eee;
            padding-bottom: 15px;
        }
        h2 { 
            font-size: 1.7rem; 
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            margin-top: 2.5rem;
        }
        h3 { font-size: 1.4rem; }
        p { margin-bottom: 1rem; }
        .text-muted { 
            font-family: 'Lato', Arial, sans-serif;
            text-align: center; 
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }
        .img-container { 
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px; 
            height: 100%;
            background-color: #fafafa;
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 15px;
        }
        .img-container img { 
            max-width: 100%; 
            max-height: 400px;
            object-fit: contain;
        }
        .figure-col {
            display: flex;
            flex-direction: column;
            margin-bottom: 30px;
        }
        .caption {
            margin-top: 10px;
            font-size: 0.9em;
            color: #555;
            text-align: center;
            padding: 0 10px;
            font-style: italic;
        }
        .section-description {
            background-color: #f7fbff;
            border-left: 4px solid #4a89dc;
            padding: 15px;
            margin-bottom: 25px;
            font-size: 0.95rem;
            border-radius: 3px;
        }
        .table {
            font-size: 0.9rem;
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        .table th {
            background-color: #f5f7f9;
            border-top: 2px solid #ddd;
        }
        .stats-card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 0 5px rgba(0,0,0,0.03);
            margin-bottom: 20px;
        }
        .stats-number {
            font-size: 2rem;
            font-weight: bold;
            color: #3a7bd5;
            font-family: 'Lato', Arial, sans-serif;
        }
        .stats-label {
            font-size: 0.9rem;
            color: #666;
            font-family: 'Lato', Arial, sans-serif;
        }
        .executive-summary {
            background-color: #f9f9f9;
            border: 1px solid #eaeaea;
            border-radius: 5px;
            padding: 20px 25px;
            margin: 30px 0;
        }
        .divider {
            height: 1px;
            background-color: #eee;
            margin: 40px 0;
        }
        @media print {
            body { background-color: white; }
            .container {
                box-shadow: none;
                max-width: 100%;
                padding: 0;
            }
            .img-container { break-inside: avoid; }
        }
    """


def get_report_template() -> str:
    """Get HTML template for matching report.

    Returns:
        str: HTML template as a string

    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Causal Analysis: {method_name} Matching Results</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600&family=Lato:wght@300;400;700&display=swap">
        <style>
            {css_style}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Matching Analysis</h1>
            <p class="text-muted">Method: {method_name} | Generated on {timestamp}</p>
            
            <div class="executive-summary">
                <h3 style="margin-top:0">Executive Summary</h3>
                <p>This report presents the results of a matching analysis using the <strong>{method_name}</strong> approach. 
                The matching procedure reduced the original sample of <strong>{n_total}</strong> subjects to <strong>{n_matched}</strong> matched subjects, 
                achieving a matching ratio of <strong>{matching_ratio:.2f}:1</strong> (control:treatment).</p>
                
                <p>The analysis used <strong>{distance_method}</strong> distance for matching
                {caliper_text} 
                {exact_match_text}</p>
                
                <p>After matching, {prop_balanced_after:.1f}% of covariates had standardized mean differences below 0.1 
                (compared to {prop_balanced_before:.1f}% before matching), indicating 
                {balance_quality} 
                overall balance achievement.</p>
            </div>
            
            <h2>1. Sample Characteristics</h2>
            <div class="section-description">
                This section provides an overview of the sample sizes before and after matching,
                illustrating how many treatment and control units were included in the analysis
                and the resulting matching ratio.
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="stats-card text-center">
                                <div class="stats-number">{n_treat_orig}</div>
                                <div class="stats-label">Original Treatment Units</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stats-card text-center">
                                <div class="stats-number">{n_control_orig}</div>
                                <div class="stats-label">Original Control Units</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stats-card text-center">
                                <div class="stats-number">{n_treat_matched}</div>
                                <div class="stats-label">Matched Treatment Units</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stats-card text-center">
                                <div class="stats-number">{n_control_matched}</div>
                                <div class="stats-label">Matched Control Units</div>
                            </div>
                        </div>
                        <div class="col-md-12">
                            <div class="stats-card text-center">
                                <div class="stats-number">{matching_ratio:.2f}:1</div>
                                <div class="stats-label">Matching Ratio (Control:Treatment)</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{summary_img}" alt="Matching Summary">
                    </div>
                    <div class="caption">
                        <strong>Figure 1:</strong> Sample sizes before and after matching. The bars represent counts of treatment and control units
                        in the original and matched datasets.
                    </div>
                </div>
            </div>
            
            <h2>2. Covariate Balance Assessment</h2>
            <div class="section-description">
                Covariate balance is crucial for valid causal inference. This section evaluates how well
                the matching procedure balanced the distribution of covariates between treatment and control groups.
                Standardized mean differences (SMD) below 0.1 indicate good balance.
            </div>
                
            <div class="row">
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{balance_img}" alt="Balance Plot">
                    </div>
                    <div class="caption">
                        <strong>Figure 2:</strong> Standardized mean differences (SMD) for covariates before (blue) and after (orange) matching.
                        The red horizontal line indicates the conventional 0.1 threshold for acceptable balance.
                    </div>
                </div>
                {matched_pairs_scatter_col}
            </div>
                
            <h3>Balance Metrics Summary</h3>
            <div class="row">
                <div class="col-md-6">
                    <table class="table table-sm table-bordered">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Before Matching</th>
                                <th>After Matching</th>
                                <th>Improvement</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Mean SMD</td>
                                <td>{mean_smd_before:.4f}</td>
                                <td>{mean_smd_after:.4f}</td>
                                <td>{smd_improvement:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Maximum SMD</td>
                                <td>{max_smd_before:.4f}</td>
                                <td>{max_smd_after:.4f}</td>
                                <td>{max_smd_improvement:.1f}%</td>
                            </tr>
                            <tr>
                                <td>% Covariates with SMD < 0.1</td>
                                <td>{prop_balanced_before:.1f}%</td>
                                <td>{prop_balanced_after:.1f}%</td>
                                <td>{balance_diff:.1f} pp</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <h3>Top 10 Covariates by Initial Imbalance</h3>
            {balance_html}
            
            {propensity_section}
            
            <div class="divider"></div>
            <h2>{propensity_section_number}. Matching Quality</h2>
            <div class="section-description">
                This section evaluates the quality of matches by examining the distribution of distances
                between matched pairs. Lower distances indicate better quality matches, while outliers
                may suggest problematic matches that could affect treatment effect estimates.
            </div>
            
            <div class="row">
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{distances_img}" alt="Match Distances">
                    </div>
                    <div class="caption">
                        <strong>Figure {distance_figure_number}:</strong> Distribution of distances between matched pairs.
                        The red vertical line indicates the median distance. Clusters of matches with large distances may warrant further investigation.
                    </div>
                </div>
                {extra_scatter_plot}
            </div>
            
            {treatment_effects_section}
            
            {covariate_distributions_section}
            
            <div class="divider"></div>
            <h2>Methodology Notes</h2>
            <div class="section-description">
                This section provides details on the matching methodology used in this analysis.
            </div>
            
            <h3>Matching Configuration</h3>
            <table class="table table-sm table-bordered">
                <tr><th>Method</th><td>{method_name}</td></tr>
                <tr><th>Distance Metric</th><td>{distance_method}</td></tr>
                <tr><th>Caliper</th><td>{caliper}</td></tr>
                <tr><th>Matching Ratio</th><td>{ratio}:1</td></tr>
                <tr><th>Exact Matching Columns</th><td>{exact_match_cols_str}</td></tr>
            </table>
                        
            <div class="text-center mt-5">
                <p class="text-muted">
                    <small>Generated using CohortBalancer3 | {timestamp}</small>
                </p>
            </div>
        </div>
    </body>
    </html>
    """


def get_matched_pairs_scatter_column() -> str:
    """Get HTML for matched pairs scatter plot column.

    Returns:
        str: HTML for the scatter plot column

    """
    return """
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{scatter_img}" alt="Matched Pairs Scatter">
                    </div>
                    <div class="caption">
                        <strong>Figure 3:</strong> Scatter plot showing matched pairs in the space of two covariates with the highest
                        initial imbalance. Blue points represent treatment units, orange points represent control units,
                        and connecting lines show the matches.
                    </div>
                </div>
    """


def get_empty_column() -> str:
    """Get HTML for an empty column.

    Returns:
        str: HTML for an empty column

    """
    return """
                <div class="col-md-6">
                </div>
    """


def get_propensity_section() -> str:
    """Get HTML for propensity score section.

    Returns:
        str: HTML for the propensity score section

    """
    return """
            <div class="divider"></div>
            <h2>3. Propensity Score Analysis</h2>
            <div class="section-description">
                Propensity scores estimate the probability of treatment assignment based on observed covariates.
                Comparing propensity distributions before and after matching helps evaluate whether the matching
                procedure successfully created comparable groups.
            </div>
            
            <div class="row">
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{propensity_comparison_img}" alt="Propensity Comparison">
                    </div>
                    <div class="caption">
                        <strong>Figure 4:</strong> Comparison of propensity score distributions before matching (left) and after matching (right).
                        Better overlap after matching indicates improved overall balance across covariates.
                    </div>
                </div>
                <div class="col-md-6 figure-col">
                    <div class="img-container">
                        <img src="{calibration_img}" alt="Propensity Calibration">
                    </div>
                    <div class="caption">
                        <strong>Figure 5:</strong> Calibration plot showing how well propensity scores align with observed treatment rates.
                        Points closer to the diagonal line indicate better calibration. Point size represents bin count.
                    </div>
                </div>
            </div>
    """


def get_treatment_effects_section() -> str:
    """Get HTML for treatment effects section.

    Returns:
        str: HTML for the treatment effects section

    """
    return """
            <div class="divider"></div>
            <h2>{section_number}. Treatment Effect Estimates</h2>
            <div class="section-description">
                After achieving balance through matching, this section presents the estimated causal effects
                of treatment on the outcome(s) of interest. These estimates represent the average treatment effect
                on the treated (ATT) under the assumption of no unmeasured confounding.
            </div>
            
            <div class="row">
                <div class="col-md-12 figure-col">
                    <div class="img-container">
                        <img src="{treatment_img}" alt="Treatment Effects">
                    </div>
                    <div class="caption">
                        <strong>Figure {figure_number}:</strong> Estimated treatment effects with confidence intervals.
                        Effects crossing the zero line (dashed red vertical line) are not statistically significant at α=0.05.
                        Asterisks indicate statistically significant effects.
                    </div>
                </div>
            </div>
            
            <h3>Detailed Treatment Effect Estimates</h3>
            {effect_table_html}
    """


def get_covariate_distributions_section() -> str:
    """Get HTML for covariate distributions section.

    Returns:
        str: HTML for the covariate distributions section

    """
    return """
            <div class="divider"></div>
            <h2>{section_number}. Detailed Covariate Distributions</h2>
            <div class="section-description">
                This section provides a more detailed view of how individual covariates were balanced through matching.
                Each row shows the distribution of a specific covariate before and after matching, allowing for
                visual assessment of balance improvement.
            </div>
            
            <div class="row">
                <div class="col-md-12 figure-col">
                    <div class="img-container">
                        <img src="{covariate_distributions_img}" alt="Covariate Distributions">
                    </div>
                    <div class="caption">
                        <strong>Figure {figure_number}:</strong> 
                        Distributions of individual covariates before matching (left) and after matching (right).
                        Each row represents one covariate, with SMD values quantifying the improvement in balance.
                        For binary variables, bar charts show the proportion of '1' values in each group.
                    </div>
                </div>
            </div>
    """
