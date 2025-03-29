# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-03-29

### Removed

-   **Breaking Change:** Removed the legacy `MatchResults.match_pairs` property, which returned positional indices. Users should now use `MatchResults.pairs` (list of ID tuples) or `MatchResults.get_match_pairs()` (DataFrame of ID pairs).
-   Removed fallback logic in `visualization.plot_matched_pairs_distance` that relied on the legacy `match_pairs` property.

### Changed

-   Updated `visualization.plot_matched_pairs_scatter` to use ID-based `MatchResults.pairs` instead of the removed positional `match_pairs` property.

## [0.1.0] - 2025-03-14

-   Initial release. 