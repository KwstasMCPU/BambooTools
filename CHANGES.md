# Changelog

All notable changes to this project will be documented in this file.

## [v.0.4.0-beta]

### Added

- `duplication_summary()`
- `duplication_frequency_table()`

### Changed

### Fixed

## [v.0.3.0-beta] - 2023-10-02

### Added

- `missing_corr_matrix()` method has been added. It calculates the conditional 
probability of a column's value being NULL if another column's value is NULL.

- `format` argument was added in the `completeness_summary()` method. If True
the output is formated with pandas Styler.

### Changed
-  The outlier detector functions have been renamed to indicate that they 
are intended for internal use. A single leading underscore in front of the functions 
name has been added:

    * outlier_detector_std-> _outlier_detector_std
    * outlier_detector_iqr -> _outlier_detector_iqr
    * outlier_detector_percentiles -> _outlier_detector_percentiles

### Fixed
- Fixed bug in outlier_summary() where categorical columns were also present
in the dataset

## [v.0.2.0-alpha] - 2023-09-24

### Added
- Initial release

