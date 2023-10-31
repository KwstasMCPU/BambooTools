# Changelog

All notable changes to this project will be documented in this file.

## [v.0.4.0-beta] - 2023-10-31

### Added

- `duplication_summary()` method has been added. It produces a table with metrics regarding the duplicated records of the given dataset [#19](https://github.com/KwstasMCPU/BambooTools/issues/19).
- `duplication_frequency_table()` method has been added. It shows how many times a duplicated record appear per the number of its clones [#19](https://github.com/KwstasMCPU/BambooTools/issues/19).

### Changed

- `_conditional_probability()` internal class method converted to a standalone internal function.
- Docstring style finalized to `numpy`.

### Fixed
- Fixed `MemoryError` bug in `completeness()` caused in cases of datasets with many columns [#20](https://github.com/KwstasMCPU/BambooTools/issues/20).
- Removed unnecessary print statemens from `missing_corr_matrix()` [#21](https://github.com/KwstasMCPU/BambooTools/issues/21).

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
in the dataset.

## [v.0.2.0-alpha] - 2023-09-24

### Added
- Initial release.
