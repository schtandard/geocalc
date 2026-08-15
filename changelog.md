# Changelog of the `geocalc` package

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

### Removed

- Remove the `experimental` sub-package for now.

## 2026-08-03 - v0.4.0

### Changed

- Move `cps` module into `experimental` sub-package.

## 2025-05-12 - v0.3.0

### Added

- Experimental: Thickness correction.
- Experimental: `cps` module.

### Changed

- Rename arguments `layers_below` and `layers_above` to `below` and `above`.

### Fixed

- Accept numpy scalars as layer specification.
- Handle IDC calculation with geometry parameter arrays and non-trivial dielectric
  layers.
- Improved misleading phrasing in IDC theory.

## 2024-06-21 - v0.2.3

### Fixed

- Critical! Wrong factor in IDC capacitance calculation with infinite dielectric layers.
  Luckily, the resulting error was small.

## 2024-05-15 - v0.2.2

### Fixed

- Critical! Computation bug introduced in previous version.

## v0.2.1 - 2024-05-14

### Fixed

- Erroneous: Wrong calculation value for CPW layers with infinite thickness.
  We thought we used m = k as the argument for the elliptic integral there even though
  it was correct (m = k^2). This was "fixed", introducing a calculation error. In
  reality, it was only wrong in the documentation.
- m = k instead of m = k^2 in the theory part of the documentation.

### Added

- Support for CPW metallization thickness correction.

### Fixed

- Documentation typos.

## v0.2.0 - 2023-12-17

### Changed

- Layer specifications may now also be scalars or `None`.
- Renamed every occurrence of "beneath" to the more natural "below".

## v0.1.2 - 2023-11-13

### Changed

- Small documentation improvements.

### Fixed

- Add missing dependency `mpmath`.

## v0.1.1 - 2023-01-11

### Added

- Calculate `L` in `cpw.characteristics`.

### Changed

- Do not return `Cvac` in `cpw.characteristics`.
- Small theory fixes.
- Extend theory documentation on CPWs by lossy cases.

## v0.1.0 - 2023-01-11

### Added

- First tagged release.
