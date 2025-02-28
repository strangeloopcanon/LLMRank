# Release Instructions

This document outlines the steps to build and release SlopRank to PyPI.

## Prerequisites

Before releasing, make sure you have the following installed:
- Python 3.8 or later
- build (`pip install build`)
- twine (`pip install twine`)
- A PyPI account with permissions to upload to the sloprank package

## Release Process

### 1. Update Version Number

Ensure the version number in `pyproject.toml` has been incremented appropriately following semantic versioning.

### 2. Update CHANGELOG.md

Make sure the CHANGELOG.md file is updated with all changes in the new version.

### 3. Clean Previous Builds

```bash
rm -rf build/ dist/ *.egg-info/
```

### 4. Build the Package

```bash
python -m build
```

This will create both source distribution and wheel in the `dist/` directory.

### 5. Check the Distribution

```bash
twine check dist/*
```

This will validate that the package is ready for upload.

### 6. Upload to TestPyPI (Optional)

To test the release before publishing to the main PyPI:

```bash
twine upload --repository testpypi dist/*
```

Then install and test:

```bash
pip install --index-url https://test.pypi.org/simple/ sloprank
```

### 7. Upload to PyPI

```bash
twine upload dist/*
```

### 8. Create GitHub Release

- Go to the GitHub repository
- Go to the "Releases" tab
- Click "Draft a new release"
- Tag version: v0.2.3 (or the current version)
- Title: Version 0.2.3
- Description: Copy the relevant section from CHANGELOG.md
- Attach the built distributions from the dist/ directory
- Publish the release

### 9. Verify Installation

```bash
pip install -U sloprank
python -c "import sloprank; print(sloprank.__version__)"
```

Make sure it shows the correct version number.

## Troubleshooting

- If you get permission errors during upload, check your PyPI credentials
- If the build fails, ensure all dependencies are correctly listed in pyproject.toml
- If the package fails validation, fix the issues before attempting to upload again