# Pupil Labs Camera

[![ci](https://github.com/pupil-labs/pl-camera/actions/workflows/main.yml/badge.svg)](https://github.com/pupil-labs/pl-camera/actions/workflows/main.yml)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://pupil-labs.github.io/pl-camera/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre_commit-black?logo=pre-commit&logoColor=FAB041)](https://github.com/pre-commit/pre-commit)
[![pypi version](https://img.shields.io/pypi/v/pupil-labs-camera.svg)](https://pypi.org/project/pupil-labs-camera/)
[![python version](https://img.shields.io/pypi/pyversions/pupil-labs-camera)](https://pypi.org/project/pupil-labs-camera/)

This repo contains functionality around the usage of camera intrinsics for undistorting data, and projecting and unprojecting points.

It is mostly a wrapper around OpenCV's functionality, providing type hints, input validation, a more intuitive interface, and some changes to improve computational performance.

## Installation

```
pip install pupil-labs-camera
```

or

```bash
pip install -e git+https://github.com/pupil-labs/pl-camera.git
```
