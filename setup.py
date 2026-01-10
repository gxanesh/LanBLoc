#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script for LanBLoc package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().split("\n") 
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="lanbloc",
    version="1.0.0",
    author="Ganesh Sapkota",
    author_email="gsapkota@mst.edu",
    description="Landmark-Based Localization for GPS-Denied Environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gxanesh/lanbloc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: visual localizarion",
        "Topic :: Scientific/Engineering ::  GPS-DENIED Localization",
        "Topic :: Scientific/Engineering ::  landmark based Localization",
        "Topic :: Scientific/Engineering ::  YOLO ",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lanbloc=lanbloc.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "lanbloc": ["config/*.yaml"],
    },
)
