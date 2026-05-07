"""
CERTIFY-ED: Certified Exact Diagonalization for Quantum Many-Body Systems
========================================================================

A Python framework for verified exact diagonalization with multi-oracle
consensus validation and exportable verification certificates.

Author: Sarang Vehale
License: MIT
Target Publication: Computer Physics Communications
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="certify-ed",
    version="1.0.0",
    author="Sarang Vehale",
    author_email="sarangvehale2@gmail.com",
    description="Verified exact diagonalization for quantum many-body systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarangvehale/certify-ed",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "sympy>=1.9",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "benchmarks": [
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
        ],
        "validation": [
            "quspin>=0.3.7",
        ],
    },
)
