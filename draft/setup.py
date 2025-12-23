"""
CERTIFY-ED: Certified Exact Diagonalization Framework

Setup script for package installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="certify-ed",
    version="1.0.0",
    author="[Your Name]",
    author_email="[your.email@institution.edu]",
    description="Formal verification framework for quantum many-body exact diagonalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarang-kernel/CERTIFY-ED",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
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
    ],
    extras_require={
        "hdf5": ["h5py>=3.0.0"],
        "sagemath": ["sage>=9.8"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "certify-ed=certify_ed.cli:main",
        ],
    },
)
