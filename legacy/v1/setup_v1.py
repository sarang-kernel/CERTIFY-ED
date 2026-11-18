"""CERTIFY-ED Package Setup."""
from setuptools import setup, find_packages

setup(
    name="certify-ed",
    version="2.0.0.dev0",
    author="Sarang Vehale",
    author_email="sarangvehale2@gmail.com",
    description="Producer-independent certificates for exact-diagonalization results",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0", "scipy>=1.7.0"],
    extras_require={
        "validation": ["quspin>=1.0"],
        "rigorous": ["python-flint>=0.6"],
        "hdf5": ["h5py>=3.0"],
        "high_precision": ["mpmath>=1.2.0"],
        "plotting": ["matplotlib>=3.5.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
)
