"""CERTIFY-ED Package Setup."""
from setuptools import setup, find_packages

setup(
    name="certify-ed",
    version="1.0.0",
    author="Sarang Vehale",
    author_email="sarangvehale2@gmail.com",
    description="Verified exact diagonalization for quantum many-body systems",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0", "scipy>=1.7.0"],
    extras_require={
        "validation": ["quspin>=0.3.6"],
        "high_precision": ["mpmath>=1.2.0"],
        "plotting": ["matplotlib>=3.5.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
)
