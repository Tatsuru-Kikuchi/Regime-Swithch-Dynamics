from setuptools import setup, find_packages

setup(
    name="regime-switch-dynamics",
    version="0.1.0",
    author="Tatsuru Kikuchi",
    description="Framework for boundary detection in treatment effects",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.1",
    ],
    python_requires=">=3.9",
)
