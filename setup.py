from setuptools import setup, find_packages

setup(
    name="cnrn-reproduce",
    version="0.1.0",
    description="Vendored torchlogic + CNRN experiment scripts for reproducing paper results.",
    packages=find_packages(include=["torchlogic", "torchlogic.*"]),
    python_requires=">=3.9.5, <3.11",
)
