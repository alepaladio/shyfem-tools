from setuptools import setup, find_packages

setup(
    name="shyfem-tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "xarray>=2023.1.0",
        "netcdf4>=1.6.2",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "geopy>=2.3.0",
    ],
    author="Ale Paladio - CNR ISMAR Venezia",
    description="Tools for SHYFEM model post-processing",
    python_requires=">=3.7",
)