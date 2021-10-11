import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()


# Get the code version
version = {}
with open(os.path.join(here, "flaml/version.py")) as fp:
    exec(fp.read(), version)
__version__ = version["__version__"]

install_requires = [
    "NumPy>=1.16.2",
    "lightgbm>=2.3.1",
    "xgboost>=0.90,<=1.3.3",
    "scipy>=1.4.1",
    "pandas>=1.1.4",
    "scikit-learn>=0.24",
]


setuptools.setup(
    name="FLAML",
    version=__version__,
    author="Microsoft Corporation",
    author_email="hpo@microsoft.com",
    description="A fast and lightweight autoML system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/FLAML",
    packages=setuptools.find_packages(include=["flaml*"]),
    install_requires=install_requires,
    extras_require={
        "notebook": [
            "openml==0.10.2",
            "jupyter",
            "matplotlib==3.2.0",
            "rgf-python",
        ],
        "test": [
            "flake8>=3.8.4",
            "pytest>=6.1.1",
            "coverage>=5.3",
            "pre-commit",
            "xgboost<1.3",
            "catboost>=0.23",
            "rgf-python",
            "optuna==2.8.0",
            "vowpalwabbit",
            "openml",
            "transformers==4.4.1",
            "wandb==0.10.26",
            "torch==1.8.1",
            "datasets==1.4.1",
            "azure-storage-blob",
            "statsmodels>=0.12.2",
        ],
        "catboost": ["catboost>=0.23"],
        "blendsearch": ["optuna==2.8.0"],
        "ray": [
            "ray[tune]==1.6.0",
            "pyyaml<5.3.1",
        ],
        "azureml": [
            "azureml-mlflow",
        ],
        "nni": [
            "nni",
        ],
        "vw": [
            "vowpalwabbit",
        ],
        "nlp": [
            "ray[tune]>=1.6.0",
            "transformers",
            "datasets==1.4.1",
            "tensorboardX<=2.2",
            "torch",
        ],
        "forecast": ["prophet>=1.0.1", "statsmodels>=0.12.2"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
