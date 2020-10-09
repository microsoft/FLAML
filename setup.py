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

install_requires=[
    "NumPy>=1.16.2",
    "lightgbm>=2.3.1",
    "xgboost>=0.90",
    "scipy>=1.4.1",
    "catboost>=0.23",
    "scikit-learn>=0.22",
],


setuptools.setup(
    name="FLAML",
    version=__version__,
    author="Chi Wang & Qingyun Wu",
    author_email="wang.chi@microsoft.com",
    description="A fast and lightweight autoML system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/FLAML",
    packages=["flaml"],
    install_requires=install_requires,
    extras_require={
        "notebook": [
            "openml==0.10.2",
            "jupyter",
            "matplotlib==3.2.0",
            "rgf-python",
        ],
        "test": [
            "nose>=1.3.7",
        ],
    },
    classifiers=[],
    python_requires=">=3.6",
)
