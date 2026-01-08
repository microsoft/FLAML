import subprocess
from importlib.metadata import distributions

installed_libs = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in distributions())

first_tier_dependencies = [
    "numpy",
    "jupyter",
    "lightgbm",
    "xgboost",
    "scipy",
    "pandas",
    "scikit-learn",
    "thop",
    "pytest",
    "coverage",
    "pre-commit",
    "torch",
    "torchvision",
    "catboost",
    "rgf-python",
    "optuna",
    "openml",
    "statsmodels",
    "psutil",
    "dataclasses",
    "transformers[torch]",
    "transformers",
    "datasets",
    "evaluate",
    "nltk",
    "rouge_score",
    "hcrystalball",
    "seqeval",
    "pytorch-forecasting",
    "mlflow-skinny",
    "joblibspark",
    "joblib",
    "nbconvert",
    "nbformat",
    "ipykernel",
    "pytorch-lightning",
    "tensorboardX",
    "requests",
    "packaging",
    "dill",
    "ray",
    "prophet",
]


for lib in installed_libs:
    lib_name = lib.split("==")[0]
    if lib_name in first_tier_dependencies:
        print(lib)

# print current commit hash
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
print(f"Current commit hash: {commit_hash}")
