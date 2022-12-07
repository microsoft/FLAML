import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError
from flaml.utils import check_spark
import os
import pytest

try:
    check_spark()
    skip_spark = False
except Exception:
    print("Spark is not installed. Skip all spark tests.")
    skip_spark = True


here = os.path.abspath(os.path.dirname(__file__))


def run_notebook(input_nb, output_nb="executed_notebook.ipynb", save=False):
    try:
        file_path = os.path.join(here, input_nb)
        with open(file_path) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": here}})
    except CellExecutionError:
        raise
    except Exception as e:
        print("\nIgnoring below error:\n", e, "\n\n")
    finally:
        if save:
            with open(os.path.join(here, output_nb), "w", encoding="utf-8") as f:
                nbformat.write(nb, f)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_lightgbm_test():
    run_notebook("automl_lightgbm_test.ipynb")


if __name__ == "__main__":
    test_automl_lightgbm_test()
