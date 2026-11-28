import hashlib

import nbformat

all_notebooks = [
    "notebook/trident/FLAML Demo - Overview.ipynb",
    "notebook/trident/automl_autolog_on.ipynb",
    "notebook/trident/automl_autolog_off.ipynb",
    "notebook/trident/tune_autolog_on.ipynb",
    "notebook/trident/tune_autolog_off.ipynb",
    "notebook/trident/demo_1_flight_delays_automl.ipynb",
    "notebook/trident/demo_2_house_price_tune_synapseml.ipynb",
    "notebook/trident/demo_3_bankrupt_automl_synapseml.ipynb",
    "notebook/trident/demo_4_tune_lexicographic.ipynb",
    "notebook/automl_time_series_forecast.ipynb",
]


def main():
    merged_notebook = nbformat.v4.new_notebook()
    pkgs = {'"flaml[synapse,automl,forecast]@git+https://github.com/microsoft/FLAML.git@main"'}
    for notebook_name in all_notebooks:
        with open(notebook_name, encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        code_cells = []
        for index, cell in enumerate(notebook.cells):
            if cell["cell_type"] == "code":
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop("ms_comments", None)
                cell["id"] = hashlib.sha256(f"{notebook_name}:{index}".encode()).hexdigest()[:32]
                if "%pip install" in cell.source:
                    packages = cell.source.split("\n")[0].split(" ")[2:]
                    for package in packages:
                        if not package.startswith('"flaml') and not package.startswith("flaml"):
                            pkgs.add(package)
                else:
                    code_cells.append(cell)

        heading = nbformat.v4.new_markdown_cell(f"# {notebook_name}")
        heading["id"] = hashlib.sha256(f"heading:{notebook_name}".encode()).hexdigest()[:32]
        merged_notebook.cells.append(heading)
        merged_notebook.cells.extend(code_cells)

    install = nbformat.v4.new_code_cell(f"%pip install {' '.join(sorted(pkgs))}")
    install["id"] = "install-public-dependencies"
    merged_notebook.cells.insert(0, install)
    with open("notebook/trident/all_in_one_test.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(merged_notebook, f)


if __name__ == "__main__":
    main()
