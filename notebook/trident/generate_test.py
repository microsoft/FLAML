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
merged_notebook = nbformat.v4.new_notebook()
pkgs = {
    '"flaml[synapse,automl,ts_forecast]@https://automlsaeastus.blob.core.windows.net/releases/FLAML-latest-py3-none-any.whl"'
}
for notebook_name in all_notebooks:
    with open(notebook_name, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    # filter non-code cell
    code_cells = []
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            cell.outputs = []
            if "%pip install" in cell.source:
                package_line = cell.source.split("\n")[0]  # 只考虑单元格的第一行
                packages = package_line.split(" ")[2:]
                for package in packages:
                    if not package.startswith('"flaml') and not package.startswith("flaml"):
                        pkgs.add(package)
            else:
                code_cells.append(cell)

    merged_notebook.cells.append(nbformat.v4.new_markdown_cell(f"# {notebook_name}"))
    merged_notebook.cells.extend(code_cells)

# add install packages
pip_install_cell = nbformat.v4.new_code_cell(f"%pip install {' '.join(pkgs)}")
merged_notebook.cells.insert(0, pip_install_cell)

with open("notebook/trident/all_in_one_test.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(merged_notebook, f)
