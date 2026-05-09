# HowToTestFLAML4Fabric

## Steps to Test FLAML for Fabric

1. **Code Change and Merge:**

   - Complete the code change in FLAML.
   - Merge the changes into the main branch.

1. **Update Package:**

   - For Spark 3.4, upload the Conda tar package to the channel.
   - For Spark 3.5, upload the pip wheel (this will be automatically done by the FLAML-Internal-Tag Pipeline, add a tag on the main to trigger it).

1. **Update Synapse-Conda:**

   - Update the Synapse-Conda environment according to the repository link [Synapse-Conda](https://msdata.visualstudio.com/A365/_git/Synapse-Conda).

1. **Testing and Publishing:**

   - To test the package, using the Synapse-Conda-PullRequest pipeline version is sufficient.
   - To publish FLAML to BBC-VHD, use the Synapse-Conda-Official version after merging the PR in Synapse-Conda.

1. **Update BBC-VHD:**

   - Create a dev branch based on the Spark version (e.g., releases/spark34).
   - Update the version in `setup.sh` and `version.txt` under `Components/Conda/spark{version}` folder.

1. **Create CI Pipeline:**

   - Create the CI pipeline based on the branch according to the Spark version. For Spark 3.4, use [Spark 3.4 CI](https://msdata.visualstudio.com/A365/_release?_a=releases&view=mine&definitionId=1707) and for Spark 3.5, use [Spark 3.5 CI](https://msdata.visualstudio.com/A365/_release?_a=releases&view=mine&definitionId=1821).

1. **Library Management and Environment Update:**

   - Use the VHD ID and set the parameter DefaultPoolComputeSparkVersion to [create a session pool](https://msdata.visualstudio.com/A365/_build?definitionId=23259&_a=summary). Follow the [Wiki](https://msdata.visualstudio.com/A365/_wiki/wikis/A365.wiki/44492/Library-Management-testing-with-custom-VHD) and refer to the [Example PR](https://msdata.visualstudio.com/A365/_git/BBC-VHD/pullrequest/1427724) for guidance.
   - The Library Management stage requires uploading some Python packages first, then changing the tridentSessionPoolID to check if the change of pool id will break the existing environment.

1. **Testing Notebooks:**

   - Test the following notebooks in [Edog environment](https://powerbi-df.analysis-df.windows.net/home?trident=1&experience=data-science) using the LM version. **The two AI Sample Notebooks can be directly created from the Edog to ensure you have the latest updates** and the rest notebooks can be found in `notebook/trident`.:
     - `package_version_check.ipynb`
     - `AI Sample - Automated ML`
     - `AI Sample -Model tuning using FLAML`
     - `time_series.ipynb`
     - `automl_plot.ipynb`
     - `featurization.ipynb`
   - For a thorough test, use the `all_in_one_test.ipynb` for additional testing.
   - Check the VHD id and FLAML version for two settings.
