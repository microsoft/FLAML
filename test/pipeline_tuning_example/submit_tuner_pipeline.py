import logging
from azureml.core import Workspace, Dataset
from azure.ml.component import (
    Component,
    dsl,
)
import argparse
from pathlib import Path

LOCAL_DIR = Path(__file__).parent.absolute()

def remote_run():
    ################################################
    # connect to your Azure ML workspace
    ################################################
    ws = Workspace(subscription_id="79f57c16-00fe-48da-87d4-5192e86cd047", 
                resource_group="Alexander256",
                workspace_name="Alexander256V100")

    ################################################
    # load component functions
    ################################################

    # pipeline_tuning_func = Component.from_yaml(ws, yaml_file=LOCAL_DIR/"tuner/component_spec.yaml")
    pipeline_tuning_func = Component.from_yaml(ws, yaml_file=LOCAL_DIR/"tuner/component_spec.yaml")
    
    ################################################
    # build pipeline
    ################################################
    @dsl.pipeline(
        name="pipeline_tuning_",
        default_compute_target="cpucluster",
    )
    def sample_pipeline():
        tuner = pipeline_tuning_func()

    pipeline = sample_pipeline()

    run = pipeline.submit(regenerate_outputs=True)
    return run

def local_run():
    logger.info("Run tuner locally.")
    from tuner import tuner_func
    tuner_func.tune_pipeline(concurrent_run=2)

if __name__ == "__main__":
    # parser argument 
    parser = argparse.ArgumentParser()
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument(
        "--subscriptionId", type=str, default="48bbc269-ce89-4f6f-9a12-c6f91fcb772d"
    )
    parser.add_argument("--resourceGroup", type=str, default="aml1p-rg")
    parser.add_argument("--workspace", type=str, default="aml1p-ml-wus2")
    
    parser.add_argument('--remote', dest='remote', action='store_true')
    parser.add_argument('--local', dest='remote', action='store_false')
    parser.set_defaults(remote=True)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    if args.remote:
        remote_run()
    else:
        local_run()
