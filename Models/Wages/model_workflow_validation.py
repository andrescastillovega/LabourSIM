# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import pymc.sampling.jax as pmjax
import jax
import datetime
import argparse

from wage_utils import Capturing, sampling_output
import wage_functions as wage

# Set random seed
RANDOM_SEED = 230810
rng = np.random.default_rng(RANDOM_SEED)

# Set JAX default backend as CPU 
jax.config.update('jax_platform_name', 'cpu')
print(f"JAX default backend: {jax.default_backend()}")


# Run model validations
if __name__ == "__main__":
    # Get script arguments
    parser = argparse.ArgumentParser(description="wage model")
    parser.add_argument("model_workflow", help="Specify the .csv file with the model workflow")
    parser.add_argument("model_dataset", help="Specify the .csv file with the model dataset")
    args = parser.parse_args()

    # Load data and workflow
    model_workflow = pd.read_csv(args.model_workflow) # Contains the model workflow
    dataset = pd.read_csv(args.model_dataset) # This is the dataset for model estimation

    # Get id_runs
    id_runs = model_workflow["id_run"].unique()

    # Initialize validation routine
    print(f"Starting model validation with {len(id_runs)} models - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Run validations
    for id_run in id_runs:
        # Create data summary
        data_summary = wage.create_data_summary(model_workflow, dataset, id_run)

        # Create coordinates
        COORDS = { value["dims"]: (np.arange(dataset.shape[0]) if value["type"] == "target" else value["cats"]) for _, value in data_summary.items()}

        # Run validation
        wage.validate_workflow(id_run=id_run,
                                model_name=model_workflow.query(f"id_run == {id_run}")["model_name"].unique()[0],
                                data_summary=data_summary,
                                coords=COORDS)
        
        print(f"Validation model {id_run} completed - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    