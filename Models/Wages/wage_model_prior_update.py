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

# Ignore Arviz RuntimeWarning when samplin with low number of draws (for testing purposes)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set Arviz plotting options
rc = {"plot.max_subplots": 50}
az.rcParams.update(rc)

# Run wage model
if __name__ == "__main__":
    # Get script arguments
    parser = argparse.ArgumentParser(description="wage model")
    parser.add_argument("model_workflow", help="Specify the .csv file with the model workflow")
    parser.add_argument("model_dataset", help="Specify the .csv file with the model dataset")
    args = parser.parse_args()

    # Set sampling parameters
    nchains = 4
    ndraws = 10
    ntune = 10
    target_accept = 0.95

    # Load data and workflow
    model_workflow = pd.read_csv(args.model_workflow) # Contains the model workflow
    dataset = pd.read_csv(args.model_dataset) # This is the dataset for model estimation

    # Get id_runs
    id_runs = model_workflow["id_run"].unique()

    # Get years of the dataset
    years = dataset["year"].unique()
    years.sort()

    # Create sampling record
    sampling_record = pd.DataFrame()

    # Initialize model routine
    print(f"Starting model routine with {len(id_runs)} models - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Run models with updating priors (for each year)
    for id_run in id_runs:
        print(f">>>>>>>>>>> Run {id_run} <<<<<<<<<<<<")
        for year in years:
            print(f"Year {year}")
            # Create data summary
            data_summary = wage.create_data_summary(model_workflow, dataset, id_run, year, f"year == {year}")

            # Create coordinates
            COORDS = { value["dims"]: (np.arange(dataset.query(f"year == {year}").shape[0]) if value["type"] == "target" else value["cats"])
                                    for _, value in data_summary.items()}

            # Run model
            sampling_record, run_summary = wage.run_updating_priors(id_run=id_run,
                                    model_name=f"{year}",
                                    data_summary=data_summary,
                                    coords=COORDS,
                                    sampling_record=sampling_record,
                                    nchains=nchains,
                                    ndraws=ndraws,
                                    ntune=ntune,
                                    target_accept=target_accept)


            # Save sampling record and print summary
            sampling_record.to_csv("sampling_record.csv")
            print(f"Run {id_run} completed - Name: {run_summary['model_name']} - Start: {run_summary['start_time']} - End: {run_summary['end_time']} - Max Rhat: {run_summary['maxRhat']} - Avg It/s: {run_summary['AvgIt/s']}")


