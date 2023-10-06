import argparse
import jax
import numpyro
import os
import pandas as pd
from tqdm import tqdm
import yaml

from GammaGLM import GammaGLM
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LabourSIM wage model")
    parser.add_argument("--dataset", help="Specify the .csv file with the labour market data", required=True)
    parser.add_argument("--workflow", help="Specify the .yaml file with the workflow", required=True)
    args = parser.parse_args()

    # Create outputs folder
    workflow_filename = args.workflow.split("/")[-1].split(".")[0]
    OUTPUTS_PATH = fr"../outputs/{workflow_filename}"
    if not os.path.exists(OUTPUTS_PATH):
        os.makedirs(OUTPUTS_PATH)

    # Configure precision and platform
    jax.config.update("jax_platform_name", "gpu")

    # Load data and workflow
    data = pd.read_csv(args.dataset)
    data = data.dropna()
    with open(args.workflow, 'r') as file:
        workflow = yaml.safe_load(file)

    # Run models
    for model in workflow:
        model_name, model_specs = list(model.items())[0]
        
        # Create model folder
        if model_specs["year"] is None:
            if not os.path.exists(f"{OUTPUTS_PATH}/{model_name}"):
                os.makedirs(f"{OUTPUTS_PATH}/{model_name}")
        else:
            if not os.path.exists(f"{OUTPUTS_PATH}/{model_name}/{model_specs['year']}"):
                os.makedirs(f"{OUTPUTS_PATH}/{model_name}/{model_specs['year']}")

        # Filter dataset by year
        if model_specs["year"] is not None:
            data = data[data["year"] == model_specs["year"]]
            print(f"Dataset for year {model_specs['year']} contains {data.shape[0]} observations")

        # Create progress bar
        progress_bar = tqdm(total=5, desc=f"Running {model_name} model", ncols=100,
                             bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}' )

        # Create model object
        model = GammaGLM(model_name, 
                         model_specs["model_type"], 
                         data,
                         model_specs["target"], 
                         model_specs["parameters"], 
                         model_specs["dimensions"], 
                         model_specs["standardize_vars"],
                         OUTPUTS_PATH, 
                         model_specs["year"])
        progress_bar.update(1)

        # Build model
        model.build()
        progress_bar.update(1)

        # Run model
        trace, divergences = model.run(model_specs["run_settings"]["iterations"],
                  model_specs["run_settings"]["chains"],
                  model_specs["run_settings"]["target_accept"],
                  model_specs["run_settings"]["batch_size"])
        max_rhat = utils.get_rhat_max(trace)
        progress_bar.set_postfix({"max_rhat": f"{max_rhat:.3f}"})
        progress_bar.update(1)

        # Save trace
        if model_specs["year"] is None:
            trace.to_netcdf(f"{OUTPUTS_PATH}/{model_name}/trace.nc")
        else:
            trace.to_netcdf(f"{OUTPUTS_PATH}/{model_name}/{model_specs['year']}/trace.nc")
        progress_bar.update(1)

        # Save model run summary and workflow run summary
        utils.save_summary(workflow_filename, trace, model_name, year=model_specs["year"], OUTPUTS_PATH=OUTPUTS_PATH)
        sampling_time = round(progress_bar.format_dict["elapsed"], 2)
        run_summary = pd.DataFrame.from_dict({model_name: {"year": model_specs["year"],
                                                           "model_type": model_specs["model_type"],
                                                           "dimensions": model_specs["dimensions"],
                                                           "iterations": model_specs["run_settings"]["iterations"],
                                                           "batch_size": model_specs["run_settings"]["batch_size"],
                                                           "divergences": divergences,
                                                           "max_rhat": max_rhat,
                                                           "sampling_time": sampling_time}},
                                                orient="index")
        if os.path.exists(f"{OUTPUTS_PATH}/{workflow_filename}_summary.csv"):
            workflow_summary = pd.read_csv(f"{OUTPUTS_PATH}/{workflow_filename}_summary.csv", index_col=0)
            workflow_summary = pd.concat([workflow_summary, run_summary])
        else:
            workflow_summary = run_summary.copy()
        workflow_summary.to_csv(f"{OUTPUTS_PATH}/{workflow_filename}_summary.csv")
        progress_bar.update(1)
        os.system('clear')
        print(workflow_summary)



    
