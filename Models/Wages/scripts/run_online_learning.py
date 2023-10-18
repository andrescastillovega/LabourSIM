import argparse
import jax
import numpyro
import os
import pandas as pd
import pickle
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
    with open(args.workflow, 'r') as file:
        workflow = yaml.safe_load(file)

    # Split data into training and testing (75/25)
    data = data[data["year"] <= 2007].copy()

    # Get unique years and sort them
    years = data["year"].unique()
    years.sort()

    # Run models
    for model in workflow:
        model_name, model_specs = list(model.items())[0]

        for i, year in enumerate(years):
            # Create model folder
            model_specs["year"] = year
            if model_specs["year"] is None:
                if not os.path.exists(f"{OUTPUTS_PATH}/{model_name}"):
                    os.makedirs(f"{OUTPUTS_PATH}/{model_name}")
            else:
                if not os.path.exists(f"{OUTPUTS_PATH}/{model_name}/{model_specs['year']}"):
                    os.makedirs(f"{OUTPUTS_PATH}/{model_name}/{model_specs['year']}")

            # Filter dataset by year
            if model_specs["year"] is not None:
                data_year = data[data["year"] == model_specs["year"]]
                print(f"Dataset for year {model_specs['year']} contains {data_year.shape[0]} observations")

            # Create progress bar
            progress_bar = tqdm(total=5, desc=f"Running {model_name} model", ncols=100,
                                bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}')
            
            # Extract previous posterior
            if i==0:
                from_posterior = None
            else:
                with open(f"{OUTPUTS_PATH}/{model_name}/{years[i-1]}/model.pickle", "rb") as previous_model:
                    mcmc = pickle.load(previous_model)
                from_posterior = mcmc.get_samples()

            # Create model object
            initial_params = model_specs["initial_params"] if "initial_params" in model_specs.keys() else dict()
            model = GammaGLM(model_name, 
                            model_specs["model_type"], 
                            data_year,
                            model_specs["target"], 
                            model_specs["parameters"], 
                            model_specs["dimensions"], 
                            model_specs["standardize_vars"],
                            OUTPUTS_PATH, 
                            model_specs["year"],
                            from_posterior,
                            **initial_params)
            progress_bar.update(1)

            # Build model
            model.build()
            progress_bar.update(1)

            # Run model
            trace, divergences, converg_iterations = model.run(tune=model_specs["run_settings"]["tune"],
                                            draws=model_specs["run_settings"]["draws"],
                                            chains=model_specs["run_settings"]["chains"],
                                            target_accept_prob=model_specs["run_settings"]["target_accept"],
                                            batch_size=model_specs["run_settings"]["batch_size"],
                                            iterations=model_specs["run_settings"]["iterations"])
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
                                                            "dimensions": model_specs["dimensions"][0] if model_specs["dimensions"] is not None else None,
                                                            "tune": model_specs["run_settings"]["tune"],
                                                            "draws": model_specs["run_settings"]["draws"],
                                                            "iterations": model_specs["run_settings"]["iterations"],
                                                            "batch_size": model_specs["run_settings"]["batch_size"],
                                                            "convergence_iterations": converg_iterations,
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
            progress_bar.close()
            os.system('clear')
            print(workflow_summary)



    
