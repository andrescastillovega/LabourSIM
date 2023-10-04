import argparse
import jax
import numpyro
import os
import pandas as pd
import psutil
from tqdm import tqdm
import yaml

from GammaGML import GammaGML
import utils

if __name__ == "__main__":
    # Set run parameters
    parser = argparse.ArgumentParser(description="LabourSIM wage model")
    parser.add_argument("--dataset", help="Specify the .csv file with the labour market data")
    parser.add_argument("--workflow", help="Specify the .yaml file with the workflow")
    parser.add_argument("--year", help="Specify the year of analysis", type=int, default=None)
    parser.add_argument("--nchains", help="Specify the number of chains for the sampling", type=int, default=4)
    parser.add_argument("--target_accept", help="Specify the target acceptance rate for the sampling", type=float, default=0.95)
    parser.add_argument("--ngpus", help="Specify the number of GPUs to use", type=int, default=1)
    parser.add_argument("--batch_size", help="Specify the batch size for the sampling", type=int, default=None)
    parser.add_argument("--progressbar", help="Specify whether to show the progress bar", type=bool, default=True)
    args = parser.parse_args()

    year = args.year
    nchains = args.nchains
    target_accept = args.target_accept
    ngpus = args.ngpus
    workflow_filename = args.workflow.split("/")[-1].split(".")[0]

    # Set number of cores
    available_gpus = len(jax.devices("gpu"))
    if ngpus > available_gpus:
        raise NameError(f"Number of GPUs specified ({ngpus}) is greater than the number of available ({available_gpus}).\n\
                        >>> Please use --ngpus to specify a number less than or equal to {available_gpus}.")
    else:
        # numpyro.set_platform("gpu")
        # # numpyro.set_host_device_count(ncores)
        jax.config.update("jax_platform_name", "gpu")
        jax.config.update("jax_enable_x64", True)

    # Load data
    data = pd.read_csv(args.dataset)

    # Load workflow
    with open(args.workflow, 'r') as file:
        workflow = yaml.safe_load(file)
    utils.check_workflow(workflow)

    # Create workflows summary
    workflow_summary = pd.DataFrame(columns=["Year", "Draws", "Warmup", "Divergences", "MaxRhat", "SamplingTime"])
    
    # Run models
    for model in workflow:
        # Delete previous model outputs
        if os.path.exists(f"../outputs/{model}"):
            os.rmdir(f"../outputs/{model}")

        # Get model specs
        model_name, model_specs = list(model.items())[0]
        model_year = model_specs["year"]
        ndraws = model_specs["draws"]
        ntune = model_specs["warmup"]
        model_standardize_vars = model_specs["standardize_vars"]
        model_params = model_specs["parameters"]
        model_run_bar = tqdm(total=7, desc=f"Running {model_name} model", ncols=100,
                             bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}' )
        model_run_bar.update(0)
        
        # Create GammaGML object
        gamma = GammaGML(model_name, dataset=data, parameters=model_params,
                         target_var="salary", standardize_vars=model_standardize_vars, year=model_year)
        model_run_bar.update(1)
        
        # Build model
        model = gamma.build()
        model_run_bar.update(1)

        # Render model
        if model_year is None:
            if not os.path.exists(f"../outputs/{model_name}"):
                os.makedirs(f"../outputs/{model_name}")
        else:
            if not os.path.exists(f"../outputs/{model_name}/{model_year}"):
                os.makedirs(f"../outputs/{model_name}/{model_year}")
        gamma.render_model()
        model_run_bar.update(1)
        
        # Run model
        trace, divergences = gamma.run(model, chains=nchains, draws=ndraws, warmup=ntune,
                                        target_accept_prob=target_accept, batch_size=args.batch_size, progress_bar=args.progressbar)
        rhat_max = utils.get_rhat_max(trace)
        model_run_bar.update(1)
        model_run_bar.set_postfix({"Max. rhat": f"{rhat_max:.3f}"})

        # Save run summary
        utils.save_summary(workflow_filename, trace, model_name, year=model_year)
        model_run_bar.update(1)    

        # Save trace
        if model_year is None:
            trace.to_netcdf(f"../outputs/{model_name}/trace.nc")
        else:
            trace.to_netcdf(f"../outputs/{model_name}/{model_year}/trace.nc")
        model_run_bar.update(1)

        # Update workflow summary
        sampling_time = round(model_run_bar.format_dict["elapsed"], 2)
        workflow_summary = pd.concat([workflow_summary, pd.DataFrame.from_dict({model_name: {"Year": model_year, "Draws": ndraws, "Warmup": ntune,
                                                    "Divergences": divergences, "MaxRhat": rhat_max,
                                                    "SamplingTime": sampling_time}}, orient="index")])
        workflow_summary.to_csv(f"../outputs/{workflow_filename}_summary.csv")
        model_run_bar.update(1)

        

    

    