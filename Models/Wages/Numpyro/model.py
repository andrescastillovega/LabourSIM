import argparse
import arviz as az
import datetime as dt
import jax
import jax.numpy as jnp
from jax import random
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from numpy.lib.recfunctions import drop_fields, append_fields
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import os
import pandas as pd
import psutil
import seaborn as sns
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
    parser.add_argument("--ndraws", help="Specify the number of draws for the sampling", type=int, default=4000)
    parser.add_argument("--ntune", help="Specify the number of tuning steps for the sampling", type=int, default=4000)
    parser.add_argument("--target_accept", help="Specify the target acceptance rate for the sampling", type=float, default=0.95)
    parser.add_argument("--ncores", help="Specify the number of CPU cores to use", type=int, default=4)
    args = parser.parse_args()

    year = args.year
    nchains = args.nchains
    ndraws = args.ndraws
    ntune = args.ntune
    target_accept = args.target_accept
    ncores = args.ncores

    # Set number of cores
    available_cores = psutil.cpu_count(logical=True)
    if ncores > available_cores:
        raise NameError(f"Number of cores specified ({ncores}) is greater than the number of available cores ({available_cores}).\n\
                        >>> Please use --ncores to specify a number less than or equal to {available_cores}.")
    else:
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(ncores)

    # Load data
    data = pd.read_csv(args.dataset)

    # Load workflow
    with open(args.workflow, 'r') as file:
        workflow = yaml.safe_load(file)

    # Run models
    for model in workflow:
        model_name, model_specs = list(model.items())[0]
        model_year = model_specs["year"]
        model_standardize_vars = model_specs["standardize_vars"]
        model_params = model_specs["parameters"]
        model_run_bar = tqdm(total=4, desc=f"Running {model_name} model", ncols=100,
                             bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}' )
        model_run_bar.update(0)
        

        # Create GammaGML object
        gamma = GammaGML(model_name, dataset=data, parameters=model_params,
                         target_var="salary", standardize_vars=model_standardize_vars, year=model_year)
        model_run_bar.update(1)
        
        # Build model
        model = gamma.build()
        model_run_bar.update(1)
        
        # Run model
        trace = gamma.run(model, chains=nchains, draws=ndraws, warmup=ntune, target_accept_prob=target_accept)
        rhat_max = utils.get_rhat_max(trace)
        model_run_bar.update(1)
        model_run_bar.set_postfix({"Max. rhat": f"{rhat_max:.3f}"})

        # Save summary
        if not os.path.exists("outputs/"):
            os.makedirs("outputs/")
        utils.save_summary(trace, model_name, year=model_year)        

        # Save trace
        if model_year is None:
            if not os.path.exists(f"outputs/{model_name}"):
                os.makedirs(f"outputs/{model_name}")
            trace.to_netcdf(f"outputs/{model_name}/trace.nc")
        else:
            if not os.path.exists(f"outputs/{model_name}/{model_year}"):
                os.makedirs(f"outputs/{model_name}/{model_year}")
            trace.to_netcdf(f"outputs/{model_name}/{model_year}/trace.nc")
        model_run_bar.update(1)

    

    
