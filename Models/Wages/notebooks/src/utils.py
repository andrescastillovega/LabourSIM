import pandas as pd
import numpy as np
import pickle
import arviz as az
from numpyro.infer import Predictive
from jax import random

from src.models import pooled, no_pooled, hierarchical, hierarchical_ind_occ, hierarchical_lognormal, hierarchical_normal

# Set random seed
rng_key = random.PRNGKey(0)

def filter_data(year, data, columns=None, occ_dim=False, samples=None, operator="=="):
    # Prepare data for running the model
    if columns is None:
        columns = ["exp","sex","no_edu","elementary_edu", "highschool_edu", "postsec_edu",
                "undergrad_edu", "graduate_edu", "age", "tenure", "union", "public_sector", "self_emp"]
        
    if samples is None:
        dataset = data.query(f'year {operator} {year}').copy()
    else:
        dataset = data.query(f'year {operator} {year}').sample(samples, random_state=0).copy()

    X = dataset[columns].values
    y = dataset["salary"].values
    ind = dataset["ind_codes"].values
    occ = dataset["occ_codes"].values
    if occ_dim:
        return X, y, ind, occ
    else:
        return X, y, ind
    
def create_model(model_type):
    if model_type == "pooled":
        return pooled
    elif model_type == "no_pooled":
        return no_pooled
    elif model_type == "hierarchical":
        return hierarchical
    elif model_type == "hierarchical_ind_occ":
        return hierarchical_ind_occ
    elif model_type == "hierarchical_lognormal":
        return hierarchical_lognormal
    elif model_type == "hierarchical_normal":
        return hierarchical_normal
    else:
        raise ValueError("Invalid model type")
    

def set_coords(mcmc, dimensions, categories, data):
    model_coords = {"coords": {dim: categories[i] for i, dim in enumerate(dimensions)}}
    model_coords["coords"]["obs"] = np.arange(0,data.shape[0])
    model_coords["dims"] = {}
    for latent_var in mcmc._states['z'].keys():
        if any(latent_var.startswith(field) for field in ["avg_","beta_"]):
            model_coords["dims"][latent_var] = ["industry"] if latent_var.endswith("ind") else ["occupation"]
    return model_coords

def export_model_outputs(mcmc, model, path, *model_params, **model_coords):
    # Export mcmc
    with open(f"{path}/model.pickle", "wb") as file:
        pickle.dump(mcmc, file)
    # Create posterior predictive samples
    predictive = Predictive(model, mcmc.get_samples())
    posterior_samples = predictive(rng_key, *model_params)
    # Add posterior predictive samples to trace
    if model_coords=={}:
        trace = az.from_numpyro(mcmc, posterior_predictive=posterior_samples)
    else:
        trace = az.from_numpyro(mcmc, posterior_predictive=posterior_samples, coords=model_coords["coords"], dims=model_coords["dims"])
    # Export trace
    trace.to_netcdf(f"{path}/trace.nc")
    # Export summary
    summary = az.summary(trace, round_to=5)
    summary.to_csv(f"{path}/summary.csv")
    # Return max Rhat
    return summary["r_hat"].max()   