import arviz as az
import os
import pandas as pd
from numpyro import distributions as dist

DISTRIBUTIONS = {
    "normal": dist.Normal,
    "half_normal": dist.HalfNormal,
    "student_t": dist.StudentT,
    "laplace": dist.Laplace,
    "uniform": dist.Uniform,
    "gamma": dist.Gamma,
    "lognormal": dist.LogNormal
}

def standardize_vars(data, vars):
    """
    Standardize variables by subtracting the mean and dividing by the standard deviation
    """
    for var in vars:
        data[var] = (data[var] - data[var].mean()) / data[var].std()
    return data

def get_dims(features_names, dimensions):
    """
    Get dimensions for ArviZ inference data
    """
    dims = {}
    for feature in features_names:
        dims[feature] = dimensions
    return dims

def get_rhat_max(trace):
    """
    Get maximum rhat value from ArviZ inference data
    """
    summary = az.summary(trace)
    rhat_max = summary["r_hat"].max()
    return rhat_max

def save_summary(workflow, trace, model_name, OUTPUTS_PATH, year=None):
    """
    Save model summary and compilate summary    
    """
    outputs_path = f"{OUTPUTS_PATH}"

    if os.path.exists(f"{OUTPUTS_PATH}/compilate_summary_{workflow}.csv"):
        compilate_summary = pd.read_csv(f"{OUTPUTS_PATH}/compilate_summary_{workflow}.csv", index_col=0)
    else:
        compilate_summary = pd.DataFrame()

    summary = az.summary(trace)
    summary["model"] = model_name
    if year is None:
        summary["year"] = "all"
        summary.to_csv(f"{OUTPUTS_PATH}/{model_name}/summary.csv", index=True)
    else:
        summary["year"] = year
        summary.to_csv(f"{OUTPUTS_PATH}/{model_name}/{year}/summary.csv", index=True)

    compilate_summary = pd.concat([compilate_summary, summary])
    compilate_summary.to_csv(f"{OUTPUTS_PATH}/compilate_summary_{workflow}.csv", index=True)
