import arviz as az
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import pickle
import xarray as xr

from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import log_likelihood

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


def concat_samples(samples):
    """
    Concatenate samples when sampling in batches    
    """
    for key in samples[0]:
        samples[0][key] = jnp.concatenate([samples[0][key], samples[1][key]], axis=1)
    return samples[0]


def save_model_pickle(mcmc, outputs_path):
    """
    Save model pickle
    """
    with open(fr"{outputs_path}/model.pickle", "wb") as output_file:
        pickle.dump(mcmc, output_file)

def concat_samples(samples):
    for key in samples[0]:
        samples[0][key] = jnp.concatenate([samples[0][key], samples[1][key]], axis=1)
    return samples[0]


def create_inference_data(mcmc,
                          samples, 
                          divergences, 
                          loglikelihood, 
                          dimension_name, 
                          dimension,
                          target):
    trace = az.from_numpyro(mcmc)
    chains, draws, dim = samples["avg_salary"].shape
    obs = target.shape[0]

    posterior_dataset = xr.Dataset(
        data_vars = { var: (["chain", "draw", f"{dimension_name}"], samples[var]) 
                     if len(samples[var].shape)==3 else (["chain", "draw"], samples[var])
                     for var in samples.keys()  },
        coords = { "chain": np.arange(chains), "draw": np.arange(draws), f"{dimension_name}": dimension },
        attrs = trace.posterior.attrs,
    )
    loglike_dataset = xr.Dataset(
        data_vars=dict(log_likelihood=(["chain", "draw", "obs"], loglikelihood)),
        coords=dict(chain=np.arange(chains), draw=np.arange(draws), obs=np.arange(obs)),
        attrs=trace.log_likelihood.attrs
    )
    sample_stats = xr.Dataset(
        data_vars=dict(diverging=(["chain", "draw"], divergences)),
        coords=dict(chain=np.arange(chains),draw=np.arange(draws)),
        attrs=trace.sample_stats.attrs,
    )
    observed_data = xr.Dataset(
        data_vars=dict(salary=(["obs"], target)),
        coords=dict(obs=np.arange(obs)),
        attrs=trace.observed_data.attrs,
    )

    return az.InferenceData(posterior=posterior_dataset,
                 log_likelihood=loglike_dataset,
                 sample_stats=sample_stats,
                 observed_data=observed_data)

def create_mcmc(model, warmup, samples, chains, target_accept=0.95, last_sample=None):
    kernel = NUTS(model, target_accept_prob=target_accept, init_strategy=init_to_median(num_samples=100))
    return MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains, chain_method='vectorized', progress_bar=True)

def get_batch_results(model, mcmc, samples, divergences, loglikehood, iteration, chains, nsamples, obs):    
    if iteration == 0:
        samples = mcmc.get_samples(group_by_chain=True)
        divergences = mcmc.get_extra_fields(group_by_chain=True)["diverging"]
        loglikehood = log_likelihood(model = model,
                                     posterior_samples = mcmc.get_samples(), 
                                     batch_ndims = 1)["salary_hat"]\
                                     .reshape(chains, nsamples, obs)
    else:
        samples = concat_samples([samples, mcmc.get_samples(group_by_chain=True)])
        divergences = jnp.concatenate([divergences, mcmc.get_extra_fields(group_by_chain=True)["diverging"]], axis=1)
        loglikehood = jnp.concatenate([loglikehood,log_likelihood(model = model,
                                                                  posterior_samples = mcmc.get_samples(),
                                                                  batch_ndims = 1)["salary_hat"]\
                                                                  .reshape(chains, nsamples, obs)], axis=1)
    sample_rhat = az.summary(az.from_dict(mcmc.get_samples(group_by_chain=True)))["r_hat"].max()
    cumulative_rhat = az.summary(az.from_dict(samples),round_to=5)["r_hat"].max()
    return samples, divergences, loglikehood, sample_rhat, cumulative_rhat