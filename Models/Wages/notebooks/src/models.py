import pandas as pd
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import validate_sample
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
import jax
from jax import random
from jax.scipy.stats import gaussian_kde
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import os
import pickle
import yaml

import warnings
warnings.filterwarnings('ignore')

# Create random seed for JAX
rng_key = random.PRNGKey(0)

# Define distributions
DISTRIBUTIONS = {
    "normal": dist.Normal,
    "half_normal": dist.HalfNormal,
    "student_t": dist.StudentT,
    "laplace": dist.Laplace,
    "uniform": dist.Uniform,
    "gamma": dist.Gamma,
    "log-normal": dist.LogNormal,
    "exponential": dist.Exponential,
}

# Define models
def pooled(X, y, ind, features_names, from_posterior=None, **init_params_kwargs):
    prior_dist = init_params_kwargs.get("prior_dist", "normal")
    prior_params = init_params_kwargs.get("prior_params", {"loc": 0, "scale": 1})
    shape_dist = init_params_kwargs.get("shape_dist", "uniform")
    shape_params = init_params_kwargs.get("shape_params", {"low": 1, "high": 100})
    target_dist = init_params_kwargs.get("target_dist", "gamma")

    if from_posterior is None:
        avg_salary = numpyro.sample("avg_salary", DISTRIBUTIONS[prior_dist](**prior_params))
        priors = []
        for i, feature in enumerate(features_names):
            priors.append(numpyro.sample(f"beta_{feature}", DISTRIBUTIONS[prior_dist](**prior_params)))
    else:
        avg_salary = numpyro.sample("avg_salary", DISTRIBUTIONS[prior_dist](from_posterior["avg_salary"].mean(), from_posterior["avg_salary"].std()))
        priors = []
        for i, feature in enumerate(features_names):
            priors.append(numpyro.sample(f"beta_{feature}", DISTRIBUTIONS[prior_dist](from_posterior[f"beta_{feature}"].mean(), from_posterior[f"beta_{feature}"].std())))
    shape = numpyro.sample("shape", DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary
    for i, prior in enumerate(priors):
        mu += prior * X[:,i]
    mu = jnp.exp(mu)
    rate = shape / mu

    # Likelihood
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=y)

def no_pooled(X, y, ind, features_names, from_posterior=None, **init_params_kwargs):
    # Initial parameters
    prior_dist = init_params_kwargs.get("prior_dist", "normal")
    prior_params = init_params_kwargs.get("prior_params", {"loc": 0, "scale": 1})
    shape_dist = init_params_kwargs.get("shape_dist", "uniform")
    shape_params = init_params_kwargs.get("shape_params", {"low": 1, "high": 100})
    target_dist = init_params_kwargs.get("target_dist", "gamma")

    # Priors
    if from_posterior is None:
        with numpyro.plate("industry", 16):
            avg_salary = numpyro.sample("avg_salary", DISTRIBUTIONS[prior_dist](**prior_params))
            priors = []
            for i, feature in enumerate(features_names):
                priors.append(numpyro.sample(f"beta_{feature}", DISTRIBUTIONS[prior_dist](**prior_params)))
    else:
        with numpyro.plate("industry", 16):
            avg_salary = numpyro.sample("avg_salary", 
                                        DISTRIBUTIONS[prior_dist](from_posterior["avg_salary"].mean(axis=0), from_posterior["avg_salary"].std(axis=0)))
            priors = []
            for i, feature in enumerate(features_names):
                priors.append(numpyro.sample(f"beta_{feature}", 
                                             DISTRIBUTIONS[prior_dist](from_posterior[f"beta_{feature}"].mean(axis=0), from_posterior[f"beta_{feature}"].std(axis=0))))
    shape = numpyro.sample("shape", DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary[ind]
    for i, prior in enumerate(priors):
        mu += prior[ind] * X[:,i]
    mu = jnp.exp(mu)
    rate = shape / mu

    # Likelihood
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=y)

def hierarchical(X, y, ind, features_names, from_posterior=None, **init_params_kwargs):
    # Initial parameters
    mu_dist = init_params_kwargs.get("mu_dist", "normal")
    mu_params = init_params_kwargs.get("mu_params", {"loc": 0, "scale": 3})
    sigma_dist = init_params_kwargs.get("sigma_dist", "half_normal")
    sigma_params = init_params_kwargs.get("sigma_params", {"scale": 3})
    shape_dist = init_params_kwargs.get("shape_dist", "uniform")
    shape_params = init_params_kwargs.get("shape_params", {"low": 1, "high": 100})
    target_dist = init_params_kwargs.get("target_dist", "gamma")

    # Hyperpriors
    mus = []
    sigmas = []
    if from_posterior is None:
        mu_avg_salary = numpyro.sample("mu_avg_salary", DISTRIBUTIONS[mu_dist](**mu_params))
        sigma_avg_salary = numpyro.sample("sigma_avg_salary", DISTRIBUTIONS[sigma_dist](**sigma_params))
        
        for feature in features_names:
            mus.append(numpyro.sample(f"mu_{feature}", DISTRIBUTIONS[mu_dist](**mu_params)))
            sigmas.append(numpyro.sample(f"sigma_{feature}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
    else:
        mu_avg_salary = numpyro.sample("mu_avg_salary", 
                                       DISTRIBUTIONS[mu_dist](from_posterior["mu_avg_salary"].mean(axis=0), from_posterior["mu_avg_salary"].std(axis=0)))
        sigma_avg_salary = numpyro.sample("sigma_avg_salary", 
                                          DISTRIBUTIONS[sigma_dist](from_posterior["sigma_avg_salary"].mean(axis=0)))
        
        for feature in features_names:
            mus.append(numpyro.sample(f"mu_{feature}", 
                                      DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}"].mean(axis=0), from_posterior[f"mu_{feature}"].std(axis=0))))
            sigmas.append(numpyro.sample(f"sigma_{feature}", 
                                         DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}"].mean(axis=0))))

    with numpyro.plate(f"industry", 16):
        offset_avg_salary = numpyro.sample("offset_avg_salary", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary = numpyro.deterministic("avg_salary", mu_avg_salary + offset_avg_salary * sigma_avg_salary)
        priors = []
        for i, feature in enumerate(features_names):
            offset = numpyro.sample(f"offset_{feature}", DISTRIBUTIONS["normal"](loc=0, scale=1))
            priors.append(numpyro.deterministic(f"beta_{feature}", mus[i] + offset * sigmas[i]))
        shape = numpyro.sample("shape", DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary[ind]
    for i, feature in enumerate(features_names):
        mu += priors[i][ind] * X[:,i]

    mu = jnp.exp(mu)
    rate = shape[ind] / mu

    # Likelihood
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](concentration=shape[ind], rate=rate), obs=y)

def hierarchical_ind_occ(X, y, ind, occ, features_names, from_posterior=None, **init_params_kwargs):
    # Initial parameters
    mu_dist = init_params_kwargs.get("mu_dist", "normal")
    mu_params = init_params_kwargs.get("mu_params", {"loc": 0, "scale": 3})
    sigma_dist = init_params_kwargs.get("sigma_dist", "half_normal")
    sigma_params = init_params_kwargs.get("sigma_params", {"scale": 3})
    shape_dist = init_params_kwargs.get("shape_dist", "uniform")
    shape_params = init_params_kwargs.get("shape_params", {"low": 1, "high": 100})
    target_dist = init_params_kwargs.get("target_dist", "gamma")

    # Hyperpriors
    mus_ind = []
    sigmas_ind = []
    mus_occ = []
    sigmas_occ = []
    for dim in ["ind", "occ"]:
        if from_posterior is None:
            if dim == "ind":
                mu_avg_salary_ind = numpyro.sample(f"mu_avg_salary_ind", DISTRIBUTIONS[mu_dist](**mu_params))
                sigma_avg_salary_ind = numpyro.sample(f"sigma_avg_salary_ind", DISTRIBUTIONS[sigma_dist](**sigma_params))
                for feature in features_names:
                    mus_ind.append(numpyro.sample(f"mu_{feature}_{dim}", DISTRIBUTIONS[mu_dist](**mu_params)))
                    sigmas_ind.append(numpyro.sample(f"sigma_{feature}_{dim}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
            else:
                mu_avg_salary_occ = numpyro.sample(f"mu_avg_salary_occ", DISTRIBUTIONS[mu_dist](**mu_params))
                sigma_avg_salary_occ = numpyro.sample(f"sigma_avg_salary_occ", DISTRIBUTIONS[sigma_dist](**sigma_params))
                for feature in features_names:
                    mus_occ.append(numpyro.sample(f"mu_{feature}_{dim}", DISTRIBUTIONS[mu_dist](**mu_params)))
                    sigmas_occ.append(numpyro.sample(f"sigma_{feature}_{dim}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
            
        else:
            if dim == "ind":
                mu_avg_salary_ind = numpyro.sample(f"mu_avg_salary_ind", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_avg_salary_ind"].mean(axis=0), 
                                                                   from_posterior[f"mu_avg_salary_ind"].std(axis=0)))
                sigma_avg_salary_ind = numpyro.sample(f"sigma_avg_salary_ind", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_avg_salary_ind"].mean(axis=0)))
                for feature in features_names:
                    mus_ind.append(numpyro.sample(f"mu_{feature}_{dim}", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}_{dim}"].mean(axis=0), 
                                                                   from_posterior[f"mu_{feature}_{dim}"].std(axis=0))))
                    sigmas_ind.append(numpyro.sample(f"sigma_{feature}_{dim}", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}_{dim}"].mean(axis=0))))
            else:
                mu_avg_salary_occ = numpyro.sample(f"mu_avg_salary_occ", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_avg_salary_occ"].mean(axis=0), 
                                                                   from_posterior[f"mu_avg_salary_occ"].std(axis=0)))
                sigma_avg_salary_occ = numpyro.sample(f"sigma_avg_salary_occ", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_avg_salary_occ"].mean(axis=0)))
                for feature in features_names:
                    mus_occ.append(numpyro.sample(f"mu_{feature}_{dim}", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}_{dim}"].mean(axis=0), 
                                                                   from_posterior[f"mu_{feature}_{dim}"].std(axis=0))))
                    sigmas_occ.append(numpyro.sample(f"sigma_{feature}_{dim}", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}_{dim}"].mean(axis=0))))
            
    priors_ind = []
    priors_occ = []
    with numpyro.plate(f"industry", 16):
        offset_avg_salary_ind = numpyro.sample(f"offset_avg_salary_ind", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary_ind = numpyro.deterministic(f"avg_salary_ind", mu_avg_salary_ind + offset_avg_salary_ind * sigma_avg_salary_ind)
        for i, feature in enumerate(features_names):
            offset = numpyro.sample(f"offset_{feature}_ind", DISTRIBUTIONS["normal"](loc=0, scale=1))
            priors_ind.append(numpyro.deterministic(f"beta_{feature}_ind", mus_ind[i] + offset * sigmas_ind[i]))
        shape_ind = numpyro.sample("shape_ind", DISTRIBUTIONS[shape_dist](**shape_params))
    
    with numpyro.plate(f"occupation", 24):
        offset_avg_salary_occ = numpyro.sample(f"offset_avg_salary_occ", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary_occ = numpyro.deterministic(f"avg_salary_occ", mu_avg_salary_occ + offset_avg_salary_occ * sigma_avg_salary_occ)
        for i, feature in enumerate(features_names):
            offset = numpyro.sample(f"offset_{feature}_occ", DISTRIBUTIONS["normal"](loc=0, scale=1))
            priors_occ.append(numpyro.deterministic(f"beta_{feature}_occ", mus_occ[i] + offset * sigmas_occ[i]))
        shape_occ = numpyro.sample("shape_occ", DISTRIBUTIONS[shape_dist](**shape_params))


    # Expected value
    mu = avg_salary_ind[ind] + avg_salary_occ[occ]
    for i, feature in enumerate(features_names):
        mu += priors_ind[i][ind] * X[:,i] + priors_occ[i][occ] * X[:,i]

    shape = shape_ind[ind] + shape_occ[occ]

    mu = jnp.exp(mu)
    rate = shape / mu

    # Likelihood
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=y)

def hierarchical_ind_occ_orig(X, y, ind, occ, features_names, from_posterior=None, **init_params_kwargs):
    # Initial parameters
    mu_dist = init_params_kwargs.get("mu_dist", "normal")
    mu_params = init_params_kwargs.get("mu_params", {"loc": 0, "scale": 3})
    sigma_dist = init_params_kwargs.get("sigma_dist", "half_normal")
    sigma_params = init_params_kwargs.get("sigma_params", {"scale": 3})
    shape_dist = init_params_kwargs.get("shape_dist", "uniform")
    shape_params = init_params_kwargs.get("shape_params", {"low": 1, "high": 100})
    target_dist = init_params_kwargs.get("target_dist", "gamma")

    # Hyperpriors
    mus_ind = []
    sigmas_ind = []
    mus_occ = []
    sigmas_occ = []
    for dim in ["ind", "occ"]:
        if from_posterior is None:
            if dim == "ind":
                mu_avg_salary_ind = numpyro.sample(f"mu_avg_salary_ind", DISTRIBUTIONS[mu_dist](**mu_params))
                sigma_avg_salary_ind = numpyro.sample(f"sigma_avg_salary_ind", DISTRIBUTIONS[sigma_dist](**sigma_params))
                for feature in features_names:
                    mus_ind.append(numpyro.sample(f"mu_{feature}_{dim}", DISTRIBUTIONS[mu_dist](**mu_params)))
                    sigmas_ind.append(numpyro.sample(f"sigma_{feature}_{dim}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
            else:
                mu_avg_salary_occ = numpyro.sample(f"mu_avg_salary_occ", DISTRIBUTIONS[mu_dist](**mu_params))
                sigma_avg_salary_occ = numpyro.sample(f"sigma_avg_salary_occ", DISTRIBUTIONS[sigma_dist](**sigma_params))
                for feature in features_names:
                    mus_occ.append(numpyro.sample(f"mu_{feature}_{dim}", DISTRIBUTIONS[mu_dist](**mu_params)))
                    sigmas_occ.append(numpyro.sample(f"sigma_{feature}_{dim}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
            
        else:
            if dim == "ind":
                mu_avg_salary_ind = numpyro.sample(f"mu_avg_salary_ind", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_avg_salary_ind"].mean(axis=0), 
                                                                   from_posterior[f"mu_avg_salary_ind"].std(axis=0)))
                sigma_avg_salary_ind = numpyro.sample(f"sigma_avg_salary_ind", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_avg_salary_ind"].mean(axis=0)))
                for feature in features_names:
                    mus_ind.append(numpyro.sample(f"mu_{feature}_{dim}", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}_{dim}"].mean(axis=0), 
                                                                   from_posterior[f"mu_{feature}_{dim}"].std(axis=0))))
                    sigmas_ind.append(numpyro.sample(f"sigma_{feature}_{dim}", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}_{dim}"].mean(axis=0))))
            else:
                mu_avg_salary_occ = numpyro.sample(f"mu_avg_salary_occ", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_avg_salary_occ"].mean(axis=0), 
                                                                   from_posterior[f"mu_avg_salary_occ"].std(axis=0)))
                sigma_avg_salary_occ = numpyro.sample(f"sigma_avg_salary_occ", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_avg_salary_occ"].mean(axis=0)))
                for feature in features_names:
                    mus_occ.append(numpyro.sample(f"mu_{feature}_{dim}", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}_{dim}"].mean(axis=0), 
                                                                   from_posterior[f"mu_{feature}_{dim}"].std(axis=0))))
                    sigmas_occ.append(numpyro.sample(f"sigma_{feature}_{dim}", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}_{dim}"].mean(axis=0))))
            
    priors_ind = []
    priors_occ = []
    with numpyro.plate(f"industry", 16):
        offset_avg_salary_ind = numpyro.sample(f"offset_avg_salary_ind", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary_ind = numpyro.deterministic(f"avg_salary_ind", mu_avg_salary_ind + offset_avg_salary_ind * sigma_avg_salary_ind)
        for i, feature in enumerate(features_names):
            offset = numpyro.sample(f"offset_{feature}_ind", DISTRIBUTIONS["normal"](loc=0, scale=1))
            priors_ind.append(numpyro.deterministic(f"beta_{feature}_ind", mus_ind[i] + offset * sigmas_ind[i]))
    
    with numpyro.plate(f"occupation", 24):
        offset_avg_salary_occ = numpyro.sample(f"offset_avg_salary_occ", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary_occ = numpyro.deterministic(f"avg_salary_occ", mu_avg_salary_occ + offset_avg_salary_occ * sigma_avg_salary_occ)
        for i, feature in enumerate(features_names):
            offset = numpyro.sample(f"offset_{feature}_occ", DISTRIBUTIONS["normal"](loc=0, scale=1))
            priors_occ.append(numpyro.deterministic(f"beta_{feature}_occ", mus_ind[i] + offset * sigmas_ind[i]))

    shape = numpyro.sample("shape", DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary_ind[ind] + avg_salary_occ[occ]
    for i, feature in enumerate(features_names):
        mu += priors_ind[i][ind] * X[:,i] + priors_occ[i][occ] * X[:,i]

    mu = jnp.exp(mu)
    rate = shape / mu

    # Likelihood
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=y)

def hierarchical_lognormal(X, y, ind, occ, features_names, from_posterior=None, **init_params_kwargs):
    # Initial parameters
    mu_dist = init_params_kwargs.get("mu_dist", "normal")
    mu_params = init_params_kwargs.get("mu_params", {"loc": 0, "scale": 3})
    sigma_dist = init_params_kwargs.get("sigma_dist", "half_normal")
    sigma_params = init_params_kwargs.get("sigma_params", {"scale": 3})
    shape_dist = init_params_kwargs.get("shape_dist", "exponential")
    shape_params = init_params_kwargs.get("shape_params", {"rate": 1})
    target_dist = init_params_kwargs.get("target_dist", "log-normal")

    # Hyperpriors
    mus_ind = []
    sigmas_ind = []
    mus_occ = []
    sigmas_occ = []
    for dim in ["ind", "occ"]:
        if from_posterior is None:
            if dim == "ind":
                mu_avg_salary_ind = numpyro.sample(f"mu_avg_salary_ind", DISTRIBUTIONS[mu_dist](**mu_params))
                sigma_avg_salary_ind = numpyro.sample(f"sigma_avg_salary_ind", DISTRIBUTIONS[sigma_dist](**sigma_params))
                for feature in features_names:
                    mus_ind.append(numpyro.sample(f"mu_{feature}_{dim}", DISTRIBUTIONS[mu_dist](**mu_params)))
                    sigmas_ind.append(numpyro.sample(f"sigma_{feature}_{dim}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
            else:
                mu_avg_salary_occ = numpyro.sample(f"mu_avg_salary_occ", DISTRIBUTIONS[mu_dist](**mu_params))
                sigma_avg_salary_occ = numpyro.sample(f"sigma_avg_salary_occ", DISTRIBUTIONS[sigma_dist](**sigma_params))
                for feature in features_names:
                    mus_occ.append(numpyro.sample(f"mu_{feature}_{dim}", DISTRIBUTIONS[mu_dist](**mu_params)))
                    sigmas_occ.append(numpyro.sample(f"sigma_{feature}_{dim}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
            
        else:
            if dim == "ind":
                mu_avg_salary_ind = numpyro.sample(f"mu_avg_salary_ind", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_avg_salary_ind"].mean(axis=0), from_posterior[f"mu_avg_salary_ind"].std(axis=0)))
                sigma_avg_salary_ind = numpyro.sample(f"sigma_avg_salary_ind", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_avg_salary_ind"].mean(axis=0)))
                for feature in features_names:
                    mus_ind.append(numpyro.sample(f"mu_{feature}_{dim}", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}_{dim}"].mean(axis=0), from_posterior[f"mu_{feature}_{dim}"].std(axis=0))))
                    sigmas_ind.append(numpyro.sample(f"sigma_{feature}_{dim}", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}_{dim}"].mean(axis=0))))
            else:
                mu_avg_salary_occ = numpyro.sample(f"mu_avg_salary_occ", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_avg_salary_occ"].mean(axis=0), from_posterior[f"mu_avg_salary_occ"].std(axis=0)))
                sigma_avg_salary_occ = numpyro.sample(f"sigma_avg_salary_occ", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_avg_salary_occ"].mean(axis=0)))
                for feature in features_names:
                    mus_occ.append(numpyro.sample(f"mu_{feature}_{dim}", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}_{dim}"].mean(axis=0), from_posterior[f"mu_{feature}_{dim}"].std(axis=0))))
                    sigmas_occ.append(numpyro.sample(f"sigma_{feature}_{dim}", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}_{dim}"].mean(axis=0))))
            
    priors_ind = []
    priors_occ = []
    with numpyro.plate(f"industry", 16):
        offset_avg_salary_ind = numpyro.sample(f"offset_avg_salary_ind", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary_ind = numpyro.deterministic(f"avg_salary_ind", mu_avg_salary_ind + offset_avg_salary_ind * sigma_avg_salary_ind)
        for i, feature in enumerate(features_names):
                offset = numpyro.sample(f"offset_{feature}_ind", DISTRIBUTIONS["normal"](loc=0, scale=1))
                priors_ind.append(numpyro.deterministic(f"beta_{feature}_ind", mus_ind[i] + offset * sigmas_ind[i]))
    
    with numpyro.plate(f"occupation", 24):
        offset_avg_salary_occ = numpyro.sample(f"offset_avg_salary_occ", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary_occ = numpyro.deterministic(f"avg_salary_occ", mu_avg_salary_occ + offset_avg_salary_occ * sigma_avg_salary_occ)
        for i, feature in enumerate(features_names):
                offset = numpyro.sample(f"offset_{feature}_occ", DISTRIBUTIONS["normal"](loc=0, scale=1))
                priors_occ.append(numpyro.deterministic(f"beta_{feature}_occ", mus_ind[i] + offset * sigmas_ind[i]))

    sigma = numpyro.sample("shape", DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary_ind[ind] + avg_salary_occ[occ]
    for i, feature in enumerate(features_names):
        mu += priors_ind[i][ind] * X[:,i] + priors_occ[i][occ] * X[:,i]

    # Likelihood
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](loc=mu, scale=sigma), obs=y)

def hierarchical_normal(X, y, ind, occ, features_names, from_posterior=None, **init_params_kwargs):
    # Initial parameters
    mu_dist = init_params_kwargs.get("mu_dist", "normal")
    mu_params = init_params_kwargs.get("mu_params", {"loc": 0, "scale": 3})
    sigma_dist = init_params_kwargs.get("sigma_dist", "half_normal")
    sigma_params = init_params_kwargs.get("sigma_params", {"scale": 3})
    shape_dist = init_params_kwargs.get("shape_dist", "half_normal")
    shape_params = init_params_kwargs.get("shape_params", {"rate": 10})
    target_dist = init_params_kwargs.get("target_dist", "normal")

    # Hyperpriors
    mus_ind = []
    sigmas_ind = []
    mus_occ = []
    sigmas_occ = []
    for dim in ["ind", "occ"]:
        if from_posterior is None:
            if dim == "ind":
                mu_avg_salary_ind = numpyro.sample(f"mu_avg_salary_ind", DISTRIBUTIONS[mu_dist](**mu_params))
                sigma_avg_salary_ind = numpyro.sample(f"sigma_avg_salary_ind", DISTRIBUTIONS[sigma_dist](**sigma_params))
                for feature in features_names:
                    mus_ind.append(numpyro.sample(f"mu_{feature}_{dim}", DISTRIBUTIONS[mu_dist](**mu_params)))
                    sigmas_ind.append(numpyro.sample(f"sigma_{feature}_{dim}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
            else:
                mu_avg_salary_occ = numpyro.sample(f"mu_avg_salary_occ", DISTRIBUTIONS[mu_dist](**mu_params))
                sigma_avg_salary_occ = numpyro.sample(f"sigma_avg_salary_occ", DISTRIBUTIONS[sigma_dist](**sigma_params))
                for feature in features_names:
                    mus_occ.append(numpyro.sample(f"mu_{feature}_{dim}", DISTRIBUTIONS[mu_dist](**mu_params)))
                    sigmas_occ.append(numpyro.sample(f"sigma_{feature}_{dim}", DISTRIBUTIONS[sigma_dist](**sigma_params)))
            
        else:
            if dim == "ind":
                mu_avg_salary_ind = numpyro.sample(f"mu_avg_salary_ind", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_avg_salary_ind"].mean(axis=0), from_posterior[f"mu_avg_salary_ind"].std(axis=0)))
                sigma_avg_salary_ind = numpyro.sample(f"sigma_avg_salary_ind", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_avg_salary_ind"].mean(axis=0)))
                for feature in features_names:
                    mus_ind.append(numpyro.sample(f"mu_{feature}_{dim}", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}_{dim}"].mean(axis=0), from_posterior[f"mu_{feature}_{dim}"].std(axis=0))))
                    sigmas_ind.append(numpyro.sample(f"sigma_{feature}_{dim}", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}_{dim}"].mean(axis=0))))
            else:
                mu_avg_salary_occ = numpyro.sample(f"mu_avg_salary_occ", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_avg_salary_occ"].mean(axis=0), from_posterior[f"mu_avg_salary_occ"].std(axis=0)))
                sigma_avg_salary_occ = numpyro.sample(f"sigma_avg_salary_occ", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_avg_salary_occ"].mean(axis=0)))
                for feature in features_names:
                    mus_occ.append(numpyro.sample(f"mu_{feature}_{dim}", 
                                            DISTRIBUTIONS[mu_dist](from_posterior[f"mu_{feature}_{dim}"].mean(axis=0), from_posterior[f"mu_{feature}_{dim}"].std(axis=0))))
                    sigmas_occ.append(numpyro.sample(f"sigma_{feature}_{dim}", 
                                                DISTRIBUTIONS[sigma_dist](from_posterior[f"sigma_{feature}_{dim}"].mean(axis=0))))
            
    priors_ind = []
    priors_occ = []
    with numpyro.plate(f"industry", 16):
        offset_avg_salary_ind = numpyro.sample(f"offset_avg_salary_ind", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary_ind = numpyro.deterministic(f"avg_salary_ind", mu_avg_salary_ind + offset_avg_salary_ind * sigma_avg_salary_ind)
        for i, feature in enumerate(features_names):
                offset = numpyro.sample(f"offset_{feature}_ind", DISTRIBUTIONS["normal"](loc=0, scale=1))
                priors_ind.append(numpyro.deterministic(f"beta_{feature}_ind", mus_ind[i] + offset * sigmas_ind[i]))
    
    with numpyro.plate(f"occupation", 24):
        offset_avg_salary_occ = numpyro.sample(f"offset_avg_salary_occ", DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary_occ = numpyro.deterministic(f"avg_salary_occ", mu_avg_salary_occ + offset_avg_salary_occ * sigma_avg_salary_occ)
        for i, feature in enumerate(features_names):
                offset = numpyro.sample(f"offset_{feature}_occ", DISTRIBUTIONS["normal"](loc=0, scale=1))
                priors_occ.append(numpyro.deterministic(f"beta_{feature}_occ", mus_ind[i] + offset * sigmas_ind[i]))

    sigma = numpyro.sample("shape", DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary_ind[ind] + avg_salary_occ[occ]
    for i, feature in enumerate(features_names):
        mu += priors_ind[i][ind] * X[:,i] + priors_occ[i][occ] * X[:,i]

    # Likelihood
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](loc=mu, scale=sigma), obs=y)