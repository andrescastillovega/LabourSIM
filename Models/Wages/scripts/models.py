import jax.numpy as jnp
import numpyro
import utils
from numpyro import distributions as dist

def pooled(features,
           feature_names,
           target,
           from_posterior = None,
           **init_params_kwargs):
    # Initial parameters
    prior_dist = init_params_kwargs.get("prior_dist", "normal")
    avg_salary_params = init_params_kwargs.get("avg_salary_params", {"loc": 10, "scale": 1})
    prior_params = init_params_kwargs.get("prior_params", {"loc": 0, "scale": 1})
    shape_dist = init_params_kwargs.get("shape_dist", "uniform")
    shape_params = init_params_kwargs.get("shape_params", {"low": 1, "high": 100})
    target_dist = init_params_kwargs.get("target_dist", "gamma")

    # Priors and expected value
    if from_posterior is None:
        avg_salary = numpyro.sample("avg_salary", utils.DISTRIBUTIONS[prior_dist](**avg_salary_params))
        mu = avg_salary
        for feature, name in zip(features.T, feature_names):
            prior = numpyro.sample(f"beta_{name}", utils.DISTRIBUTIONS[prior_dist](**prior_params))
            mu += prior * feature
    else:
        avg_salary = numpyro.sample("avg_salary", utils.DISTRIBUTIONS["custom"](from_posterior["avg_salary"]))
        mu = avg_salary
        for feature, name in zip(features.T, feature_names):
            prior = numpyro.sample(f"beta_{name}", utils.DISTRIBUTIONS["custom"](from_posterior[f"beta_{name}"]))
            mu += prior * feature
    shape = numpyro.sample("shape", utils.DISTRIBUTIONS[shape_dist](**shape_params))

    mu = jnp.exp(mu)
    rate = shape / mu

    # Likelihood
    with numpyro.plate("data", features.shape[0]):
        numpyro.sample("salary_hat", utils.DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=target)
    
    
def hierarchical(features,
                 features_names,
                 dimension,
                 dimension_name,
                 target,
                 argmax_dim,
                 from_posterior = None,
                 **init_params_kwargs):
    # Initial parameters
    mu_dist = init_params_kwargs.get("mu_dist", "normal")
    mu_avg_salary_params = init_params_kwargs.get("mu_avg_salary_params", {"loc": 10, "scale": 3})
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
        mu_avg_salary = numpyro.sample("mu_avg_salary", utils.DISTRIBUTIONS[mu_dist](**mu_avg_salary_params))
        sigma_avg_salary = numpyro.sample("sigma_avg_salary", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
        
        for feature in features_names:
            mus.append(numpyro.sample(f"mu_{feature}", utils.DISTRIBUTIONS[mu_dist](**mu_params)))
            sigmas.append(numpyro.sample(f"sigma_{feature}", utils.DISTRIBUTIONS[sigma_dist](**sigma_params)))
    else:
        mu_avg_salary = numpyro.sample("mu_avg_salary", utils.DISTRIBUTIONS["custom"](from_posterior["mu_avg_salary"]))
        sigma_avg_salary = numpyro.sample("sigma_avg_salary", utils.DISTRIBUTIONS["custom"](from_posterior["sigma_avg_salary"]))
        
        for feature in features_names:
            mus.append(numpyro.sample(f"mu_{feature}", utils.DISTRIBUTIONS["custom"](from_posterior[f"mu_{feature}"])))
            sigmas.append(numpyro.sample(f"sigma_{feature}", utils.DISTRIBUTIONS["custom"](from_posterior[f"sigma_{feature}"])))

    with numpyro.plate(f"{dimension_name}", argmax_dim):
        offset_avg_salary = numpyro.sample("offset_avg_salary", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary = numpyro.deterministic("avg_salary", mu_avg_salary + offset_avg_salary * sigma_avg_salary)
        priors = []
        for i, feature in enumerate(features_names):
            offset = numpyro.sample(f"offset_{feature}", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
            priors.append(numpyro.deterministic(feature, mus[i] + offset * sigmas[i]))
    shape = numpyro.sample("shape", utils.DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary[dimension]
    for i, feature in enumerate(features_names):
        mu += priors[i][dimension] * features[:,i]

    mu = jnp.exp(mu)
    rate = shape / mu

    # Likelihood
    with numpyro.plate("data", target.shape[0]):
        numpyro.sample("salary_hat", utils.DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=target)


def no_pooled(features,
              features_names,
              dimension,
              dimension_name,
              target,
              argmax_dim,
              from_posterior = None,
              **init_params_kwargs):
    # Initial parameters
    prior_dist = init_params_kwargs.get("prior_dist", "normal")
    avg_salary_params = init_params_kwargs.get("avg_salary_params", {"loc": 10, "scale": 1})
    prior_params = init_params_kwargs.get("prior_params", {"loc": 0, "scale": 1})
    shape_dist = init_params_kwargs.get("shape_dist", "uniform")
    shape_params = init_params_kwargs.get("shape_params", {"low": 1, "high": 100})
    target_dist = init_params_kwargs.get("target_dist", "gamma")

    # Priors
    if from_posterior is None:
        with numpyro.plate(f"{dimension_name}", argmax_dim):
            avg_salary = numpyro.sample("avg_salary", utils.DISTRIBUTIONS[prior_dist](**avg_salary_params))
            priors = []
            for i, feature in enumerate(features_names):
                priors.append(numpyro.sample(f"beta_{feature}", utils.DISTRIBUTIONS[prior_dist](**prior_params)))
    else:
        with numpyro.plate(f"{dimension_name}", argmax_dim):
            avg_salary = numpyro.sample("avg_salary", utils.DISTRIBUTIONS["custom"](from_posterior["avg_salary"]))
            priors = []
            for i, feature in enumerate(features_names):
                priors.append(numpyro.sample(f"beta_{feature}", utils.DISTRIBUTIONS["custom"](from_posterior[f"beta_{feature}"])))
    shape = numpyro.sample("shape", utils.DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary[dimension]
    for i, prior in enumerate(priors):
        mu += prior[dimension] * features[:,i]
    mu = jnp.exp(mu)
    rate = shape / mu

    # Likelihood
    with numpyro.plate("data", target.shape[0]):
        numpyro.sample("salary_hat", utils.DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=target)