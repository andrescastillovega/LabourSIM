import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.distributions.util import validate_sample
import jax
from jax import random
from jax import numpy as jnp
import numpy as np
from numpy.lib.recfunctions import drop_fields, append_fields
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az
import datetime

jax.config.update("jax_enable_x64", True)

data = pd.read_csv("../datasets/model_dataset.csv")

# Drop missing values
data = data.dropna(subset="industry")
data = data.dropna(subset="occup")
data = data.dropna(subset="exp")
data = data.dropna(subset="salary")

data = data.query("year == 1996")
data = data[['year','industry', 'occup', 'exp','salary',"sex",'elementary_edu','highschool_edu','postsec_edu','undergrad_edu','graduate_edu',
                'grad_highschool_refyear','grad_college_refyear','grad_univ_refyear','tenure','union','part_time',
                'public_sector','self_emp','age','firm_size','loc_size']].copy()

data.head(3)
dataset = data.copy()

dataset["exp"] = (dataset["exp"] - dataset["exp"].mean()) / dataset["exp"].std()
dataset["age"] = (dataset["age"] - dataset["age"].mean()) / dataset["age"].std()
dataset["tenure"] = (dataset["tenure"] - dataset["tenure"].mean()) / dataset["tenure"].std()

features = dataset[['exp','sex']].values
target = dataset['salary'].values
industry = pd.factorize(dataset['industry'])[0]

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

DISTRIBUTIONS = {
    "normal": dist.Normal,
    "half_normal": dist.HalfNormal,
    "student_t": dist.StudentT,
    "laplace": dist.Laplace,
    "uniform": dist.Uniform,
    "gamma": dist.Gamma,
    "lognormal": dist.LogNormal
}

def hierarchical(features, target, industry, prior_dist="normal", prior_params={"loc": 0, "scale": 1}, shape_dist="uniform", shape_params={"low": 1, "high": 100}, target_dist="gamma"):
    # Hyperpriors
    mu_avg_salary = numpyro.sample("mu_avg_salary", dist.Normal(loc=0, scale=1))
    sigma_avg_salary = numpyro.sample("sigma_avg_salary", dist.HalfNormal(scale=1))
    mu_exp = numpyro.sample("mu_exp", dist.Normal(loc=0, scale=1))
    sigma_exp = numpyro.sample("sigma_exp", dist.HalfNormal(scale=1))
    mu_sex = numpyro.sample("sex", dist.Normal(loc=0, scale=1))
    sigma_sex = numpyro.sample("sex", dist.HalfNormal(scale=1))

    # Priors
    with numpyro.plate("industry", industry.max()+1):
        offset_avg_salary = numpyro.sample("offset_avg_salary", dist.Normal(loc=0, scale=1))
        avg_salary = numpyro.deterministic("avg_salary", mu_avg_salary + offset_avg_salary * sigma_avg_salary)
        offset_exp = numpyro.sample("offset_exp", dist.Normal(loc=0, scale=1))
        exp = numpyro.deterministic("exp", mu_exp + offset_exp * sigma_exp)
        offset_sex = numpyro.sample("offset_sex", dist.Normal(loc=0, scale=1))
        sex = numpyro.deterministic("sex", mu_sex + offset_sex * sigma_sex) 
        
    
    shape = numpyro.sample("shape", dist.Uniform(low=1, high=100))
    mu = jnp.exp(avg_salary + exp[industry] * features[:,0] + sex[industry] * features[:,1])
    rate = shape / mu
    numpyro.sample("salary_hat", dist.Gamma(concentration=shape, rate=rate), obs=target)

numpyro.render_model(hierarchical, model_args=(features, target, industry), render_distributions=True)

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
# Run NUTS
kernel = NUTS(hierarchical, target_accept_prob=0.95)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=2, chain_method='vectorized', progress_bar=True)
mcmc.run(rng_key, features, target, industry)


# Run NUTS
kernel = NUTS(hierarchical, target_accept_prob=0.95)
mcmc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=4, chain_method='vectorized', progress_bar=True)
sharding = PositionalSharding(mesh_utils.create_device_mesh((2,)))
features_shard = jax.device_put(features, sharding.reshape(2, 1))
target_shard = jax.device_put(target, sharding.reshape(2))
industry_shard = jax.device_put(industry, sharding.reshape(2))
mcmc.run(jax.random.PRNGKey(0), features_shard, target_shard, industry_shard)









def pooled(features, target, prior_dist="normal", prior_params={"loc": 0, "scale": 1}, shape_dist="half_normal", shape_params={"scale": 1}, target_dist="normal"):
    beta = numpyro.sample("beta", DISTRIBUTIONS[prior_dist](**prior_params), sample_shape=(features.shape[1],))
    sigma = numpyro.sample("shape", DISTRIBUTIONS[shape_dist](**shape_params))
    mu = jnp.dot(features, beta)
    numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](loc=mu, scale=sigma), obs=target)

numpyro.render_model(pooled, model_args=(features, target), render_distributions=True)









def pooled(features, target, prior_dist="normal", prior_params={"loc": 0, "scale": 1}, shape_dist="uniform", shape_params={"low": 1, "high": 100}, target_dist="gamma"):
    beta = numpyro.sample("beta", DISTRIBUTIONS[prior_dist](**prior_params), sample_shape=(features.shape[1],))
    shape = numpyro.sample("shape", DISTRIBUTIONS[shape_dist](**shape_params))
    mu = jnp.exp(jnp.dot(features, beta))
    rate = shape / mu
    numpyro.sample("salary_hat", DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=target)

numpyro.render_model(pooled, model_args=(features, target), render_distributions=True)

# Run NUTS
kernel = NUTS(pooled, target_accept_prob=0.95)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4, chain_method='vectorized', progress_bar=True)
print("start: ", datetime.datetime.now().strftime("%H:%M:%S"))
mcmc.run(rng_key, features, target)
print("end: ", datetime.datetime.now().strftime("%H:%M:%S"))