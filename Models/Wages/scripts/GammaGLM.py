import arviz as az
import utils
import pandas as pd
import pickle
from models import pooled, hierarchical, no_pooled
import jax
from jax import random
from jax import numpy as jnp
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import log_likelihood
import os


class GammaGLM():
    def __init__(self, name, model_type, dataset, target, parameters, dimensions, standardize_vars, OUTPUTS_PATH, year=None, **init_params_kwargs):
        self.name = name
        self.model_type = model_type
        self.features_names = parameters
        self.dimensions_names = dimensions
        self.year = year
        self.init_params_kwargs = init_params_kwargs

        data = dataset.copy()
        data = utils.standardize_vars(data, standardize_vars)
        self.features = data[parameters].values
        self.target = data[target].values
        if dimensions is None:
            self.idx_dims = None
            self.coords = None
            self.dims = None
        else:
            self.idx_dims = pd.factorize(data[dimensions[0]])[0]
            self.coords = { dimensions[0]: list(pd.factorize(data[dimensions[0]])[1]) }
            self.dims = utils.get_dims(self.features_names, dimensions)
        if year is None:
            self.outputs_path = fr"{OUTPUTS_PATH}/{self.name}"
        else:
            self.outputs_path = fr"{OUTPUTS_PATH}/{self.name}/{self.year}"
        
    def __repr__(self):
        if self.year is not None:
            return f"Model: {self.name} - Type: {self.model_type} - Dims: {self.dimensions_names} - Params: {self.features_names} - Year: {self.year}"
        else:
            return f"Model: {self.name} - Type: {self.model_type} - Dims: {self.dimensions_names} - Params: {self.features_names}"
        
    def model_fn(self):
        features_sharded, target_sharded, idx_dims_sharded = utils.get_sharded_data(self.features, self.target, self.idx_dims)
        argmax_dim = self.idx_dims.max() + 1 if self.idx_dims is not None else None
        if self.model_type == "pooled":
            return pooled(features_sharded, self.features_names, target_sharded, **self.init_params_kwargs)
        elif self.model_type == "hierarchical":
            return hierarchical(features_sharded, 
                                self.features_names, 
                                idx_dims_sharded, 
                                self.dimensions_names, 
                                target_sharded, argmax_dim,
                                **self.init_params_kwargs)
        elif self.model_type == "no_pooled":
            return no_pooled(features_sharded, 
                             self.features_names, 
                             idx_dims_sharded, 
                             self.dimensions_names, 
                             target_sharded, 
                             argmax_dim,
                             **self.init_params_kwargs)
        else:
            raise ValueError("Invalid model type")

    def build(self):
        rng_key = random.PRNGKey(0)
        self.rng_key, self.rng_key_ = random.split(rng_key)
        self.model = self.model_fn
        return f"Model {self.name} | {self.model_type} built"

    def run(self, tune=1000, draws=1000, chains=4, target_accept_prob=0.95, batch_size=None, iterations=None):
        if batch_size is None:
            nwarmup = tune
            nsamples = draws
            mcmc = utils.create_mcmc(self.model, nwarmup, nsamples, chains, target_accept_prob)
            mcmc.run(self.rng_key)
            utils.save_model_pickle(mcmc, self.outputs_path)
            trace = az.from_numpyro(mcmc)
            divergences_count = (trace.sample_stats["diverging"].values == True).sum()
            return trace, divergences_count, None
        else:
            samples = None
            divergences = None
            logll = None
            rng_key = random.PRNGKey(0)
            rng_key, rng_key_ = random.split(rng_key)
            kernel = NUTS(self.model, target_accept_prob=0.98, dense_mass=True, max_tree_depth=12, init_strategy=init_to_median(num_samples=100))
            mcmc = MCMC(kernel, num_warmup=batch_size, num_samples=batch_size, num_chains=4, chain_method="vectorized")
            mcmc.run(rng_key)
            samples = { key: jnp.array(value) for key, value in mcmc.get_samples(group_by_chain=True).items() }
            divergences = jnp.array(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
            logll = jnp.array(log_likelihood(self.model, mcmc.get_samples(), batch_ndims=1)["salary_hat"].reshape(4, batch_size, -1))

            samples = { key: jax.device_put(value, device=jax.devices("cpu")[0]) for key, value in samples.items() }
            divergences = jax.device_put(divergences, jax.devices("cpu")[0])
            logll = jax.device_put(logll, jax.devices("cpu")[0])
            trace = utils.create_inference_data(mcmc, samples, divergences, logll, self.dimensions_names, self.coords, self.target)
            max_rhat = az.summary(trace, round_to=5)["r_hat"].max()
            utils.save_model_pickle(mcmc, self.outputs_path)
            trace.to_netcdf(f"{self.outputs_path}/intermediate_trace.nc")
            print(f">>>>>>>>>>>>>>>> Warmup complete - max_rhat: {max_rhat} <<<<<<<<<<<<<<<<<<<<")

            for it in range(iterations):
                mcmc.post_warmup_state = mcmc.last_state
                mcmc.run(mcmc.post_warmup_state.rng_key)
                samples = utils.concat_samples([samples, mcmc.get_samples(group_by_chain=True)])
                divergences = jnp.concatenate([divergences, mcmc.get_extra_fields(group_by_chain=True)["diverging"]], axis=1)
                logll = jnp.concatenate([logll, log_likelihood(self.model, mcmc.get_samples(), batch_ndims=1)["salary_hat"].reshape(4,batch_size,-1)], axis=1)
                trace = utils.create_inference_data(mcmc, samples, divergences, logll, self.dimensions_names, self.coords, self.target)
                max_rhat = az.summary(trace, round_to=5)["r_hat"].max()
                utils.save_model_pickle(mcmc, self.outputs_path)
                trace.to_netcdf(f"{self.outputs_path}/intermediate_trace.nc")
                print(f">>>>>>>>>>>>>>>> Iteration {it+1}/{iterations} complete - max_rhat: {max_rhat} <<<<<<<<<<<<<<<<<<<<")
                if max_rhat <= 1.01:
                    print(f"Convergence reached iteration: {it}")
                    break
            trace.to_netcdf(f"{self.outputs_path}/trace.nc")
            os.remove(f"{self.outputs_path}/intermediate_trace.nc")
            trace = utils.create_inference_data(mcmc, samples, divergences, logll, self.dimensions_names, self.coords, self.target)
            divergences_count = (trace.sample_stats["diverging"].values == True).sum()
            return trace, divergences_count, it



