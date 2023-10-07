import arviz as az
import utils
import pandas as pd
import pickle
from base_models import pooled, hierarchical, no_pooled
from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_likelihood

class GammaGLM():
    def __init__(self, name, model_type, dataset, target, parameters, dimensions, standardize_vars, OUTPUTS_PATH, year=None):
        self.name = name
        self.model_type = model_type
        self.features_names = parameters
        self.dimensions_names = dimensions
        self.year = year

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
        if self.model_type == "pooled":
            return pooled(self.features, self.features_names, self.target)
        elif self.model_type == "hierarchical":
            return hierarchical(self.features, self.features_names, self.idx_dims, self.dimensions_names, self.target)
        elif self.model_type == "no_pooled":
            return no_pooled(self.features, self.features_names, self.idx_dims, self.dimensions_names, self.target)
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
            return trace, divergences_count
        else:
            # Initial run settings
            iterations = iterations
            prev_rhat = 10e10
            nwarmup = batch_size
            nsamples = batch_size
            chains = chains
            last_sample = None
            mcmc = utils.create_mcmc(self.model, nwarmup, nsamples, chains, target_accept_prob)
            samples = None
            divergences = None
            loglikelihood = None
            nsample_updated = False

            # Run NUTS
            rng_key = random.PRNGKey(0)
            rng_key, rng_key_ = random.split(rng_key)
            for it in range(iterations):
                if it == 0:
                    mcmc.run(rng_key)
                    utils.save_model_pickle(mcmc, self.outputs_path)
                elif nsample_updated:
                    mcmc.run(rng_key, init_params=last_sample)
                    utils.save_model_pickle(mcmc, self.outputs_path)
                else:
                    mcmc.post_warmup_state = mcmc.last_state
                    mcmc.run(mcmc.post_warmup_state.rng_key)
                    utils.save_model_pickle(mcmc, self.outputs_path)
                    
                samples, divergences, loglikelihood, sample_rhat, cumulative_rhat = utils.get_batch_results(self.model,
                                                                                                            mcmc,
                                                                                                            samples,
                                                                                                            divergences,
                                                                                                            loglikelihood,
                                                                                                            it,
                                                                                                            chains,
                                                                                                            nsamples,
                                                                                                            self.target.shape[0])
                
                trace = utils.create_inference_data(mcmc, samples, divergences, loglikelihood, "industry", self.coords["industry"], self.target)
                trace.to_netcdf(f"{self.outputs_path}/intermediate_trace.nc")

                print(f"It:{it} - sample rhat: {sample_rhat} - cumulative rhat: {cumulative_rhat}")
                if cumulative_rhat < 1.01:
                    print(f"Model converged at iteration: {it} with {loglikelihood.shape[1]} samples")
                    break
                if prev_rhat - cumulative_rhat < 0.05:
                    last_sample = {k: v[:,-1] for k, v in mcmc.get_samples(group_by_chain=True).items()}
                    nwarmup += 100
                    nsamples += 100
                    mcmc = utils.create_mcmc(self.model, nwarmup, nsamples, chains)
                    nsample_updated = True
                    print(f"Increasing warmup to {nwarmup} and samples to {nsamples}")
                prev_rhat = cumulative_rhat

            divergences_count = (divergences == True).sum()

            trace = utils.create_inference_data(mcmc, samples, divergences, loglikelihood, "industry", self.coords["industry"], self.target)
            return trace, divergences_count



