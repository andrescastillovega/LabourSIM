import arviz as az
import utils
import pandas as pd
import pickle
from base_models import pooled, hierarchical, no_pooled, hierarchical_test
from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

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
        elif self.model_type == "hierarchical_test":
            return hierarchical_test(self.features, self.features_names, self.idx_dims, self.dimensions_names, self.target)
        else:
            raise ValueError("Invalid model type")

    def build(self):
        rng_key = random.PRNGKey(0)
        self.rng_key, self.rng_key_ = random.split(rng_key)
        self.model = self.model_fn
        return f"Model {self.name} | {self.model_type} built"


    def run(self, warmup=2000, draws=2000, chains=4, target_accept_prob=0.95, batch_size=None):
        if batch_size is None:
            kernel = NUTS(self.model, target_accept_prob=target_accept_prob, dense_mass=True)
            mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws,
                        num_chains=chains, chain_method='vectorized')
            mcmc.run(self.rng_key)
            with open(fr"{self.outputs_path}/model.pickle", "wb") as output_file:
                pickle.dump(mcmc, output_file)
            trace = az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)
            divergences = mcmc.get_extra_fields()['diverging'].sum()
        else:
            iterations = int(warmup / batch_size)
            divergences = 0

            kernel = NUTS(self.model, target_accept_prob=target_accept_prob, dense_mass=True)
            mcmc = MCMC(kernel, num_warmup=warmup, num_samples=batch_size, 
                        num_chains=chains, chain_method='vectorized')

            for i in range(iterations):
                if i == 0:
                    mcmc.run(self.rng_key)
                    with open(fr"{self.outputs_path}/model.pickle", "wb") as output_file:
                        pickle.dump(mcmc, output_file)
                    trace = az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)
                    divergences = mcmc.get_extra_fields()['diverging'].sum()
                    print(f"It{i}-rhat: {az.summary(trace)['r_hat'].max():.3f}")
                    if az.summary(trace)['r_hat'].max() < 1.01:
                        print(f"Rhat is less than 1.01 - Run stopped at iteration {i}")
                        break
                else:
                    mcmc.post_warmup_state = mcmc.last_state
                    mcmc.run(mcmc.post_warmup_state.rng_key)
                    with open(fr"{self.outputs_path}/model.pickle", "wb") as output_file:
                        pickle.dump(mcmc, output_file)
                    trace = az.concat([trace, az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)], dim="draw")
                    divergences += mcmc.get_extra_fields()['diverging'].sum()
                    print(f"It{i}-rhat: {az.summary(trace)['r_hat'].max():.3f}")
                    if az.summary(trace)['r_hat'].max() < 1.01:
                        print(f"Rhat is less than 1.01 - Run stopped at iteration {i}")
                        break
        return trace, divergences



