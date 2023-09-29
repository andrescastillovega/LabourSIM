import arviz as az
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
import pandas as pd
import pickle

import utils

class GammaGML():
    def __init__(self, name, parameters, target_var, dataset, year=None, standardize_vars=[]):
        self.name = name
        self.target = target_var
        self.priors = {
            "normal": dist.Normal,
            "uniform": dist.Uniform,
            "beta": dist.Beta,
            "gamma": dist.Gamma,
            "poisson": dist.Poisson,
            "student_t": dist.StudentT,
            "halfnormal": dist.HalfNormal,
            "halfcauchy": dist.HalfCauchy}
        self.vars = {}
        self.dimensions = []
        self.year = year

        if self.year is None:
            self.outputs_path = f"../outputs/{self.name}"
        else:
            self.outputs_path = f"../outputs/{self.name}/{self.year}"

        self.get_var_dims(parameters)
        self.data_processing(dataset, standardize_vars, year)
        self.get_plates()
        self.get_coords_dims()

    def __repr__(self):
        if self.year is not None:
            return f"Model: {self.name} - params: {list(self.vars.keys())} - year: {self.year}"
        else:
            return f"Model: {self.name} - params: {list(self.vars.keys())}"
    
    def standardize_var(self, variable):
        variable = (variable - variable.mean()) / variable.std()
        return variable

    def get_var_dims(self, parameters):
        # Create vars and dimensions attributes
        for var, params in parameters.items():
            self.vars[var] = params

            if params.get("dims") is not None:
                self.dimensions += [ dim for dim in params.get("dims") if dim not in self.dimensions ]

    def get_coords_dims(self):
        # Get coords and dims attributes for Arviz
        coords = { coord[0]: list(coord[1]) for coord in self.dimensions }
        dims = {}
        for var in self.vars.keys():
            if self.vars[var]["type"] == "intercept":
                var_name = f"{var}"
            elif self.vars[var]["type"] == "slope":
                var_name = f"beta_{var}"

            if self.vars[var]["dims"] is not None:
                dims[var_name] = [f"{dim}" for dim in self.vars[var]["dims"]]
        
        self.coords = coords
        self.dims = dims

    def data_processing(self, dataset, standardize_vars, year):
        # Data processing
        self.dataset = dataset.to_records(index=False)
        if year is not None:
            if 'year' in self.dataset.dtype.names:
                self.dataset = self.dataset[self.dataset["year"] == year]
                self.dataset = drop_fields(self.dataset, "year")
            else:
                raise ValueError("year is not a column in the dataset")
            
        for var in standardize_vars:
            self.dataset[var] = self.standardize_var(self.dataset[var])

        for var_id, var  in enumerate(self.dimensions):
            coord, coord_idx = np.unique(self.dataset[var], return_inverse=True)
            self.dimensions[var_id] = (var, coord)
            self.dataset = drop_fields(self.dataset, var)
            self.dataset = append_fields(self.dataset, var, coord_idx, usemask=False)

    def get_prior_name(self, var):
        # Prior name
        if self.vars[var]["type"] == "intercept":
            prior_name = f"{var}"
        elif self.vars[var]["type"] == "slope":
            prior_name = f"beta_{var}"
        else:
            prior_name = f"{var}"
        return prior_name
    
    def get_dist_params(self, var, prior_name):
        if self.vars[var]["hyperprior"]:
            dist_params = self.build_hyperprior(var, prior_name)
        else:
            dist_params = self.vars[var]["initial_values"]
        return dist_params

    def get_plates(self):
        # Create plates
        self.plates = {}
        self.plates[None] = {"vars": [ var for var, params in self.vars.items() if params["dims"] is None ]}
        for dim, dim_categories in self.dimensions:
            self.plates[dim] = {"vars": [ var for var, params in self.vars.items() if dim in list(params["dims"] if params["dims"] is not None else []) ]}
            self.plates[dim]["dimensions"] = dim_categories
    
    def build_hyperprior(self, var, prior_name):
        hyperpriors = {}
        mu_class = self.priors["normal"]
        sigma_class = self.priors["halfnormal"]
        mu_prior = mu_class(**self.vars[var]["initial_values"])
        sigma_prior = sigma_class(scale=1)

        mu = numpyro.sample(name = f"mu_{prior_name}", fn = mu_prior)
        sigma = numpyro.sample(name = f"sigma_{prior_name}", fn = sigma_prior)
        hyperpriors["loc"] = mu
        hyperpriors["scale"] = sigma

        return hyperpriors

    def build_prior(self, var, prior_name, dist_params):
        distribution_class = self.priors[self.vars[var]["dist"]]
        distribution = distribution_class(**dist_params)

        if self.vars[var]["parameterization"] == "non-centered":
            prior = numpyro.sample(name = prior_name,
                                fn = dist.TransformedDistribution(dist.Normal(loc=0, scale=1),
                                                                    dist.transforms.AffineTransform(**dist_params)))
        else:
            prior = numpyro.sample(name = prior_name, fn = distribution)
        return prior
    
    def model(self):
            # Add prior names
            for plate, plate_config in self.plates.items():
                self.plates[plate]["prior_names"] = []
                for var in plate_config["vars"]:
                    prior_name = self.get_prior_name(var)
                    self.plates[plate]["prior_names"].append(prior_name)

            # Add distribution parameters (or hyperpriors)
            for plate, plate_config in self.plates.items():
                self.plates[plate]["dist_params"] = []
                for var in plate_config["vars"]:
                    dist_params = self.get_dist_params(var, plate_config["prior_names"][plate_config["vars"].index(var)])
                    self.plates[plate]["dist_params"].append(dist_params)
                    
            # Priors
            mu = 0
            for plate, plate_config in self.plates.items():
                if plate is None:
                    for var in plate_config["vars"]:
                        prior_name = plate_config["prior_names"][plate_config["vars"].index(var)]
                        dist_params = plate_config["dist_params"][plate_config["vars"].index(var)]
                        # Priors  
                        if var == "shape":
                            shape = self.build_prior(var, prior_name, dist_params)
                        elif self.vars[var]["type"] == "intercept":
                            prior = self.build_prior(var, prior_name, dist_params)
                            mu += prior
                        else:
                            prior = self.build_prior(var, prior_name, dist_params)
                            mu += prior * self.dataset[self.vars[var]["var"]]
                        
                else:
                     # Hyperpriors should be outside the plate
                    with numpyro.plate(plate, len(plate_config["dimensions"])):
                        for var in plate_config["vars"]:
                            prior_name = plate_config["prior_names"][plate_config["vars"].index(var)]
                            dist_params = plate_config["dist_params"][plate_config["vars"].index(var)]
                            if self.vars[var]["type"] == "intercept":
                                prior = self.build_prior(var, prior_name, dist_params)
                                mu += prior[self.dataset[plate]]
                            else:
                                prior = self.build_prior(var, prior_name, dist_params)
                                mu += prior[self.dataset[plate]] * self.dataset[self.vars[var]["var"]]
                                
            mu = jnp.exp(mu)
            rate = shape / mu    

            with numpyro.plate("data", len(self.dataset)):
                likelihood = numpyro.sample(self.target, dist.Gamma(concentration=shape, rate=rate), obs=self.dataset[self.target])

    def build(self):
        model = self.model
        return model
        
    def add_unconstrained_vars(self, trace):
        for var in trace.posterior.data_vars:
            if var not in ['shape']:
                new_var_name = f"unconstrained_{var}"
                trace.posterior = trace.posterior.assign({new_var_name: np.exp(trace.posterior[var])})
        return trace

    def render_model(self):
        model = self.build()
        graph = numpyro.render_model(model, render_distributions=True)
        graph.render(filename=f"{self.outputs_path}/model", format='svg')

    def run(self, model, draws=4000, warmup=4000, chains=4, target_accept_prob=0.95, batch_size=None, progress_bar=True):
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        if batch_size is None:
            kernel = NUTS(model, target_accept_prob=target_accept_prob)
            mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws,
                        num_chains=chains, chain_method='vectorized', progress_bar=progress_bar)
            mcmc.run(rng_key)
            with open(fr"{self.outputs_path}/model.pickle", "wb") as output_file:
                pickle.dump(mcmc, output_file)
            trace = az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)
            divergences = mcmc.get_extra_fields()['diverging'].sum()
        else:
            iterations = int(warmup / batch_size)
            divergences = 0

            kernel = NUTS(model, target_accept_prob=target_accept_prob)
            mcmc = MCMC(kernel, num_warmup=warmup, num_samples=batch_size, 
                        num_chains=chains, chain_method='vectorized', progress_bar=progress_bar)

            for i in range(iterations):
                if i == 0:
                    mcmc.run(rng_key)
                    with open(fr"{self.outputs_path}/model.pickle", "wb") as output_file:
                        pickle.dump(mcmc, output_file)
                    trace = az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)
                    divergences = mcmc.get_extra_fields()['diverging'].sum()
                    print(f"It {i} - rhat: {az.summary(trace)['r_hat'].max():.3f}")
                else:
                    mcmc.post_warmup_state = mcmc.last_state
                    mcmc.run(mcmc.post_warmup_state.rng_key)
                    with open(fr"{self.outputs_path}/model.pickle", "wb") as output_file:
                        pickle.dump(mcmc, output_file)
                    trace = az.concat([trace, az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)], dim="draw")
                    divergences += mcmc.get_extra_fields()['diverging'].sum()
                    print(f"It {i} - rhat: {az.summary(trace)['r_hat'].max():.3f}")

            trace = self.add_unconstrained_vars(trace)

        return trace, divergences