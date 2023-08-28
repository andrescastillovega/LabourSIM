# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import pymc.sampling.jax as pmjax
import jax
import datetime

from wage_utils import Capturing, sampling_output

# Set random seed
RANDOM_SEED = 230810
rng = np.random.default_rng(RANDOM_SEED)

############################################## Functions ###############################################################
def make_data(data_summary, model):
    with model:
        model_data = {}
        for variable, data in data_summary.items():
            if data["element"] in ["slope","dimension"]:
                model_data[f"{variable}"] = pm.Data(f"data_{variable}", data["data"], mutable=False)
    return model, model_data


def make_hyperpriors(variable, data, model):
    with model:
        if data["type"] == "parameter":
            mu = pm.Normal(f'mu_{variable}', mu=0, sigma=10, dims=data["dims"])
            sigma = pm.Exponential(f'sigma_{variable}', lam=10, dims=data["dims"])
    return model


def make_prior(variable, data, model, param="centered"):
    with model:
        if data["type"] == "parameter":
            # Get hyperpriors
            mu = [ var for var in model.free_RVs if f"mu_{variable}" in var.name ][0]
            sigma = [ var for var in model.free_RVs if f"sigma_{variable}" in var.name ][0]

            # Define if centered or non-centered parametization
            if param == "centered":
                pm.Normal(f"beta_{variable}", mu=mu, sigma=sigma, dims=data["dims"])
            elif param == "non-centered":
                offset = pm.Normal(f"offset_{variable}", mu=0, sigma=1, dims=data["dims"])
                pm.Deterministic(f"beta_{variable}", mu + sigma * offset)
    return model


def make_ev_level(variables, model, level, model_data=None):
    id_level = int(float(level.split("_")[1]))
    mu = 0
    with model:
        # Create expected value expression for the level
        for variable, data in variables:
            if data["type"] == "parameter":
                # Set parameter
                parameter = [ var for var in model.unobserved_RVs if f"beta_{variable}" in var.name ][0]

                # Define if intercept or slope
                if data["element"] == "intercept":
                    beta = 1
                elif data["element"] == "slope":
                    beta = model_data[f"{variable}"]
                    
                # Define define dimensions and parameter contribution to expected value
                if data["dims"] == None:
                    mu += parameter * beta
                else:
                    mu += parameter[model_data[f"{data['dims']}"]] * beta
        # Add expected value from previous level to the current level
        if id_level > 1:
            mu += [ var for var in model.unobserved_RVs if f"ev_level_{id_level-1}" in var.name ][0]
        # Apply exponential transformation (log-link)
        pm.Deterministic(f"ev_{level}", pm.math.exp(mu))
    return model


def make_levels(data_summary, model, model_data=None):
    with model:
        levels = { f"level_{v['level']}": [ (key, val) for key, val in data_summary.items() if val.get('level') == v['level']]
                      for v in data_summary.values() if v.get('level') is not None }
        for level, variables in levels.items():
            # Create hyperpriors and priors for the level
            for variable in variables:
                var_name, var_data = variable
                make_hyperpriors(var_name, var_data, model)
                make_prior(var_name, var_data, model, param="non-centered")
            # Create expected value expression for the level
            make_ev_level(variables, model, level, model_data)
    return model


def make_likelihood(id_run, model_name, data_summary, model):
    with model:
        target = [ data["data"] for _, data in data_summary.items() if data["type"]=="target" ][0]
        shape = pm.Exponential("shape", lam=10)
        mu = [ var for var in model.unobserved_RVs if "ev" in var.name ][-1]
        y = pm.Gamma("salary_hat", alpha=shape, beta=shape/mu,  observed=target)

        # Save Model graph
        model_graph = pm.model_to_graphviz(model)
        model_graph.render(f"outputs/{id_run}_{model_name}/{id_run}_graph_{model_name}", format="svg")
    return model


def validate_workflow(id_run, model_name, data_summary, coords):
    # Setting up the model
    model = pm.Model(coords=coords)
    model, model_data = make_data(data_summary, model)
    model = make_levels(data_summary, model, model_data)
    model = make_likelihood(id_run, model_name, data_summary, model)


def sample(id_run, model_name, model, nchains=4, ndraws=1000, ntune=1000, target_accept=0.95, postprocess_chunks=10):
    # Sampling
    with Capturing() as sampling_info: # This code captures the numpyro sampler stdout prints 
        with model:
            trace = pmjax.sample_numpyro_nuts(draws=ndraws, tune=ntune, target_accept=target_accept, chains=nchains, progressbar=True,
                                              idata_kwargs={"log_likelihood": True}, postprocessing_chunks=postprocess_chunks)
            trace.to_netcdf(f"outputs/{id_run}_{model_name}/{id_run}_trace_{model_name}.nc")
    # Save trace plot
    az.plot_trace(trace, combined=True, var_names=["~mu_","~sigma_","~ev_"], filter_vars="like")\
                    .ravel()[0].figure.savefig(f"outputs/{id_run}_{model_name}/{id_run}_traceplot_{model_name}.svg")
    # Save summary
    sampling_summary = pm.summary(trace, var_names=["~ev_"], filter_vars="like")
    sampling_summary.to_csv(f"outputs/{id_run}_{model_name}/{id_run}_summary_{model_name}.csv")
    # Save sampling metadata
    sampling_metadata = sampling_output(sampling_info, nchains=nchains, ndraws=ndraws, ntunes=ntune)
    sampling_metadata["maxRhat"] = sampling_summary["r_hat"].max()
    
    return sampling_metadata


def run(id_run, model_name, data_summary, coords, sampling_record, nchains, ndraws, ntune, target_accept):
    start_time = datetime.datetime.now()
    # Setting up the model
    model = pm.Model(coords=coords)
    model, model_data = make_data(data_summary, model)
    model = make_levels(data_summary, model, model_data)
    model = make_likelihood(id_run, model_name, data_summary, model)

    # Sampling
    sampling = sample(id_run, model_name, model, nchains=nchains, ndraws=ndraws, ntune=ntune, target_accept=target_accept)
    sampling["start_time"] = start_time.strftime("%Y-%m-%d %H:%M")
    sampling["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sampling["model_name"] = model_name

    # Save sampling metadata info
    sampling_record = pd.concat([sampling_record, pd.DataFrame.from_dict({ id_run: sampling }, orient="index")])
    sampling_record.to_csv("sampling_record.csv")

    return sampling_record, sampling

def create_data_summary(model_workflow, dataset, id_run):
    data_summary = {}
    for _, row in model_workflow.query(f"id_run == {id_run}").iterrows():
        # Set data and cats None when dims is not defined
        if (row["element"]=="intercept"):
            data = None
            cats = None
        else:
            data = pd.factorize(dataset[row["variable"]])[0] if row["type"] in ["parameter","dimension"] else dataset[row["variable"]].values
            cats = pd.factorize(dataset[row["variable"]])[1] if row["type"] in ["parameter","dimension"] else None

        # cats: If dims!=None, then cats is a list of the unique values of the variable
        data_summary[row["variable"]] = {
            "type": row["type"],
            "element": row["element"] if row["type"] in ["parameter","dimension"] else None,
            "data": data,
            "cats": cats,
            "dims": row["dims"] if not pd.isna(row["dims"]) else None,
            "level": row["level"] if row["type"] == "parameter" else None
        }
    return data_summary