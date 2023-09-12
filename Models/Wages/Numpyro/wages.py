import argparse
import datetime as dt
import jax as jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import psutil


if __name__ == "__main__":
    # Set run parameters
    parser = argparse.ArgumentParser(description="LabourSIM wage model")
    parser.add_argument("--dataset", help="Specify the .csv file with the labour market data")
    parser.add_argument("--year", help="Specify the year of analysis", type=int, default=None)
    parser.add_argument("--nchains", help="Specify the number of chains for the sampling", type=int, default=4)
    parser.add_argument("--ndraws", help="Specify the number of draws for the sampling", type=int, default=4000)
    parser.add_argument("--ntune", help="Specify the number of tuning steps for the sampling", type=int, default=4000)
    parser.add_argument("--target_accept", help="Specify the target acceptance rate for the sampling", type=float, default=0.95)
    parser.add_argument("--ncores", help="Specify the number of CPU cores to use", type=int, default=4)
    args = parser.parse_args()

    year = args.year
    nchains = args.nchains
    ndraws = args.ndraws
    ntune = args.ntune
    target_accept = args.target_accept
    ncores = args.ncores

    # Set number of cores
    available_cores = psutil.cpu_count(logical=True)
    if ncores > available_cores:
        raise NameError(f"Number of cores specified ({ncores}) is greater than the number of available cores ({available_cores}).\n\
                        >>> Please use --ncores to specify a number less than or equal to {available_cores}.")
    else:
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(ncores)
    

    
