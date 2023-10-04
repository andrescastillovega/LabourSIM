import jax.numpy as jnp
import numpyro
import utils
from numpyro import distributions as dist

def pooled(features,
           feature_names,
           target,
           prior_dist="normal",
           avg_salary_params={"loc": 10, "scale": 1}, 
           prior_params={"loc": 0, "scale": 1},
           shape_dist="uniform", 
           shape_params={"low": 1, "high": 100},
           target_dist="gamma"):
    
    # Priors and expected value
    avg_salary = numpyro.sample("avg_salary", utils.DISTRIBUTIONS[prior_dist](**avg_salary_params))
    mu = avg_salary
    for feature, name in zip(features.T, feature_names):
        prior = numpyro.sample(f"beta_{feature}", utils.DISTRIBUTIONS[prior_dist](**prior_params))
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
                 mu_dist="normal",
                 mu_avg_salary_params={"loc": 0, "scale": 1},
                 mu_params={"loc": 0, "scale": 1},
                 sigma_dist="half_normal",
                 sigma_params={"scale": 1},
                 shape_dist="uniform",
                 shape_params={"low": 1, "high": 100},
                 target_dist="gamma"):
    
    # Hyperpriors
    mu_avg_salary = numpyro.sample("mu_avg_salary", utils.DISTRIBUTIONS[mu_dist](**mu_avg_salary_params))
    sigma_avg_salary = numpyro.sample("sigma_avg_salary", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_hyperpriors = numpyro.sample("mu_hyperpriors", utils.DISTRIBUTIONS[mu_dist](**mu_params), sample_shape=(len(features_names),))
    sigma_hyperpriors = numpyro.sample("sigma_hyperpriors", utils.DISTRIBUTIONS[sigma_dist](**sigma_params), sample_shape=(len(features_names),))

    offset_avg_salary = numpyro.sample("offset_avg_salary", 
                                       utils.DISTRIBUTIONS["normal"](loc=0, scale=1),
                                       sample_shape=(dimension.max() + 1,))
    avg_salary = numpyro.deterministic("avg_salary", mu_avg_salary + offset_avg_salary * sigma_avg_salary)
    offset_features = numpyro.sample("offset_features",
                                     utils.DISTRIBUTIONS["normal"](loc=0, scale=1),
                                     sample_shape=(len(features_names), dimension.max() + 1))
    prior_features = numpyro.deterministic("features", mu_hyperpriors + offset_features * sigma_hyperpriors)

    mu = avg_salary[dimension]
    print(features.shape, prior_features.shape)
    mu_sum = jnp.dot(prior_features[:, dimension], features)

    shape = numpyro.sample("shape", utils.DISTRIBUTIONS[shape_dist](**shape_params))
    mu = jnp.exp(mu + mu_sum)
    rate = shape / mu

    # Likelihood
    with numpyro.plate("data", target.shape[0]):
        numpyro.sample("salary_hat", utils.DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=target)


def no_pooled(features,
              features_names,
              dimension,
              dimension_name,
              target,
              prior_dist="normal",
              avg_salary_params={"loc": 10, "scale": 1},
              prior_params={"loc": 0, "scale": 1},
              shape_dist="uniform", 
              shape_params={"low": 1, "high": 100},
              target_dist="gamma"):
    # Priors
    with numpyro.plate(f"{dimension_name}", dimension.max() + 1):
        avg_salary = numpyro.sample("offset_avg_salary", utils.DISTRIBUTIONS[prior_dist](**avg_salary_params))
        priors = []
        for i, feature in enumerate(features_names):
            priors.append(numpyro.sample(f"offset_{feature}", utils.DISTRIBUTIONS[prior_dist](**prior_params)))
    shape = numpyro.sample("shape", utils.DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = avg_salary[dimension]
    for i, prior in enumerate(priors):
        mu += prior[dimension] * features[:,i]
    mu = jnp.exp(mu)
    rate = shape / mu

    # Likelihood
    # with numpyro.plate("data", target.shape[0]):
    numpyro.sample("salary_hat", utils.DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=target)


def hierarchical_test(features,
                 features_names,
                 dimension,
                 dimension_name,
                 target,
                 mu_dist="normal",
                 mu_avg_salary_params={"loc": 10, "scale": 1},
                 mu_params={"loc": 0, "scale": 1},
                 sigma_dist="half_normal",
                 sigma_params={"scale": 1},
                 shape_dist="uniform",
                 shape_params={"low": 1, "high": 100},
                 target_dist="gamma"):
    
    # Hyperpriors
    mu_avg_salary = numpyro.sample("mu_avg_salary", utils.DISTRIBUTIONS[mu_dist](**mu_avg_salary_params))
    sigma_avg_salary = numpyro.sample("sigma_avg_salary", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_exp = numpyro.sample("mu_exp", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_exp = numpyro.sample("sigma_exp", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_sex = numpyro.sample("mu_sex", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_sex = numpyro.sample("sigma_sex", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_elementary_edu = numpyro.sample("mu_elementary_edu", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_elementary_edu = numpyro.sample("sigma_elementary_edu", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_highschool_edu = numpyro.sample("mu_highschool_edu", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_highschool_edu = numpyro.sample("sigma_highschool_edu", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_postsec_edu = numpyro.sample("mu_postsec_edu", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_postsec_edu = numpyro.sample("sigma_postsec_edu", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_undergrad_edu = numpyro.sample("mu_undergrad_edu", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_undergrad_edu = numpyro.sample("sigma_undergrad_edu", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_graduate_edu = numpyro.sample("mu_graduate_edu", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_graduate_edu = numpyro.sample("sigma_graduate_edu", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_age = numpyro.sample("mu_age", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_age = numpyro.sample("sigma_age", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_tenure = numpyro.sample("mu_tenure", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_tenure = numpyro.sample("sigma_tenure", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_union = numpyro.sample("mu_union", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_union = numpyro.sample("sigma_union", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_part_time = numpyro.sample("mu_part_time", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_part_time = numpyro.sample("sigma_part_time", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_public_sector = numpyro.sample("mu_public_sector", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_public_sector = numpyro.sample("sigma_public_sector", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_self_emp = numpyro.sample("mu_self_emp", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_self_emp = numpyro.sample("sigma_self_emp", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_grad_highschool_refyear = numpyro.sample("mu_grad_highschool_refyear", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_grad_highschool_refyear = numpyro.sample("sigma_grad_highschool_refyear", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_grad_college_refyear = numpyro.sample("mu_grad_college_refyear", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_grad_college_refyear = numpyro.sample("sigma_grad_college_refyear", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))
    mu_grad_univ_refyear = numpyro.sample("mu_grad_univ_refyear", utils.DISTRIBUTIONS[mu_dist](**mu_params))
    sigma_grad_univ_refyear = numpyro.sample("sigma_grad_univ_refyear", utils.DISTRIBUTIONS[sigma_dist](**sigma_params))

    # Priors
    with numpyro.plate(f"{dimension_name}", dimension.max() + 1):
        offset_avg_salary = numpyro.sample("offset_avg_salary", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        avg_salary = numpyro.deterministic("avg_salary", mu_avg_salary + offset_avg_salary * sigma_avg_salary)
        offset_exp = numpyro.sample("offset_exp", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        exp = numpyro.deterministic("exp", mu_exp + offset_exp * sigma_exp)
        offset_sex = numpyro.sample("offset_sex", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        sex = numpyro.deterministic("sex", mu_sex + offset_sex * sigma_sex)
        offset_elementary_edu = numpyro.sample("offset_elementary_edu", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        elementary_edu = numpyro.deterministic("elementary_edu", mu_elementary_edu + offset_elementary_edu * sigma_elementary_edu)
        offset_highschool_edu = numpyro.sample("offset_highschool_edu", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        highschool_edu = numpyro.deterministic("highschool_edu", mu_highschool_edu + offset_highschool_edu * sigma_highschool_edu)
        offset_postsec_edu = numpyro.sample("offset_postsec_edu", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        postsec_edu = numpyro.deterministic("postsec_edu", mu_postsec_edu + offset_postsec_edu * sigma_postsec_edu)
        offset_undergrad_edu = numpyro.sample("offset_undergrad_edu", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        undergrad_edu = numpyro.deterministic("undergrad_edu", mu_undergrad_edu + offset_undergrad_edu * sigma_undergrad_edu)
        offset_graduate_edu = numpyro.sample("offset_graduate_edu", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        graduate_edu = numpyro.deterministic("graduate_edu", mu_graduate_edu + offset_graduate_edu * sigma_graduate_edu)
        offset_age = numpyro.sample("offset_age", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        age = numpyro.deterministic("age", mu_age + offset_age * sigma_age)
        offset_tenure = numpyro.sample("offset_tenure", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        tenure = numpyro.deterministic("tenure", mu_tenure + offset_tenure * sigma_tenure)
        offset_union = numpyro.sample("offset_union", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        union = numpyro.deterministic("union", mu_union + offset_union * sigma_union)
        offset_part_time = numpyro.sample("offset_part_time", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        part_time = numpyro.deterministic("part_time", mu_part_time + offset_part_time * sigma_part_time)
        offset_public_sector = numpyro.sample("offset_public_sector", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        public_sector = numpyro.deterministic("public_sector", mu_public_sector + offset_public_sector * sigma_public_sector)
        offset_self_emp = numpyro.sample("offset_self_emp", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        self_emp = numpyro.deterministic("self_emp", mu_self_emp + offset_self_emp * sigma_self_emp)
        offset_grad_highschool_refyear = numpyro.sample("offset_grad_highschool_refyear", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        grad_highschool_refyear = numpyro.deterministic("grad_highschool_refyear", mu_grad_highschool_refyear + offset_grad_highschool_refyear * sigma_grad_highschool_refyear)
        offset_grad_college_refyear = numpyro.sample("offset_grad_college_refyear", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        grad_college_refyear = numpyro.deterministic("grad_college_refyear", mu_grad_college_refyear + offset_grad_college_refyear * sigma_grad_college_refyear)
        offset_grad_univ_refyear = numpyro.sample("offset_grad_univ_refyear", utils.DISTRIBUTIONS["normal"](loc=0, scale=1))
        grad_univ_refyear = numpyro.deterministic("grad_univ_refyear", mu_grad_univ_refyear + offset_grad_univ_refyear * sigma_grad_univ_refyear)      

    shape = numpyro.sample("shape", utils.DISTRIBUTIONS[shape_dist](**shape_params))

    # Expected value
    mu = jnp.exp(avg_salary[dimension] + exp[dimension] * features[:,0] + sex[dimension] * features[:,1] +
                    elementary_edu[dimension] * features[:,2] + highschool_edu[dimension] * features[:,3] +
                    postsec_edu[dimension] * features[:,4] + undergrad_edu[dimension] * features[:,5] +
                    graduate_edu[dimension] * features[:,6] + age[dimension] * features[:,7] +
                    tenure[dimension] * features[:,8] + union[dimension] * features[:,9] +
                    part_time[dimension] * features[:,10] + public_sector[dimension] * features[:,11] +
                    self_emp[dimension] * features[:,12] + grad_highschool_refyear[dimension] * features[:,13] +
                    grad_college_refyear[dimension] * features[:,14] + grad_univ_refyear[dimension] * features[:,15])
    rate = shape / mu

    # Likelihood

    numpyro.sample("salary_hat", utils.DISTRIBUTIONS[target_dist](concentration=shape, rate=rate), obs=target)