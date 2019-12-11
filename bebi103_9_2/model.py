from functools import partial

import numpy as np
import scipy
import scipy.stats as st

rg = np.random.default_rng()

def linear_growth_model(a_0, k, t):
    """Compute bacterial area using linear model.

    :param a_0: initial area
    :type a_0: float
    :param k: growth rate
    :type k: float
    :param t: time since last division, in minutes
    :type t: float

    :return: estimated bacterial area based on provided parameters
    :rtype: float
    """
    return a_0 * (1 + k * t)

def exponential_growth_model(a_0, k, t):
    """Compute bacterial area using exponential model.

    :param a_0: initial area
    :type a_0: float
    :param k: growth rate
    :type k: float
    :param t: time since last division, in minutes
    :type t: float

    :return: estimated bacterial area based on provided parameters
    :rtype: float
    """
    return a_0 * np.exp(k * t)

def residual(params, times, areas, model):
    """Residual for the given bacterial growth model.

    :param params: parameters of the model
    :type params: tuple
    :param times: list of times since division
    :type times: list
    :param areas: list of bacterial areas
    :type areas: list
    :param model: model to pass in params and get a theoretical area
    :type model: callable
    """
    return areas - model(*params, times)

def growth_area_mle_lstq(
    data,
    model,
    initial_params=np.array([1, 0]),
    bounds=([0, 0], [np.inf, np.inf])
):
    """Compute MLE for parameters of the given bacterial growth model.

    :param data: list of (time, area) tuples
    :type data: list
    :param model: model that returns a theoretical area
    :type model: callable
    :param initial_params: initial parameters for mle calculation, defaults to
                           [1, 0]
    :type initial_params: numpy.array, optional
    :param bounds: parameter bounds, defaults to ([0, 0], [np.inf, np.inf])
    :type bounds: tuple of lists, optional

    :return: parameter estimates, with an additional estimate for sigma
             (standard deviation)
    :rtype: tuple
    """
    times = data[:, 0]
    areas = data[:, 1]

    r = partial(residual, model=model)
    res = scipy.optimize.least_squares(
        r,
        initial_params,
        args=(times, areas),
        bounds=bounds
    )

    # Compute residual sum of squares from optimal params
    rss_mle = np.sum(r(res.x, times, areas) ** 2)

    # Compute MLE for sigma
    sigma_mle = np.sqrt(rss_mle / len(times))

    return tuple([x for x in res.x] + [sigma_mle])

def generate_growth_data(params, model, times, size=1):
    """Generate a new growth area data set.

    :param params: parameters of the model
    :type params: tuple
    :param model: model to pass in params and get a theoretical area
    :type model: callable
    :param times: list of times since division
    :type times: list
    :param size: number of points to generate
    :type size: int
    """
    # The last element of params is the standard deviation
    samples = np.empty((size, len(times)))

    for i in range(size):
        mu = model(*params[:-1], times)
        sigma = params[-1]
        samples[i] = rg.normal(mu, sigma)

    return samples

def log_likelihood(params, model, times, areas):
    """Log likelihood of the given bacterial growth model.

    :param params: parameters of the model
    :type params: tuple
    :param model: model to pass in params and get a theoretical area
    :type model: callable
    :param times: list of times since division in minutes
    :type times: list
    :param areas: list of bacterial areas
    :type areas: list

    :return: log-likelihood
    :rtype: float
    """
    a_0, k, sigma = params

    mu = model(a_0, k, times)
    return np.sum(st.norm.logpdf(areas, mu, sigma))

def compute_AIC(params, model, times, areas):
    """Compute the Akaike information criterion, or AIC, of the given model using
    the provided parameters.

    :param params: parameters of the model
    :type params: tuple
    :param model: model to pass in params and get a theoretical area
    :type model: callable
    :param times: list of times since division in minutes
    :type times: list
    :param areas: list of bacterial areas
    :type areas: list

    :return: AIC
    :rtype: float
    """
    return -2 * (log_likelihood(params, model, times, areas) - len(params))
