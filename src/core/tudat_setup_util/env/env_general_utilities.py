
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np


###########################################################################
# ENVIRONMENT GENERAL UTILITIES ###########################################
###########################################################################

def normalize_spherical_harmonic_coefficients(unnormalized_coefficients):
    """
    Basic helper-function to normalize a given set of spherical harmonic gravity
    field parameters.

    Parameters
    ----------
    unnormalized_coefficients: [float]

    Returns
    -------
    normalized_coefficients: [float]
    """
    # Extract degree and order from shape of coefficient-array
    degree, order = np.shape(unnormalized_coefficients)
    # Initialise return-container
    normalized_coefficients = np.zeros_like(unnormalized_coefficients)
    # Loop over associated degrees and orders to calculate normalization factors
    for l in range(degree):
        for m in range(order):
            if m > l:
                break
            if m == 0:
                normalization_factor = np.sqrt(((2 - 1) * (2 * l + 1) * np.math.factorial(l - m)) /
                                               (np.math.factorial(l + m)))
            else:
                normalization_factor = np.sqrt(((2 - 0) * (2 * l + 1) * np.math.factorial(l - m)) /
                                               (np.math.factorial(l + m)))

            # Normalize coefficients
            normalized_coefficients[l, m] = unnormalized_coefficients[l, m] / normalization_factor

    return normalized_coefficients
