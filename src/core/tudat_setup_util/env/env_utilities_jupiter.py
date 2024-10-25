
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
# tudatpy imports
from tudatpy.kernel.numerical_simulation import environment_setup
# Problem-specific imports
from src.core.tudat_setup_util.env.env_general_utilities import normalize_spherical_harmonic_coefficients


###########################################################################
# JUPITER ENVIRONMENT SETTING UTILITIES ###################################
###########################################################################

def get_jupiter_gravity_field_settings(jupiter_body_fixed_frame='IAU_Jupiter'):
    """
    Sets the spherical harmonic gravity field coefficients of Jupiter according to the values provided by
    Iess et al. (2018). Gravitational parameter GM taken from Folkner et al. (2017).

    Parameters
    ----------
    jupiter_body_fixed_frame : string
        Body fixed frame of Jupiter, default: IAU_Jupiter

    Returns
    -------
    jupiter_spherical_harmonic_gravity_field : tudatpy.kernel.numerical_simulation.environment_setup.
                                               gravity_field.SphericalHarmonicsGravityFieldSettings
        The settings for Jupiter's spherical harmonic gravity field (coefficients).
    """
    # Gravitational parameter taken from Folkner et al. (2017)
    jupiter_gravitational_parameter = 1.26686533E17
    # Reference radius taken from Iess et al. (2018)
    jupiter_radius = 71492E3

    # Unnormalized coefficients taken from Iess et al. (2018)
    unnormalized_cosine_coefficients = np.zeros(shape=(13, 13))
    unnormalized_cosine_coefficients[0, 0] = 1.0
    unnormalized_cosine_coefficients[2, 0] = -14696.572E-6
    unnormalized_cosine_coefficients[2, 1] = -0.013E-6
    unnormalized_cosine_coefficients[3, 0] = 0.042E-6
    unnormalized_cosine_coefficients[4, 0] = 586.609E-6
    unnormalized_cosine_coefficients[5, 0] = 0.069E-6
    unnormalized_cosine_coefficients[6, 0] = -34.198E-6
    unnormalized_cosine_coefficients[7, 0] = -0.124E-6
    unnormalized_cosine_coefficients[8, 0] = 2.426E-6
    unnormalized_cosine_coefficients[9, 0] = 0.106E-6
    unnormalized_cosine_coefficients[10, 0] = -0.172E-6
    unnormalized_cosine_coefficients[11, 0] = -0.033E-6
    unnormalized_cosine_coefficients[12, 0] = -0.047E-6

    unnormalized_sine_coefficients = np.zeros(shape=(13, 13))
    unnormalized_sine_coefficients[2, 1] = -0.003E-6

    # Normalize the unnormalized coefficients
    normalized_cosine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_cosine_coefficients)
    normalized_sine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_sine_coefficients)

    jupiter_spherical_harmonic_gravity_field = environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter=jupiter_gravitational_parameter,
        reference_radius=jupiter_radius,
        normalized_cosine_coefficients=normalized_cosine_coefficients,
        normalized_sine_coefficients=normalized_sine_coefficients,
        associated_reference_frame=jupiter_body_fixed_frame)

    return jupiter_spherical_harmonic_gravity_field
