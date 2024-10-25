
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
# tudatpy imports
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
# Problem-specific imports
from src.core.tudat_setup_util.env.env_general_utilities import normalize_spherical_harmonic_coefficients


###########################################################################
# GALILEAN MOONS ENVIRONMENT SETTING UTILITIES ############################
###########################################################################

### GRAVITY FIELDS ###

def get_io_gravity_field_settings(io_body_fixed_frame='IAU_Io'):
    """
    Sets the spherical harmonic gravity field coefficients of Io according to the values provided by
    Schubert et al. (2004).

    Parameters
    ----------
    io_body_fixed_frame : string (default=IAU_Io)
        Body fixed frame of Io

    Returns
    -------
    io_spherical_harmonic_gravity_field : tudatpy.kernel.numerical_simulation.environment_setup.
                                          gravity_field.SphericalHarmonicsGravityFieldSettings
        The settings for Io's spherical harmonic gravity field (coefficients).
    """
    # Gravitational parameter taken from Schubert et al. (2004)
    io_gravitational_parameter = 5959.91E9
    io_reference_radius = spice_interface.get_average_radius('Io')

    # Unnormalized coefficients taken from Schubert et al. (2004)
    unnormalized_cosine_coefficients = np.zeros(shape=(3, 3))
    unnormalized_cosine_coefficients[0, 0] = 1.0
    unnormalized_cosine_coefficients[2, 0] = -1859.5E-6
    unnormalized_cosine_coefficients[2, 2] = 558.8E-6

    unnormalized_sine_coefficients = np.zeros(shape=(3, 3))

    normalized_cosine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_cosine_coefficients)
    normalized_sine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_sine_coefficients)

    io_spherical_harmonic_gravity_field = environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter=io_gravitational_parameter,
        reference_radius=io_reference_radius,
        normalized_cosine_coefficients=normalized_cosine_coefficients,
        normalized_sine_coefficients=normalized_sine_coefficients,
        associated_reference_frame=io_body_fixed_frame)

    return io_spherical_harmonic_gravity_field


def get_europa_gravity_field_settings(europa_body_fixed_frame='IAU_Europa'):
    """
    Sets the spherical harmonic gravity field coefficients of Europa according to the values provided by
    Schubert et al. (2004).

    Parameters
    ----------
    europa_body_fixed_frame : string (default=IAU_Europa)
        Body fixed frame of Europa

    Returns
    -------
    europa_spherical_harmonic_gravity_field : tudatpy.kernel.numerical_simulation.environment_setup.
                                              gravity_field.SphericalHarmonicsGravityFieldSettings
        The settings for Europa's spherical harmonic gravity field (coefficients).
    """
    # Gravitational parameter taken from Schubert et al. (2004)
    europa_gravitational_parameter = 3202.72E9
    europa_reference_radius = spice_interface.get_average_radius('Europa')

    # Unnormalized coefficients taken from Schubert et al. (2004)
    unnormalized_cosine_coefficients = np.zeros(shape=(5, 5))
    unnormalized_cosine_coefficients[0, 0] = 1.0
    unnormalized_cosine_coefficients[2, 0] = -435.5E-6
    unnormalized_cosine_coefficients[2, 2] = 131.5E-6

    unnormalized_sine_coefficients = np.zeros(shape=(5, 5))

    normalized_cosine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_cosine_coefficients)
    normalized_sine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_sine_coefficients)

    europa_spherical_harmonic_gravity_field = environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter=europa_gravitational_parameter,
        reference_radius=europa_reference_radius,
        normalized_cosine_coefficients=normalized_cosine_coefficients,
        normalized_sine_coefficients=normalized_sine_coefficients,
        associated_reference_frame=europa_body_fixed_frame)

    return europa_spherical_harmonic_gravity_field


def get_ganymede_gravity_field_settings(ganymede_body_fixed_frame='IAU_Ganymede'):
    """
    Sets the spherical harmonic gravity field coefficients of Ganymede according to the values provided by
    Schubert et al. (2004).

    Parameters
    ----------
    ganymede_body_fixed_frame : string (default=IAU_Ganymede)
        Body fixed frame of Ganymede

    Returns
    -------
    Ganymede_spherical_harmonic_gravity_field : tudatpy.kernel.numerical_simulation.environment_setup.
                                                gravity_field.SphericalHarmonicsGravityFieldSettings
        The settings for Ganymede's spherical harmonic gravity field (coefficients).
    """
    # Gravitational parameter taken from Schubert et al. (2004)
    ganymede_gravitational_parameter = 9887.83E9
    ganymede_reference_radius = spice_interface.get_average_radius('Ganymede')

    # Unnormalized coefficients taken from Schubert et al. (2004)
    unnormalized_cosine_coefficients = np.zeros(shape=(13, 13))
    unnormalized_cosine_coefficients[0, 0] = 1.0
    unnormalized_cosine_coefficients[2, 0] = -127.53E-6
    unnormalized_cosine_coefficients[2, 2] = 38.26E-6

    unnormalized_sine_coefficients = np.zeros(shape=(13, 13))

    normalized_cosine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_cosine_coefficients)
    normalized_sine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_sine_coefficients)

    ganymede_spherical_harmonic_gravity_field = environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter=ganymede_gravitational_parameter,
        reference_radius=ganymede_reference_radius,
        normalized_cosine_coefficients=normalized_cosine_coefficients,
        normalized_sine_coefficients=normalized_sine_coefficients,
        associated_reference_frame=ganymede_body_fixed_frame)

    return ganymede_spherical_harmonic_gravity_field


def get_callisto_gravity_field_settings(callisto_body_fixed_frame='IAU_Callisto'):
    """
    Sets the spherical harmonic gravity field coefficients of Callisto according to the values provided by
    Schubert et al. (2004).

    Parameters
    ----------
    callisto_body_fixed_frame : string (default=IAU_Callisto)
        Body fixed frame of Callisto

    Returns
    -------
    callisto_spherical_harmonic_gravity_field : tudatpy.kernel.numerical_simulation.environment_setup.
                                                gravity_field.SphericalHarmonicsGravityFieldSettings
        The settings for Callisto's spherical harmonic gravity field (coefficients).
    """
    # Gravitational parameter taken from Schubert et al. (2004)
    callisto_gravitational_parameter = 7179.29E9
    callisto_reference_radius = spice_interface.get_average_radius('Callisto')

    # Unnormalized coefficients taken from Schubert et al. (2004)
    unnormalized_cosine_coefficients = np.zeros(shape=(7, 7))
    unnormalized_cosine_coefficients[0, 0] = 1.0
    unnormalized_cosine_coefficients[2, 0] = -32.7E-6
    unnormalized_cosine_coefficients[2, 2] = 10.2E-6

    unnormalized_sine_coefficients = np.zeros(shape=(7, 7))

    normalized_cosine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_cosine_coefficients)
    normalized_sine_coefficients = normalize_spherical_harmonic_coefficients(unnormalized_sine_coefficients)

    callisto_spherical_harmonic_gravity_field = environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter=callisto_gravitational_parameter,
        reference_radius=callisto_reference_radius,
        normalized_cosine_coefficients=normalized_cosine_coefficients,
        normalized_sine_coefficients=normalized_sine_coefficients,
        associated_reference_frame=callisto_body_fixed_frame)

    return callisto_spherical_harmonic_gravity_field


### ROTATION MODEL ###

def define_tidally_locked_rotation_galilean_moons(body_settings):
    """
    Defines a tidally locked rotation for all four Galilean satellites.

    Parameters
    ----------
    body_settings: tudatpy.kernel.numerical_simulation.environment_setup.BodyListSettings
        List that contains a set of body settings.
    """
    # Define overall parameters describing the synchronous rotation model
    central_body_name = 'Jupiter'
    original_frame = 'ECLIPJ2000'
    # Define satellite specific parameters and change rotation model settings
    target_frame = 'IAU_Io'
    body_settings.get('Io').rotation_model_settings = environment_setup.rotation_model.synchronous(
        central_body_name, original_frame, target_frame)
    target_frame = 'IAU_Europa'
    body_settings.get('Europa').rotation_model_settings = environment_setup.rotation_model.synchronous(
        central_body_name, original_frame, target_frame)
    target_frame = 'IAU_Ganymede'
    body_settings.get('Ganymede').rotation_model_settings = environment_setup.rotation_model.synchronous(
        central_body_name, original_frame, target_frame)
    target_frame = 'IAU_Callisto'
    body_settings.get('Callisto').rotation_model_settings = environment_setup.rotation_model.synchronous(
        central_body_name, original_frame, target_frame)
