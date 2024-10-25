
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
# tudatpy imports
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
# Problem-specific imports
from src.core.tudat_setup_util.env.env_utilities_jupiter import get_jupiter_gravity_field_settings
from src.core.tudat_setup_util.env.env_utilities_galilean_moons import get_io_gravity_field_settings
from src.core.tudat_setup_util.env.env_utilities_galilean_moons import get_europa_gravity_field_settings
from src.core.tudat_setup_util.env.env_utilities_galilean_moons import get_ganymede_gravity_field_settings
from src.core.tudat_setup_util.env.env_utilities_galilean_moons import get_callisto_gravity_field_settings
from src.core.tudat_setup_util.env.env_utilities_galilean_moons import define_tidally_locked_rotation_galilean_moons


###########################################################################
# ENVIRONMENT SETTING UTILITIES ###########################################
###########################################################################

def create_simulation_bodies(simulation_start_epoch,
                             simulation_end_epoch,
                             moons_to_propagate,
                             io_com_cof_offset=None):
    """
    Creates the set of bodies relevant for the simulation. This includes both
    natural (moons, planets, stars) and artificial (JUICE) bodies.

    Parameters
    ----------
    simulation_start_epoch: float
        Start of the simulation [s] with t=0 at J2000.
    simulation_end_epoch: float
        End of the simulation [s] with t=0 at J2000.
    moons_to_propagate: list[String]
    io_com_cof_offset: ndarray[float]

    Returns
    -------
    body_settings: tudatpy.kernel.numerical_simulation.environment_setup.BodyListSettings
        List that contains a set of body settings relative to bodies object.
    bodies : tudatpy.kernel.numerical_simulation.environment_setup.SystemOfBodies
        Object that contains a set of body objects and associated frame information.
    """
    ### CELESTIAL BODIES ###
    # Create default body settings for selected celestial bodies
    jovian_moons_to_create = ['Io', 'Europa', 'Ganymede', 'Callisto']
    planets_to_create = ['Jupiter', 'Saturn']
    stars_to_create = ['Sun']
    bodies_to_create = np.concatenate((jovian_moons_to_create, planets_to_create, stars_to_create))

    # Create default body settings for bodies_to_create, with 'Jupiter'/'J2000'
    # as global frame origin and orientation.
    global_frame_origin = 'Jupiter'
    global_frame_orientation = 'ECLIPJ2000'
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    for moon in moons_to_propagate:
        body_settings.get(moon).ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
            body_settings.get(moon).ephemeris_settings,
            int(simulation_start_epoch) - 10 * constants.JULIAN_DAY,
            int(simulation_end_epoch) + 10 * constants.JULIAN_DAY,
            time_step=3600.0)

    # Adapt physical settings of the celestial bodies
    # Gravity fields
    body_settings.get('Jupiter').gravity_field_settings = get_jupiter_gravity_field_settings()
    if io_com_cof_offset is None:
        body_settings.get('Io').gravity_field_settings = get_io_gravity_field_settings()
    else:
        body_settings.get('Io').gravity_field_settings = get_io_gravity_field_settings(
            com_cof_offset=io_com_cof_offset)
    body_settings.get('Europa').gravity_field_settings = get_europa_gravity_field_settings()
    body_settings.get('Ganymede').gravity_field_settings = get_ganymede_gravity_field_settings()
    body_settings.get('Callisto').gravity_field_settings = get_callisto_gravity_field_settings()

    # Tidally locked rotation models
    define_tidally_locked_rotation_galilean_moons(body_settings)

    ### VEHICLE BODY ###
    # Create vehicle object
    body_settings.add_empty_settings('JUICE')
    # Create ephemeris settings for JUICE and Clipper from spice cassini_kernels
    body_settings.get('JUICE').ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
        environment_setup.ephemeris.direct_spice("Jupiter", "ECLIPJ2000"),
        simulation_start_epoch - 10 * constants.JULIAN_DAY,
        simulation_end_epoch + 10 * constants.JULIAN_DAY,
        time_step=60.0)

    # Create system of selected bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    ### VEHICLE BODY ###
    # Set constant mass for JUICE
    bodies.get('JUICE').set_constant_mass(2400.0)
    # Create radiation pressure coefficients interface
    reference_area_radiation = 100.0
    radiation_pressure_coefficient = 1.2
    occulting_bodies = ['Jupiter']
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        'Sun', reference_area_radiation, radiation_pressure_coefficient, occulting_bodies)
    environment_setup.add_radiation_pressure_interface(
        bodies, 'JUICEJUICE', radiation_pressure_settings)

    return body_settings, bodies
