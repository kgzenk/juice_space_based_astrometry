
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import math
import numpy as np
# tudatpy imports
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
# Utilities
from validation import validation_util as Util

current_dir = os.path.dirname(__file__)
validation_output_path = os.path.join(os.path.dirname(current_dir), 'tests')

###########################################################################
# SPACE-BASED ASTROMETRY UTILITIES ########################################
###########################################################################

def space_based_astrometry_epochs_errors(dependent_variables_dict,
                                         closest_approach_input_path):
    # Extract epochs and dependent variables from dictionary
    epochs = np.array(list(dependent_variables_dict.keys()))
    dependent_variables = np.vstack(list(dependent_variables_dict.values()))
    # Split dependent variables into individual lists
    distance_juice_io = dependent_variables[:, 0:3]
    distance_juice_sun = dependent_variables[:, 3:6]
    distance_juice_jupiter = dependent_variables[:, 6:9]
    distance_jupiter_io = dependent_variables[:, 9:12]
    distance_io_sun = dependent_variables[:, 12:15]

    flybys_dict = Util.determine_closest_approaches(closest_approach_input_path)
    flybys_epochs = np.sort(np.concatenate(list(flybys_dict.values()), axis=0))
    flybys_epochs_idx = 0

    # Reference radius taken from Iess et al. (2018)
    jupiter_radius = 71492E3
    io_radius = spice_interface.get_average_radius('Io')

    # Select correct camera settings for JUICE NavCam
    camera_fov = 4  # [deg]
    resolution = 1024  # [pixels]

    # Define return-object
    epochs_with_errors = dict()
    ganymede_orbit_insertion_epoch = 34.967 * constants.JULIAN_YEAR
    flyby_initiated = False
    # Loop over all epochs over the course of the orbit
    for idx, epoch in enumerate(epochs):

        if epoch > ganymede_orbit_insertion_epoch:
            break

        if flybys_epochs[flybys_epochs_idx] - 12 * 3600 < epoch < \
                flybys_epochs[flybys_epochs_idx] + 12 * 3600:
            flyby_initiated = True
            continue
        elif flyby_initiated and epoch > flybys_epochs[flybys_epochs_idx] + 12 * 3600:
            flyby_initiated = False
            flybys_epochs_idx =+ 1

        image_diameter = 2 * np.tan((camera_fov / 2) * math.pi / 180) * np.linalg.norm(distance_juice_io[idx])
        spatial_resolution = image_diameter / resolution
        # pixels_filled_by_io = math.pi * (io_radius / spatial_resolution) ** 2

        ### SUN-SPACECRAFT-MOON ANGLE ###
        sun_spacecraft_moon_angle = Util.angle_between(distance_juice_sun[idx], distance_juice_io[idx])
        if sun_spacecraft_moon_angle > 30 * math.pi / 180:
            sun_spacecraft_moon_angle_temp = True
        else:
            sun_spacecraft_moon_angle_temp = False

        ### JUPITER-LIMB-SPACECRAFT-MOON ANGLE ###
        apparent_size_jupiter = 2 * np.arcsin(jupiter_radius / np.linalg.norm(distance_juice_jupiter[idx]))
        distance_juice_jupiter_limb = np.add(distance_juice_jupiter[idx],
                                             jupiter_radius * Util.unit_vector(distance_jupiter_io[idx]))
        jupiter_limb_spacecraft_moon_angle = Util.angle_between(distance_juice_jupiter_limb, distance_juice_io[idx])
        if apparent_size_jupiter > 4 * math.pi / 180:
            if jupiter_limb_spacecraft_moon_angle >= 5 * math.pi / 180:
                jupiter_limb_spacecraft_moon_angle_temp = True
            else:
                jupiter_limb_spacecraft_moon_angle_temp = False
        else:
            if jupiter_limb_spacecraft_moon_angle >= 10 * math.pi / 180:
                jupiter_limb_spacecraft_moon_angle_temp = True
            else:
                jupiter_limb_spacecraft_moon_angle_temp = False

        ### SUN-MOON-SPACECRAFT ANGLE ###
        sun_moon_spacecraft_angle = Util.angle_between(distance_io_sun[idx], -1 * distance_juice_io[idx])
        if sun_moon_spacecraft_angle < 130 * math.pi / 180:
            sun_moon_spacecraft_angle_temp = True
        else:
            sun_moon_spacecraft_angle_temp = False

        criteria_comprehensive = [sun_spacecraft_moon_angle_temp,
                                  jupiter_limb_spacecraft_moon_angle_temp,
                                  sun_moon_spacecraft_angle_temp]

        if np.all(criteria_comprehensive):
            ### SPACE-BASED ASTROMETRY UNCERTAINTY ###
            pointing_uncertainty = 0.2741 * math.pi / 648000  # [rad]
            # pointing_uncertainty = \
            #     2 * np.tan(0.5 * pointing_uncertainty) * np.linalg.norm(distance_juice_io[idx])  # [m]

            centre_of_figure_uncertainty = \
                np.sqrt(0.095 ** 2 + (0.0014 * (2 * io_radius / spatial_resolution)) ** 2)  # [pixels]
            centre_of_figure_uncertainty = spatial_resolution * centre_of_figure_uncertainty  # [m]
            centre_of_figure_uncertainty = \
                2 * np.arctan(centre_of_figure_uncertainty / (2 * np.linalg.norm(distance_juice_io[idx])))  # [rad]
            # np.tan(uncertainty_history_astrometry[:, 1]) * distance_history_astrometry * 1E-3

            spacecraft_position_uncertainty = np.arcsin(100 / np.linalg.norm(distance_juice_io[idx]))  # [rad]
            # spacecraft_position_uncertainty = \
            #     2 * np.tan(0.5 * spacecraft_position_uncertainty) * np.linalg.norm(distance_juice_io[idx])  # [m]

            ### CONVERSION TO UNCERTAINTIES IN CELESTIAL ELEMENTS ###
            declination = np.arctan(
                distance_juice_io[idx, 2] / np.sqrt(distance_juice_io[idx, 0] ** 2 + distance_juice_io[idx, 1] ** 2))

            uncertainty_declination = np.sqrt(pointing_uncertainty ** 2 +
                                              centre_of_figure_uncertainty ** 2 +
                                              spacecraft_position_uncertainty ** 2)  # [rad]

            uncertainty_right_ascension = uncertainty_declination / np.cos(declination)

            # Uncertainties in rad, distance in metres
            epochs_with_errors[epoch] = [uncertainty_right_ascension,
                                         uncertainty_declination,
                                         np.linalg.norm(distance_juice_io[idx])]

    return epochs_with_errors


def get_astrometry_arcs(epochs_with_errors_path=None,
                        simulation_end_epoch=None):
    ### LOAD DATA ###
    if epochs_with_errors_path is None:
        epochs_with_errors_path = os.path.join(validation_output_path, 'validation_data/space_based_astrometry_data',
                                               'epochs_with_errors_list.dat')
    epochs_with_errors_list = np.loadtxt(epochs_with_errors_path)
    if simulation_end_epoch is None:
        simulation_end_epoch = epochs_with_errors_list[-1, 0]

    ### MANIPULATE DATA ###
    epochs_with_errors_dict = dict()
    epochs_with_errors_metric_dict = dict()
    for idx in range(len(epochs_with_errors_list)):
        if epochs_with_errors_list[idx, 0] > simulation_end_epoch:
            break
        epochs_with_errors_dict[epochs_with_errors_list[idx, 0]] = epochs_with_errors_list[idx, 1:]
        epochs_with_errors_metric_dict[epochs_with_errors_list[idx, 0]] = [
            2 * np.tan(0.5 * epochs_with_errors_list[idx, 1]) * epochs_with_errors_list[idx, -1],
            2 * np.tan(0.5 * epochs_with_errors_list[idx, 2]) * epochs_with_errors_list[idx, -1]
        ]

    ### RELATIVE GEOMETRY - IO-FIXED FRAME ###
    juice_wrt_to_io_normalized = dict()
    for epoch in epochs_with_errors_dict.keys():
        juice_from_spice = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name='JUICE',
            observer_body_name='Io',
            reference_frame_name='IAU_Io',
            aberration_corrections='none',
            ephemeris_time=epoch)
        juice_wrt_to_io_normalized[epoch] = np.divide(juice_from_spice[:3], np.linalg.norm(juice_from_spice[:3]))

    ### DEFINE COHERENT ASTROMETRY-ARCS ###
    # Initialise containers
    single_arcs_astrometry = dict()
    single_arcs_astrometry_idx_list = list()
    astrometry_epochs = np.array(list(epochs_with_errors_dict.keys()))
    # Set overall first epoch as initial past epoch
    past_epoch = astrometry_epochs[0]
    # Initialise arc-counter and append past epoch to dictionary of initial arc
    single_arcs_idx = 0
    single_arcs_astrometry[single_arcs_idx] = [past_epoch]
    # Loop over all epochs skipping the already allocated initial epoch
    for epoch_idx, epoch in enumerate(astrometry_epochs[0:]):
        present_epoch = epoch
        present_arc = single_arcs_astrometry[single_arcs_idx]
        if present_epoch - past_epoch <= 48 * 3600:
            present_arc.append(present_epoch)
            if epoch_idx == len(astrometry_epochs[0:] - 1):
                present_arc = np.array(present_arc)
            single_arcs_astrometry[single_arcs_idx] = present_arc
            single_arcs_astrometry_idx_list.append(single_arcs_idx)
        else:
            present_arc = np.array(present_arc)
            single_arcs_astrometry[single_arcs_idx] = present_arc
            single_arcs_idx += 1
            single_arcs_astrometry[single_arcs_idx] = [present_epoch]
        past_epoch = present_epoch

    single_arcs_astrometry_idx_list = np.array(single_arcs_astrometry_idx_list)
    if single_arcs_astrometry_idx_list[0] == 0:
        single_arcs_astrometry_idx_list = np.delete(single_arcs_astrometry_idx_list, 0)

    return epochs_with_errors_dict, epochs_with_errors_metric_dict, juice_wrt_to_io_normalized, \
        single_arcs_astrometry, single_arcs_astrometry_idx_list


def get_angles_with_unit_vectors(juice_wrt_to_io_normalized):
    # Define unit-vectors in Io-fixed frame (because plane normal vector is in IAU-Io)
    x_unit_vector = np.array([1.0, 0.0, 0.0])
    y_unit_vector = np.array([0.0, 1.0, 0.0])
    z_unit_vector = np.array([0.0, 0.0, 1.0])

    juice_angle_with_optical_plane = dict()
    for epoch in juice_wrt_to_io_normalized.keys():
        angle_with_x = np.arccos(np.clip(np.dot(juice_wrt_to_io_normalized[epoch], x_unit_vector), -1.0, 1.0))
        angle_with_y = np.arccos(np.clip(np.dot(juice_wrt_to_io_normalized[epoch], y_unit_vector), -1.0, 1.0))
        angle_with_z = np.arccos(np.clip(np.dot(juice_wrt_to_io_normalized[epoch], z_unit_vector), -1.0, 1.0))
        juice_angle_with_optical_plane[epoch] = np.array([angle_with_x, angle_with_y, angle_with_z])

    return juice_angle_with_optical_plane
