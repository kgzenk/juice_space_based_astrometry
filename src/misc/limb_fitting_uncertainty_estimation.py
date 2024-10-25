
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import math
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
# tudatpy imports
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
# Plotting imports
from matplotlib import pyplot as plt

### MATPLOTLIB STANDARD SETTINGS ###
# Grid
plt.rcParams['axes.grid'] = True
plt.rc('grid', linestyle='-.', alpha=0.5)
# Font and Size
plt.rc('font', family='serif', size=18)
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['figure.figsize'] = (16, 9)
# Line- and marker-size, tick position
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'lines.markersize': 9})
plt.rcParams['axes.axisbelow'] = True
# Tight layout and always save as png
plt.rcParams['figure.autolayout'] = True
plt.rcParams['savefig.format'] = 'png'


# Get path of current directory
current_dir = os.path.dirname(__file__)
image_save_path = os.path.join(current_dir, 'validation_plots')

###########################################################################
# DEFINE FUNCTIONS FOR LEAST SQUARES ######################################
###########################################################################

def centre_of_figure_uncertainty(apparent_diameter, sigma_min, c):
    return np.sqrt(sigma_min ** 2 + (c * apparent_diameter) ** 2)


if __name__ == '__main__':

    # Get path of current directory
    current_dir = os.path.dirname(__file__)

    ###########################################################################
    # DEFINE SIMULATION SETTINGS ##############################################
    ###########################################################################

    # Cassini: July 2004 (1st) / December 2012 (31st)
    simulation_start_epoch = 4.5 * constants.JULIAN_YEAR
    simulation_end_epoch = 12.0 * constants.JULIAN_YEAR + 365 * constants.JULIAN_DAY

    simulation_duration = simulation_end_epoch - simulation_start_epoch

    ###########################################################################
    # CREATE ENVIRONMENT ######################################################
    ###########################################################################

    path_to_kernels = os.path.join(current_dir, 'cassini_kernels')
    # Load all SPICE cassini_kernels for Cassini (https://ftp.imcce.fr/pub/softwares/caviar/)
    for filename in os.listdir(path_to_kernels):
        spice_interface.load_kernel(os.path.join(path_to_kernels, filename))

    ### CELESTIAL BODIES ###
    # Create default body settings for selected celestial bodies
    saturnian_moons_to_create = ['Tethys', 'Dione', 'Rhea', 'Iapetus', 'Phoebe']
    planets_to_create = ['Saturn']
    bodies_to_create = np.concatenate((saturnian_moons_to_create, planets_to_create))

    # Create default body settings for bodies_to_create, with 'Saturn'/'J2000'
    # as global frame origin and orientation.
    global_frame_origin = 'Saturn'
    global_frame_orientation = 'ECLIPJ2000'
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    # Create ephemeris settings for all Saturnian moons from spice cassini_kernels
    for moon in saturnian_moons_to_create:
        body_settings.get(moon).ephemeris_settings = \
            environment_setup.ephemeris.direct_spice("Saturn", "ECLIPJ2000")

    ### VEHICLE BODY ###
    # Create vehicle object
    body_settings.add_empty_settings('Cassini')

    # Create system of selected bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    ###########################################################################
    # CREATE SIMULATION AND CAMERA SETTINGS ###################################
    ###########################################################################

    # Define overall output-path for all prediction-related validation_data
    prediction_output_path = os.path.join(current_dir, 'space_based_astrometry_data', 'prediction')
    if not os.path.exists(prediction_output_path):
        os.makedirs(prediction_output_path)

    ###########################################################################
    # IMPORT DATA FROM TAJEDDINE ET AL. 2015 ##################################
    ###########################################################################

    astrometry_tethys_dione_rhea_iapetus_phoebe = np.loadtxt(os.path.join(
        current_dir,
        'space_based_astrometry_data/tethys_dione_rhea_iapetus_phoebe_astrometry.dat'), dtype='str')

    astrometry_tethys = dict()
    astrometry_dione = dict()
    astrometry_rhea = dict()
    astrometry_iapetus = dict()
    astrometry_phoebe = dict()

    for line in astrometry_tethys_dione_rhea_iapetus_phoebe:
        date_with_time = line[1] + '/' + line[2] + '/' + line[3] + ' ' + line[4]
        epoch_since_j2000 = time_conversion.julian_day_to_seconds_since_epoch(
            time_conversion.calendar_date_to_julian_day(datetime.strptime(date_with_time, '%Y/%b/%d %H:%M:%S.%f')))
        if line[5] == 'TETHYS':
            distance_cassini_moon = np.linalg.norm(spice_interface.get_body_cartesian_position_at_epoch(
                target_body_name='Cassini',
                observer_body_name='TETHYS',
                reference_frame_name='ECLIPJ2000',
                aberration_corrections='none',
                ephemeris_time=epoch_since_j2000), axis=0)
            astrometry_tethys[epoch_since_j2000] = [float(line[9]), int(line[15]), distance_cassini_moon]
        elif line[5] == 'DIONE':
            distance_cassini_moon = np.linalg.norm(spice_interface.get_body_cartesian_position_at_epoch(
                target_body_name='Cassini',
                observer_body_name='DIONE',
                reference_frame_name='ECLIPJ2000',
                aberration_corrections='none',
                ephemeris_time=epoch_since_j2000), axis=0)
            astrometry_dione[epoch_since_j2000] = [float(line[9]), int(line[15]), distance_cassini_moon]
        elif line[5] == 'RHEA':
            distance_cassini_moon = np.linalg.norm(spice_interface.get_body_cartesian_position_at_epoch(
                target_body_name='Cassini',
                observer_body_name='RHEA',
                reference_frame_name='ECLIPJ2000',
                aberration_corrections='none',
                ephemeris_time=epoch_since_j2000), axis=0)
            astrometry_rhea[epoch_since_j2000] = [float(line[9]), int(line[15]), distance_cassini_moon]
        elif line[5] == 'IAPETUS':
            distance_cassini_moon = np.linalg.norm(spice_interface.get_body_cartesian_position_at_epoch(
                target_body_name='Cassini',
                observer_body_name='IAPETUS',
                reference_frame_name='ECLIPJ2000',
                aberration_corrections='none',
                ephemeris_time=epoch_since_j2000), axis=0)
            astrometry_iapetus[epoch_since_j2000] = [float(line[9]), int(line[15]), distance_cassini_moon]
        elif line[5] == 'PHOEBE':
            distance_cassini_moon = np.linalg.norm(spice_interface.get_body_cartesian_position_at_epoch(
                target_body_name='Cassini',
                observer_body_name='PHOEBE',
                reference_frame_name='ECLIPJ2000',
                aberration_corrections='none',
                ephemeris_time=epoch_since_j2000), axis=0)
            astrometry_phoebe[epoch_since_j2000] = [float(line[9]), int(line[15]), distance_cassini_moon]

    astrometry_raw = {'TETHYS': astrometry_tethys,
                      'DIONE': astrometry_dione,
                      'RHEA': astrometry_rhea,
                      'IAPETUS': astrometry_iapetus,
                      'PHOEBE': astrometry_phoebe}

    astrometry_refined = dict()
    popt_container = list()
    perr_container = list()

    centre_of_figure_uncertainty_list = list()
    apparent_diameter_list = list()

    spatial_resolution_list = list()
    distance_cassini_moon_list = list()
    spatial_resolution_list_cassini = list()
    apparent_diameter_cassini_list = list()

    # Cassini ISS NAC settings
    camera_fov = 0.35 * math.pi / 180  # [rad]
    camera_fov_navcam = 4 * math.pi / 180  # [rad]
    resolution = 1024  # [pixels]
    sigma_extraction = 0.552  # [pixels]

    scaling_factor = camera_fov_navcam / camera_fov

    for saturnian_moon in astrometry_raw.keys():

        spatial_resolution_list_temp = list()
        distance_cassini_moon_list_temp = list()
        spatial_resolution_list_temp_cassini = list()
        apparent_diameter_temp_cassini_list = list()

        data = np.vstack(list(astrometry_raw[saturnian_moon].values()))
        epoch_since_j2000 = np.array(list(astrometry_raw[saturnian_moon].keys()))
        moon_radius = spice_interface.get_average_radius(saturnian_moon)
        data_for_least_squares = dict()

        for idx in range(len(data)):

            uncertainty_declination = data[idx, 0] * math.pi / 180  # [rad]
            number_of_stars = data[idx, 1]  # [-]
            distance_cassini_moon = data[idx, 2]  # [m]

            pointing_uncertainty = (camera_fov * sigma_extraction) / (resolution * np.sqrt(number_of_stars))  # [rad]
            spacecraft_position_uncertainty = np.arcsin(100 / distance_cassini_moon)  # [rad]

            image_diameter_cassini = 2 * np.tan((camera_fov / 2)) * distance_cassini_moon
            spatial_resolution_cassini = image_diameter_cassini / resolution
            apparent_diameter_temp_cassini = 2 * moon_radius / spatial_resolution_cassini

            image_diameter = 2 * np.tan(((camera_fov * scaling_factor) / 2)) * distance_cassini_moon
            spatial_resolution = image_diameter / resolution
            apparent_diameter_temp = 2 * moon_radius / spatial_resolution

            if uncertainty_declination ** 2 - pointing_uncertainty ** 2 - spacecraft_position_uncertainty ** 2 >= 0:

                centre_of_figure_uncertainty_temp = \
                    scaling_factor * np.sqrt(uncertainty_declination ** 2 -
                                             pointing_uncertainty ** 2 -
                                             spacecraft_position_uncertainty ** 2)  # [rad]

                centre_of_figure_uncertainty_temp = (np.tan(0.5 * centre_of_figure_uncertainty_temp) *
                                                     (2 * distance_cassini_moon)) / spatial_resolution  # [pixels]

                data_for_least_squares[epoch_since_j2000[idx]] = [centre_of_figure_uncertainty_temp,
                                                                  apparent_diameter_temp]

                spatial_resolution_list_temp.append(spatial_resolution)
                distance_cassini_moon_list_temp.append(distance_cassini_moon)
                spatial_resolution_list_temp_cassini.append(spatial_resolution_cassini)
                apparent_diameter_temp_cassini_list.append(apparent_diameter_temp_cassini)

        astrometry_refined[saturnian_moon] = data_for_least_squares

        centre_of_figure_uncertainty_lls = np.vstack(list(astrometry_refined[saturnian_moon].values()))[:, 0]
        apparent_diameter_lls = np.vstack(list(astrometry_refined[saturnian_moon].values()))[:, 1]

        centre_of_figure_uncertainty_list.append(centre_of_figure_uncertainty_lls)
        apparent_diameter_list.append(apparent_diameter_lls)

        spatial_resolution_list.append(spatial_resolution_list_temp)
        distance_cassini_moon_list.append(distance_cassini_moon_list_temp)
        spatial_resolution_list_cassini.append(spatial_resolution_list_temp_cassini)
        apparent_diameter_cassini_list.append(apparent_diameter_temp_cassini_list)

        min_pixels = np.min(centre_of_figure_uncertainty_lls)
        min_pixels_c = np.min(centre_of_figure_uncertainty_lls) / apparent_diameter_lls[
            np.argmin(centre_of_figure_uncertainty_lls)]
        max_pixels = np.max(centre_of_figure_uncertainty_lls)
        max_pixels_c = np.max(centre_of_figure_uncertainty_lls) / apparent_diameter_lls[
            np.argmax(centre_of_figure_uncertainty_lls)]

        popt, pcov, *args = curve_fit(centre_of_figure_uncertainty,
                                      apparent_diameter_lls,
                                      centre_of_figure_uncertainty_lls,
                                      p0=[min_pixels, min_pixels_c],
                                      bounds=([0.0, min_pixels_c], [max_pixels, max_pixels_c]))

        popt_container.append(popt)
        perr = np.sqrt(np.diag(pcov))
        perr_container.append(perr)

        residuals = \
            (spatial_resolution_list_temp * centre_of_figure_uncertainty_lls -
             spatial_resolution_list_temp * centre_of_figure_uncertainty(apparent_diameter_lls, popt[0], popt[1]))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((spatial_resolution_list_temp * centre_of_figure_uncertainty_lls -
                         np.mean(spatial_resolution_list_temp * centre_of_figure_uncertainty_lls)) ** 2)

        r_squared_temp = 1 - (ss_res / ss_tot)

        print(r_squared_temp)

    popt_container = np.array(popt_container)

    print(np.average(popt_container, axis=0))
    print(np.average(perr_container, axis=0))
    print(np.average(np.add(popt_container, np.multiply(1, perr_container)), axis=0))

    fig, axes = plt.subplots(5, 2, figsize=(15, 19), sharex='col', sharey=True, constrained_layout=True)
    title_verbose = ['Tethys', 'Dione', 'Rhea', 'Iapetus', 'Phoebe']

    for idx, ax in enumerate(axes[:, 0]):
        ax.set_title(title_verbose[idx])
        ax.scatter(apparent_diameter_list[idx],
                   spatial_resolution_list[idx] * centre_of_figure_uncertainty_list[idx] * 1E-3, label='Scaled Data',
                   c='#0076C2')
        if idx == 3:
            ax.scatter(apparent_diameter_list[idx],
                       spatial_resolution_list[idx] * centre_of_figure_uncertainty(
                           apparent_diameter_list[idx], 0.25, 0.02) * 1E-3,
                       label='Original Values', c='#A50034')
        else:
            ax.scatter(apparent_diameter_list[idx],
                       spatial_resolution_list[idx] * centre_of_figure_uncertainty(
                           apparent_diameter_list[idx], 0.25, 0.01) * 1E-3,
                       label='Original Values', c='#A50034')
        ax.scatter(apparent_diameter_list[idx],
                   spatial_resolution_list[idx] * centre_of_figure_uncertainty(
                       apparent_diameter_list[idx], popt_container[idx, 0], popt_container[idx, 1]) * 1E-3,
                   label='Individual Fit', c='#009B77')
        ax.scatter(apparent_diameter_list[idx],
                   spatial_resolution_list[idx] * centre_of_figure_uncertainty(
                       apparent_diameter_list[idx], 0.095, 0.0014) * 1E-3,
                   label='Average Fit', c='#EC6842')
        ax.set_yscale('log')

    for idx, ax in enumerate(axes[:, 1]):
        ax.set_title(title_verbose[idx])
        ax.scatter(np.multiply(distance_cassini_moon_list[idx], 1E-3),
                   spatial_resolution_list[idx] * centre_of_figure_uncertainty_list[idx] * 1E-3, label='Scaled Data',
                   c='#0076C2')
        if idx == 3:
            ax.scatter(np.multiply(distance_cassini_moon_list[idx], 1E-3),
                       spatial_resolution_list[idx] * centre_of_figure_uncertainty(
                           apparent_diameter_list[idx], 0.25, 0.02) * 1E-3,
                       label='Original Values', c='#A50034')
        else:
            ax.scatter(np.multiply(distance_cassini_moon_list[idx], 1E-3),
                       spatial_resolution_list[idx] * centre_of_figure_uncertainty(
                           apparent_diameter_list[idx], 0.25, 0.01) * 1E-3,
                       label='Original Values', c='#A50034')
        ax.scatter(np.multiply(distance_cassini_moon_list[idx], 1E-3),
                   spatial_resolution_list[idx] * centre_of_figure_uncertainty(
                       apparent_diameter_list[idx], popt_container[idx, 0], popt_container[idx, 1]) * 1E-3,
                   label='Individual Fit', c='#009B77')
        ax.scatter(np.multiply(distance_cassini_moon_list[idx], 1E-3),
                   spatial_resolution_list[idx] * centre_of_figure_uncertainty(
                       apparent_diameter_list[idx], 0.095, 0.0014) * 1E-3,
                   label='Average Fit', c='#EC6842')

    axes[-1, 0].set_xlabel('Apparent Diameter [pixels]')
    axes[-1, 1].set_xlabel('Distance Cassini-Moon [km]')
    axes[2, 0].set_ylabel(r'Centre-of-Figure Uncertainty $\sigma_\mathrm{c}$ [km]')
    axes[0, 1].legend()

    plt.savefig(os.path.join(image_save_path, 'centre_of_figure_uncertainty_validation.png'))
