
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
# tudatpy imports
from tudatpy.kernel.astro import frame_conversion

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
image_save_path = os.path.join(current_dir, 'estimation_plots')
raw_data_output_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data_cov_analysis')

def plot_monte_carlo_results(total_number_of_observations):
    ### LOAD DATA ###
    number_of_observations_dict = dict()

    for total_number_of_observations_loop in total_number_of_observations:
        # Define output-paths
        output_path_dict = {
            0: os.path.join(raw_data_output_path, 'monte_carlo/random', str(total_number_of_observations_loop)),
            1: os.path.join(raw_data_output_path, 'monte_carlo/geometry', str(total_number_of_observations_loop)),
            2: os.path.join(raw_data_output_path, 'monte_carlo/uncertainty', str(total_number_of_observations_loop)),
            3: os.path.join(raw_data_output_path, 'monte_carlo/hybrid', str(total_number_of_observations_loop))
        }

        different_methods_dict = dict()
        for i in output_path_dict.keys():
            formal_errors_save_path = os.path.join(output_path_dict[i], 'formal_errors.dat')
            different_methods_dict[i] = np.loadtxt(formal_errors_save_path)

        number_of_observations_dict[total_number_of_observations_loop] = different_methods_dict

    ### MANIPULATE DATA ###
    dict2plot = dict()
    for number_of_observations in number_of_observations_dict.keys():
        number_of_observations_dict_temp = number_of_observations_dict[number_of_observations]
        for i in number_of_observations_dict_temp.keys():
            formal_errors = number_of_observations_dict_temp[i][:, -3:]
            list2plot = np.array([number_of_observations, np.average(formal_errors[:, 0]),
                                  np.average(formal_errors[:, 1]), np.average(formal_errors[:, 2])])
            if i in dict2plot:
                list2plot_temp = dict2plot[i]
                dict2plot[i] = np.append(list2plot_temp, [list2plot], axis=0)
            else:
                dict2plot[i] = [list2plot]

    ### PROCESSING & WRITING OUTPUTS ###
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.5, 19.5), sharex=True, sharey=True, constrained_layout=True)
    labels = ['Purely Randomised', 'Geometry Driven', 'Uncertainty Driven', 'Hybrid Approach']
    colors = ['#fde725', '#35b779', '#31688e', '#440154']

    for i in dict2plot.keys():

        ax1.scatter(dict2plot[i][:, 0], dict2plot[i][:, 1] * 1E-3, c=colors[i], label=labels[i])

        ax2.scatter(dict2plot[i][:, 0], dict2plot[i][:, 2] * 1E-3, c=colors[i])

        ax3.scatter(dict2plot[i][:, 0], dict2plot[i][:, 3] * 1E-3, c=colors[i])

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(0.1, 10)
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    ax1.set_xlabel(r'Number of Observations [-]')
    ax1.set_ylabel(r'Formal Errors in $R_{COF}$ [km]')
    ax1.legend()

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Number of Observations [-]')
    ax2.set_ylabel(r'Formal Errors in $S_{COF}$ [km]')

    ax3.set_xscale('log')
    ax3.set_xlabel(r'Number of Observations [-]')
    ax3.set_ylabel(r'Formal Errors in $W_{COF}$ [km]')

    plt.savefig(os.path.join(image_save_path, 'monte_carlo.png'))


def plot_uncertainty_variation_results(total_number_of_observations):
    ### LOAD DATA ###
    initial_state_estimated = \
        np.loadtxt(os.path.join(raw_data_output_path, 'uncertainty_variations', 'initial_state.dat'))
    apriori_covariance_matrix = \
        np.loadtxt(os.path.join(raw_data_output_path, 'uncertainty_variations', 'apriori_covariance.dat'))

    inertial_to_rsw_io = frame_conversion.inertial_to_rsw_rotation_matrix(initial_state_estimated[0:6])
    inertial_to_rsw_europa = frame_conversion.inertial_to_rsw_rotation_matrix(initial_state_estimated[6:12])
    inertial_to_rsw_ganymede = frame_conversion.inertial_to_rsw_rotation_matrix(initial_state_estimated[12:18])
    inertial_to_rsw_callisto = frame_conversion.inertial_to_rsw_rotation_matrix(initial_state_estimated[18:24])

    ### ALTER THE A PRIORI COVARIANCE MATRIX ###
    true_to_formal_error_ratio_comprehensive = [1, 4, 7, 10, 13, 16, 19, 22, 25]
    apriori_covariance_matrix_rsw = \
        np.matmul(np.matmul(inertial_to_rsw_io, apriori_covariance_matrix[:3, :3]), np.transpose(inertial_to_rsw_io))
    inverse_apriori_covariance_rsw_map = dict()
    for true_to_formal_error_ratio in true_to_formal_error_ratio_comprehensive:
        # Invert a-priori matrix
        inverse_apriori_covariance_rsw = (
            np.linalg.inv((true_to_formal_error_ratio ** 2) * apriori_covariance_matrix_rsw.copy()))
        inverse_apriori_covariance_rsw_map[true_to_formal_error_ratio] = inverse_apriori_covariance_rsw

    formal_errors_map = list()
    formal_errors_rsw_map = list()
    correlation_matrix_map = dict()
    c_q = list()

    for total_number_of_observations_loop in total_number_of_observations:
        # Define output-paths
        output_path = (
            os.path.join(raw_data_output_path, 'uncertainty_variations', str(total_number_of_observations_loop)))

        formal_errors_per_number_of_observations = np.loadtxt(os.path.join(output_path, 'formal_errors.dat'))[:, -3:]
        correlations_per_number_of_observations = np.loadtxt(os.path.join(output_path, 'correlations.dat'))
        scaling_factors_per_number_of_observations = np.loadtxt(os.path.join(output_path, 'scaling_factors.dat'))

        formal_errors_all_per_number_of_observations = np.loadtxt(os.path.join(output_path, 'formal_errors.dat'))
        formal_errors_initial_state_rsw_per_number_of_observations = list()
        c_q_per_number_of_observations = list()
        correlation_matrix_average = list()

        for idx, correlation_matrix in enumerate(correlations_per_number_of_observations):
            correlation_matrix = correlation_matrix.reshape((4 * 6 + 3, 4 * 6 + 3))
            current_formal_errors = formal_errors_all_per_number_of_observations[idx]
            covariance_matrix = np.zeros(np.shape(correlation_matrix))

            for j in range(4 * 6 + 3):
                for k in range(4 * 6 + 3):
                    covariance_matrix[j, k] = \
                        correlation_matrix[j, k] * (current_formal_errors[j] * current_formal_errors[k])

            current_true_to_formal_error_ratio = scaling_factors_per_number_of_observations[idx, 0]
            c_q_temp = np.diagonal(np.subtract(
                np.identity(3), np.matmul(np.matmul(
                    np.matmul(inertial_to_rsw_io, covariance_matrix[:3, :3]), np.transpose(inertial_to_rsw_io)),
                    inverse_apriori_covariance_rsw_map[current_true_to_formal_error_ratio])))

            formal_errors_initial_state_rsw_per_number_of_observations.append(np.sqrt(np.abs(np.diagonal(np.matmul(
                np.matmul(inertial_to_rsw_io, covariance_matrix[:3, :3]), np.transpose(inertial_to_rsw_io))))))
            c_q_per_number_of_observations.append(c_q_temp)

            inertial_to_rsw_full = np.zeros(np.shape(covariance_matrix))
            inertial_to_rsw_diag = (
                np.array([inertial_to_rsw_io, inertial_to_rsw_io, inertial_to_rsw_europa, inertial_to_rsw_europa,
                          inertial_to_rsw_ganymede, inertial_to_rsw_ganymede, inertial_to_rsw_callisto,
                          inertial_to_rsw_callisto, np.diag(np.ones(3))]))

            for diag_idx, diag in enumerate(inertial_to_rsw_diag):
                for row_idx_temp, row in enumerate(diag):
                    for column_idx_temp, diag_value in enumerate(row):
                        inertial_to_rsw_full[3 * diag_idx + row_idx_temp, 3 * diag_idx + column_idx_temp] \
                            = diag_value

            covariance_matrix_rotate = np.matmul(np.matmul(inertial_to_rsw_full, covariance_matrix),
                                                 np.transpose(inertial_to_rsw_full))

            current_formal_errors_rotate = np.sqrt(np.abs(np.diagonal(covariance_matrix_rotate)))

            correlation_matrix_rotate = np.zeros(np.shape(correlation_matrix))
            for j in range(4 * 6 + 3):
                for k in range(4 * 6 + 3):
                    correlation_matrix_rotate[j, k] = \
                        (covariance_matrix_rotate[j, k] /
                         (current_formal_errors_rotate[j] * current_formal_errors_rotate[k]))

            correlation_matrix_average.append(correlation_matrix_rotate)

        ### MANIPULATE DATA ###
        formal_errors_map_temp = list()
        formal_errors_rsw_map_temp = list()
        correlation_matrix_average_map_temp = list()
        c_q_map_temp = list()
        values, count = np.unique(scaling_factors_per_number_of_observations, axis=0, return_counts=True)
        for idx, value_temp in enumerate(values):
            if idx == 0:
                formal_errors_average = np.average(formal_errors_per_number_of_observations[:count[idx]], axis=0)
                formal_errors_rsw_average = \
                    np.average(formal_errors_initial_state_rsw_per_number_of_observations[:count[idx]], axis=0)
                c_q_average = np.average(c_q_per_number_of_observations[:count[idx]], axis=0)
                correlation_matrix_average_temp = np.average(correlation_matrix_average[:count[idx]], axis=0)
            else:
                formal_errors_average = \
                    np.average(formal_errors_per_number_of_observations[sum(count[:idx]):sum(count[:idx + 1])], axis=0)
                formal_errors_rsw_average = np.average(
                    formal_errors_initial_state_rsw_per_number_of_observations[sum(count[:idx]):sum(count[:idx + 1])],
                    axis=0)
                c_q_average = np.average(c_q_per_number_of_observations[sum(count[:idx]):sum(count[:idx + 1])], axis=0)
                correlation_matrix_average_temp = \
                    np.average(correlation_matrix_average[sum(count[:idx]):sum(count[:idx + 1])], axis=0)

            formal_errors_map_temp.append(np.array(
                np.concatenate(([value_temp[0]], [total_number_of_observations_loop], formal_errors_average * 1E-3))))
            formal_errors_rsw_map_temp.append(np.array(np.concatenate(
                ([value_temp[0]], [total_number_of_observations_loop], formal_errors_rsw_average * 1E-3))))
            c_q_map_temp.append(np.array(np.concatenate(
                ([value_temp[0]], [total_number_of_observations_loop], c_q_average))))
            correlation_matrix_average_map_temp.append(correlation_matrix_average_temp)

        formal_errors_map.append(formal_errors_map_temp)
        formal_errors_rsw_map.append(formal_errors_rsw_map_temp)
        c_q.append(c_q_map_temp)
        correlation_matrix_map[total_number_of_observations_loop] = correlation_matrix_average_map_temp

    formal_errors_map = np.concatenate(formal_errors_map)
    formal_errors_rsw_map = np.concatenate(formal_errors_rsw_map)
    c_q = np.concatenate(c_q)

    ### PROCESSING & WRITING OUTPUTS ###
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.8), sharex=True, sharey=True, constrained_layout=True)

    plt1 = axs[0].imshow(formal_errors_map[:, 2].reshape((len(total_number_of_observations), 9)), aspect='auto',
                         vmin=np.min(formal_errors_map[:, 2:]), vmax=np.max(formal_errors_map[:, 2:]), cmap='viridis')

    axs[1].imshow(formal_errors_map[:, 3].reshape((len(total_number_of_observations), 9)), aspect='auto',
                  vmin=np.min(formal_errors_map[:, 2:]), vmax=np.max(formal_errors_map[:, 2:]), cmap='viridis')

    axs[2].imshow(formal_errors_map[:, 4].reshape((len(total_number_of_observations), 9)), aspect='auto',
                  vmin=np.min(formal_errors_map[:, 2:]), vmax=np.max(formal_errors_map[:, 2:]), cmap='viridis')

    cbar = fig.colorbar(plt1, ax=axs, shrink=0.95, location='bottom', pad=0.035, aspect=60)
    cbar.set_label(r'Formal Errors in $X_{COF}$ [km]', labelpad=10)

    axs[0].invert_yaxis()
    axs[0].xaxis.set_ticks(np.arange(0, 9, 1))
    x_label_list = true_to_formal_error_ratio_comprehensive
    axs[0].set_xticklabels(x_label_list)
    axs[0].yaxis.set_ticks(np.arange(0, len(total_number_of_observations), 1))
    axs[0].set_yticklabels(total_number_of_observations)

    axs[0].set_title('Radial')
    axs[1].set_title('Along-Track')
    axs[2].set_title('Normal')

    axs[0].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')
    axs[0].set_ylabel(r'Number of Observations [-]')
    axs[1].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')
    axs[2].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')

    plt.savefig(os.path.join(image_save_path, 'uncertainty_variation.png'))

    fig, axs = plt.subplots(1, 3, figsize=(16, 6.8), sharex=True, sharey=True, constrained_layout=True)

    plt1 = axs[0].imshow(formal_errors_rsw_map[:, 2].reshape((len(total_number_of_observations), 9)), aspect='auto',
                         vmin=np.min(formal_errors_rsw_map[:, 2:]), vmax=np.max(formal_errors_rsw_map[:, 2:]),
                         cmap='viridis')

    axs[1].imshow(formal_errors_rsw_map[:, 3].reshape((len(total_number_of_observations), 9)), aspect='auto',
                  vmin=np.min(formal_errors_rsw_map[:, 2:]), vmax=np.max(formal_errors_rsw_map[:, 2:]), cmap='viridis')

    axs[2].imshow(formal_errors_rsw_map[:, 4].reshape((len(total_number_of_observations), 9)), aspect='auto',
                  vmin=np.min(formal_errors_rsw_map[:, 2:]), vmax=np.max(formal_errors_rsw_map[:, 2:]), cmap='viridis')

    cbar = fig.colorbar(plt1, ax=axs, shrink=0.95, location='bottom', pad=0.035, aspect=60)
    cbar.set_label(r'Formal Errors in $X_{Io}$ [km]', labelpad=10)

    axs[0].invert_yaxis()
    axs[0].xaxis.set_ticks(np.arange(0, 9, 1))
    x_label_list = true_to_formal_error_ratio_comprehensive
    axs[0].set_xticklabels(x_label_list)
    axs[0].yaxis.set_ticks(np.arange(0, len(total_number_of_observations), 1))
    axs[0].set_yticklabels(total_number_of_observations)

    axs[0].set_title('Radial')
    axs[1].set_title('Along-Track')
    axs[2].set_title('Normal')

    axs[0].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')
    axs[0].set_ylabel(r'Number of Observations [-]')
    axs[1].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')
    axs[2].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')

    plt.savefig(os.path.join(image_save_path, 'uncertainty_variation_position_of_io.png'))

    fig, axs = plt.subplots(1, 3, figsize=(16, 6.8), sharex=True, sharey=True, constrained_layout=True)

    plt1 = axs[0].imshow(c_q[:, 2].reshape((len(total_number_of_observations), 9)), aspect='auto',
                         vmin=0.0, vmax=1.0, cmap='viridis')

    axs[1].imshow(c_q[:, 3].reshape((len(total_number_of_observations), 9)), aspect='auto',
                  vmin=0.0, vmax=1.0, cmap='viridis')

    axs[2].imshow(c_q[:, 4].reshape((len(total_number_of_observations), 9)), aspect='auto',
                  vmin=0.0, vmax=1.0, cmap='viridis')

    cbar = fig.colorbar(plt1, ax=axs, shrink=0.95, location='bottom', pad=0.035, aspect=60)
    cbar.set_label(r'Contribution of Observations to the Orbital Solution of Io [-]', labelpad=10)

    axs[0].invert_yaxis()
    axs[0].xaxis.set_ticks(np.arange(0, 9, 1))
    x_label_list = true_to_formal_error_ratio_comprehensive
    axs[0].set_xticklabels(x_label_list)
    axs[0].yaxis.set_ticks(np.arange(0, len(total_number_of_observations), 1))
    axs[0].set_yticklabels(total_number_of_observations)

    axs[0].set_title('Radial')
    axs[1].set_title('Along-Track')
    axs[2].set_title('Normal')

    axs[0].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')
    axs[0].set_ylabel(r'Number of Observations [-]')
    axs[1].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')
    axs[2].set_xlabel(r'RS True-to-Formal-Error Ratio [-]')

    plt.savefig(os.path.join(image_save_path, 'uncertainty_variation_rely_on_observations.png'))

    fig, axs = plt.subplots(2, 1, figsize=(6.75, 14.0), sharex=False, sharey=True, constrained_layout=True)
    label_list = \
        [r'$x_{I}$', r'$v_{I}$', r'$x_{E}$', r'$v_{E}$', r'$x_{G}$', r'$v_{G}$', r'$x_{C}$', r'$v_{C}$', r'$x_{COF}$']

    correlation_matrix_txt = correlation_matrix_map[10][0]
    plt1 = axs[0].imshow(np.round(abs(correlation_matrix_txt), 1), aspect='auto',
                         vmin=0.0, vmax=1.0, cmap='viridis', alpha=0.95)

    axs[0].xaxis.set_ticks([1, 4, 7, 10, 13, 16, 19, 22, 25])
    axs[0].set_xticklabels(label_list)
    axs[0].yaxis.set_ticks([1, 4, 7, 10, 13, 16, 19, 22, 25])
    axs[0].set_yticklabels(label_list)
    axs[0].set_title('10 - True-to-Formal-Error (1)')

    correlation_matrix_txt = correlation_matrix_map[1280][6]
    axs[1].imshow(np.round(abs(correlation_matrix_txt), 1), aspect='auto',
                  vmin=0.0, vmax=1.0, cmap='viridis', alpha=0.95)

    axs[1].xaxis.set_ticks([1, 4, 7, 10, 13, 16, 19, 22, 25])
    axs[1].set_xticklabels(label_list)
    axs[1].set_title('1280 - True-to-Formal-Error (19)')

    fig.colorbar(plt1, ax=axs, shrink=0.90, location='bottom', pad=0.01, aspect=30)

    plt.savefig(os.path.join(image_save_path, 'uncertainty_variation_correlations.png'))


plot_uncertainty_variation_results([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120])
