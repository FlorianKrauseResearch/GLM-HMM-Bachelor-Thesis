import numpy as np
import re
from sklearn.model_selection import KFold

import hmm_utils_generic as utils

cols = ['#ff7f00', '#4daf4a', '#377eb8', '#e41a1c', '#984ea3', '#f781bf', '#999999']

files = [r'/path/to/behavioral_data/csv']
pupil_files = [r'/path/to/h5']

# Potential To-Do: Create workflow for identifying states --> creating this mapping dictionary
# Potential To-Do: Function to double-check that the determined best_num_states is same as states defined in this dict

# Set the parameters of the GLM-HMM
num_states = [2, 3, 4, 5, 6]  # number of discrete states
obs_dim = 1  # number of observed dimensions
num_categories = 3  # number of categories for output
input_dim = 2  # input dimensions
n_random_starts = 100  # number of random initializations to maximize chances of finding global optimum
kf = KFold(n_splits=5)  # 5-fold cross-validation


for idx, filename in enumerate(files):
    # identifying the animal and date of the session based on the file name convention analogous to "...3p001_20241109"
    match = re.search(r"3p00[1-9]_\d{8}", filename)
    mice_and_date = match.group()

    # reading in the behavioural data
    data = utils.read_in_data(filename)
    data = utils.format_input_data(data)

    # identifying the number of trials
    num_trials = len(data['Decision'])

    # creating the input sequence to train the HMM
    # the input format needs to be quite specific, hence the use of [:, :, 0] and list()
    input_sequence = np.ones((1, num_trials, input_dim))
    input_sequence[:, :, 0] = data['Direction']
    input_sequence = list(input_sequence)

    # creating the true choices sequence to train the HMM, true choices being the behavioural decisions by the animal
    true_choices = [np.array(data['Decision'], dtype=int).reshape(-1, 1)]

    # running the GLM-HMM and returning the best number of states and the best model based on log likelihood
    best_num_states, best_model = utils.return_best_model(num_states, n_random_starts, kf, true_choices, input_sequence,
                                                          obs_dim, input_dim, num_categories)

    # Return the posterior probabilities for each state at each trial, transform the data format and add the mapped states
    posterior_probs = utils.get_expected_states(best_model, input_sequence, true_choices)
    posterior_probs = utils.transform_posterior_probs_format(posterior_probs)
    posterior_probs = utils.map_states_to_posterior_probs(posterior_probs, best_num_states)

    # Create the sequence of most probable states
    state_sequence = utils.create_state_sequence(data, best_num_states, posterior_probs)

    # Adding the most-probable-state-sequence to the hmm data
    hmm_data = utils.add_state_sequence_to_hmm_data(data, state_sequence)

    # Plot the probabilities of each state for each trial
    utils.plot_state_probabilities(best_num_states, posterior_probs, hmm_data, cols, mice_and_date)

    # Begin analysing the pupil dilation data by reading in the h5 file generated from DeepLabCut and determining the
    # total amount of frames saved from the video recording of the animal.
    pupil_file = pupil_files[idx]
    pupil_source_data = utils.read_in_h5(pupil_file)
    total_frames = len(pupil_source_data)

    # Calculating the pupil diameter for each frame based on 4 points labelled in DeepLabCut
    pupil_diameter_df = utils.calculate_pupil_diameter(pupil_source_data)

    # Add the timestamp for each frame in minutes and seconds since showing the first stimulus and aligning the
    # behavioural dataset with the pupil dataset so that the pupil data and behavioural responses are synchronized
    # for each trial.
    pupil_diameter_df = utils.add_timestamps(pupil_diameter_df, 30, total_frames)
    pupil_diameter_df = utils.sync_pupil_and_hmm_trial_data(pupil_diameter_df, hmm_data)

    # Collect basic descriptive stats about the pupil diameter data before and after outlier removal.
    # Outlier removal is based on IQR. Descriptive stats are saved to an Excel.
    # Potential To-Do: Write a function for this or find any way so that this isn't a total of 14 lines just to write
    # down the descriptive stats.
    before_outlier_removal_data = {
        'Trials': str(len(pupil_diameter_df)),
        'Min Diameter': pupil_diameter_df['Diameter'].min(),
        'Max Diameter': pupil_diameter_df['Diameter'].max(),
        'Avg Diameter': pupil_diameter_df['Diameter'].mean()
    }
    pupil_diameter_df, hmm_df = utils.drop_outliers(pupil_diameter_df, hmm_data)
    after_outlier_removal_data = {
        'Trials': str(len(pupil_diameter_df)),
        'Min Diameter': pupil_diameter_df['Diameter'].min(),
        'Max Diameter': pupil_diameter_df['Diameter'].max(),
        'Avg Diameter': pupil_diameter_df['Diameter'].mean()
    }
    utils.save_outlier_removal_data(before_outlier_removal_data, after_outlier_removal_data, mice_and_date)

    # Create dataframe for state, p(State) and diameter and add a column to the df with the relative diameter per trial.
    state_pupil_df = utils.create_state_pupil_diameter_data(hmm_df, best_num_states, posterior_probs, pupil_diameter_df)
    state_pupil_df = utils.create_relative_diameter_df(state_pupil_df)

    # Save Excel with all the metrics for state identification and plot the p(state) and relative diameter.
    # Potential To-Do: Remove the "plot_correlation_diameter_state" function altogether.

    utils.create_state_identifier_excel(hmm_df, best_num_states, mice_and_date)
    utils.plot_binned_correlation_diameter_state(best_num_states, state_pupil_df, 5, cols, mice_and_date)
    # utils.plot_correlation_diameter_state(best_num_states, state_pupil_df, mice_and_date)
