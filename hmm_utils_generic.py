import pandas as pd
import numpy as np
import numpy.random as npr
import ssm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


# reading in behavioural data from csv in the following structure:
# "Direction"           - Direction of the stimulus presented (1 = left, 2 = right), in the actual experiment values
#                         from 1-10 exist but are disregarded for the thesis
# "Correct"             - whether the response of the animal was correct (0 = incorrect, 1 = correct)
# "ReactionTimeLeft"    - Time of lick detected in seconds after signal onset, NaN if no lick was detected on left side
# "ReactionTimeRight"   - Time of lick detected in seconds after signal onset, NaN if no lick was detected on right side
# "StartTime"           - Start time of the trial in seconds after the first stimulus of the session was shown
def read_in_data(filename):
    data = pd.read_csv(filename, header=None, names=['Direction', 'Correct', 'ReactionTimeLeft', 'ReactionTimeRight',
                                                     'StartTime'])
    data = data[data['StartTime'].notna()]       # included because in some files there is an empty entry as last trial
    return data


# Formatting the "Correct" column to make sure it's understood as integer
# Mapping the "ReactionTimeLeft/Right" responses into a "Decision" column where
# "Decision" - 0 = No lick, 1 = Left Lick, 2 = Right Lick
# Mapping the gradual left/right position of the stimulus presented to either left or right side as gradual positions
# are outside the scope of the thesis. Odd numbers (1 to 9) are left side, Even numbers (2 to 10) are right side.
def format_input_data(data):
    data['Correct'] = data['Correct'].astype(int)
    data['Decision'] = 0
    for i in range(len(data)):
        if np.isnan(data['ReactionTimeLeft'][i]) and np.isnan(data['ReactionTimeRight'][i]):
            data.loc[i, 'Decision'] = 0
        if not np.isnan(data['ReactionTimeLeft'][i]) and np.isnan(data['ReactionTimeRight'][i]):
            data.loc[i, 'Decision'] = 1
        if np.isnan(data['ReactionTimeLeft'][i]) and not np.isnan(data['ReactionTimeRight'][i]):
            data.loc[i, 'Decision'] = 2

    # mapping the left/right direction values
    direction_map = {1: 0, 3: 0, 5: 0, 7: 0, 9: 0, 2: 1, 4: 1, 6: 1, 8: 1, 10: 1}
    data['Direction'] = data['Direction'].map(direction_map)

    if data['Direction'].isnull().any():
        print("CHECK DIRECTION INPUT: Invalid values detected.")
    return data


# Creating multiple GLM-HMMs in three loops:
# outside loop: Models are created to fit every number of states defined earlier, the data inspected showed 2 - 4 to be
#               the expected range, but I'm usually running it for 2 - 6 states to not miss out on more.
# middle loop: A seed is set for a pre-defined number of times (n_random_starts). Usually running it for 100 to strike
#              a balance between computational performance and allowing for many random initializations. As the initial
#              parameters of the GLM-HMM are set randomly, a new seed will change these random values. This is done as
#              we're not making informed assumptions about the starting parameters of the model.
# inside loop: Cross validation is used. From experience this has a huge effect on reducing overfitting. Before using
#              cross validation the amount of states was usually 2-3 higher and not give meaningful insight anymore.
# All models are currently evaluated based on log-likelihood (LL) and the highest average likelihood model of one
# cross validation run is chosen as the best model.
# n_iters = Maximum number of EM iterations, fitting will stop earlier if increase in LL is below the tolerance
# specified by the tolerance parameter. Here the values are hardcoded based on what was used in previous work.
# Potential To-Do: make n_iters an input variable

def return_best_model(num_states, n_random_starts, kf, true_choices, inputs, obs_dim, input_dim, num_categories):
    best_num_states = None
    best_model = None
    best_val_ll = -np.inf
    for K in num_states:
        for seed in range(n_random_starts):
            npr.seed(seed)
            val_lls = []
            for train_index, val_index in kf.split(true_choices[0]):
                # Create training and validation splits
                train_inputs = [inputs[0][train_index]]
                val_inputs = [inputs[0][val_index]]
                train_choices = [true_choices[0][train_index]]
                val_choices = [true_choices[0][val_index]]
                n_iters = 200

                # Fit the model
                glm_hmm = ssm.HMM(K, obs_dim, input_dim, observations="input_driven_obs",
                                  observation_kwargs=dict(C=num_categories), transitions="standard")
                glm_hmm.fit(train_choices, inputs=train_inputs, method="em", num_em_iters=n_iters, tolerance=1e-4)

                # Evaluate log-likelihood on validation data
                val_ll = glm_hmm.log_likelihood(val_choices, inputs=val_inputs)
                val_lls.append(val_ll)

            # Average validation log-likelihood across folds
            avg_val_ll = np.mean(val_lls)
            if avg_val_ll > best_val_ll:
                best_val_ll = avg_val_ll
                best_num_states = K
                best_model = glm_hmm
    return best_num_states, best_model


# return the posterior probabilities of each state for each trial based on the "expected_states" function
def get_expected_states(best_model, inputs, true_choices):
    posterior_probs = [best_model.expected_states(data=data, input=inpt)[0]
                       for data, inpt
                       in zip(true_choices, inputs)]
    return posterior_probs


# The posterior probabilities data format is very unusual and complicated to work with. This function transforms it into
# a nested list.
def transform_posterior_probs_format(posterior_probs):
    posterior_probs_list = []
    for trial in range(len(posterior_probs[0])):
        posterior_probs_list.append(posterior_probs[0][trial])
    posterior_probs_list = [list(arr) for arr in posterior_probs_list]
    return posterior_probs_list


def map_states_to_posterior_probs(posterior_probs, best_num_states):
    states_probs_df = pd.DataFrame()
    trial_list = []
    state_list = []
    state_probs_list = []
    for trial in range(len(posterior_probs)):
        for state in range(best_num_states):
            trial_list.append(trial)
            state_list.append(state)
            state_probs_list.append(posterior_probs[trial][state])
    states_probs_df['Trial'] = trial_list
    states_probs_df['State'] = state_list
    states_probs_df['StateProbability'] = state_probs_list
    return states_probs_df


# Determining which state is the most likely for each trial and writing them into a sequence.
# If the probability is lower than 0.8 it will be assigned state 99 to signal that for that given trial no state is
# sufficiently likely.
# Potential To-Do: make the threshold of 0.8 an input variable
def create_state_sequence(data, best_num_states, posterior_probs):
    state_sequence = []
    for trial in range(len(data)):
        highest_prob = 0
        current_state = 99
        for state in range(best_num_states):
            current_prob = posterior_probs[
                (posterior_probs["Trial"] == trial) & (posterior_probs["State"] == state)]['StateProbability'].iloc[0]
            if current_prob > highest_prob:
                if current_prob > 0.8:
                    highest_prob = current_prob
                    current_state = state
                else:
                    current_state = 99
        state_sequence.append(current_state)
    return state_sequence


# Adding the state-sequence to the hmm dataframe
def add_state_sequence_to_hmm_data(data, state_sequence):
    data['State'] = state_sequence
    return data


# Generating a figure containing a plot for each states likelihood at each given trial.
# Scatter plots are also added on top of the graph to represent hits, errors and no responses.
# Potential To-Do: Allow specification of target folder to save the file to.
def plot_state_probabilities(best_num_states, posterior_probs, data, cols, mice_and_date):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=80)

    # Plot state probabilities
    for state in posterior_probs['State'].unique():
        state_data = posterior_probs[posterior_probs["State"] == state]
        ax.plot(state_data["Trial"], state_data["StateProbability"], label=state + 1, lw=2, color=cols[state])

    ax.set_ylim(-0.01, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Time (Trials)", fontsize=15)
    ax.set_ylabel("p(State)", fontsize=15)
    ax.axhline(0.8, color="black", linestyle="--", label="Threshold")  # Add threshold line

    # Add scatter plots for hits, errors, and no responses
    hits = np.where(data['Correct'] == 1)[0]
    errors = np.where(data['Correct'] == 0)[0]
    no_responses = np.where(data['Decision'] == 0)[0]

    ax.scatter(hits, np.ones(len(hits)) * 1.09, c="black", marker="o", label="Hit", s=20)
    ax.scatter(errors, np.ones(len(errors)) * 1.06, c="red", marker="o", label="Error", s=20)
    ax.scatter(no_responses, np.ones(len(no_responses)) * 1.03, c="gray", marker="o", label="No Resp.", s=20)

    # Add legend
    ax.legend(loc="lower left", fontsize=10)

    # Title and layout
    plt.title("State Probabilities with Behavioral Responses", fontsize=16)
    plt.tight_layout()
    plt.savefig(mice_and_date + '.png')
    plt.close()


# Read in the pupil data generated by DeepLabCut. The dataframe has an unnecessary level which is removed here.
def read_in_h5(filename):
    h5_df = pd.read_hdf(filename)
    # noinspection PyUnresolvedReferences
    h5_df = h5_df.droplevel(0, axis=1)
    return h5_df


# Calculate the pupil diameter for each frame based on the 4 coordinates (North, South, West, East) that were labelled
# and trained in the DeepLabCut model. Based on these 4 coordinates a circle can be estimated that serves as diameter.
# The likelihoods are also passed on but are not used.
# Potential To-Do: Remove or utilize likelihoods.
def calculate_pupil_diameter(pupil_coordinates):
    # Accessing x and y data for EyeNorth
    eye_north_x = pupil_coordinates['EyeNorth']['x']
    eye_north_y = pupil_coordinates['EyeNorth']['y']
    eye_north_likelihood = pupil_coordinates['EyeNorth']['likelihood']

    # Accessing x and y data for EyeSouth
    eye_south_x = pupil_coordinates['EyeSouth']['x']
    eye_south_y = pupil_coordinates['EyeSouth']['y']
    eye_south_likelihood = pupil_coordinates['EyeSouth']['likelihood']

    # Accessing x and y data for EyeWest
    eye_west_x = pupil_coordinates['EyeWest']['x']
    eye_west_y = pupil_coordinates['EyeWest']['y']
    eye_west_likelihood = pupil_coordinates['EyeWest']['likelihood']

    # Accessing x and y data for EyeEast
    eye_east_x = pupil_coordinates['EyeEast']['x']
    eye_east_y = pupil_coordinates['EyeEast']['y']
    eye_east_likelihood = pupil_coordinates['EyeEast']['likelihood']

    # Calculating pupil diameter and writing it in a list
    diameter_list = []
    for i in range(len(pupil_coordinates.index)):
        d = (np.sqrt((eye_north_x[i] - eye_south_x[i]) ** 2 + (eye_north_y[i] - eye_south_y[i]) ** 2) +
             np.sqrt((eye_west_x[i] - eye_east_x[i]) ** 2 + (eye_west_y[i] - eye_east_y[i]) ** 2)) / 2
        diameter_list.append(d)

    # Adding the diameter list to the original Dataframe
    pupil_coordinates['EyeTotal', 'Diameter'] = diameter_list
    return pupil_coordinates


# Timestamps in minutes and seconds are added to the pupil data based on the frame rate and length of video.
def add_timestamps(data_df, frame_rate, total_frames):
    timestamp_sec = []
    timestamp_min = []
    for frames in range(total_frames):
        timestamp_sec.append(frames/frame_rate)
        timestamp_min.append(frames/frame_rate/60)
    data_df['Timestamp', 'Seconds'] = timestamp_sec
    data_df['Timestamp', 'Minutes'] = timestamp_min
    return data_df


# Determine when the first trial begins based on the first time the LED turns on, based on the likelihood of the LED
# being recognized in DeepLabCut.
def sync_pupil_and_hmm_trial_data(pupil_data, hmm_data):
    for i in range(1, len(pupil_data)):
        # Get current and previous likelihood values using iloc (position-based)
        current_likelihood = pupil_data.iloc[i][('LED', 'likelihood')]
        previous_likelihood = pupil_data.iloc[i - 1][('LED', 'likelihood')]

        # Condition 1: If current likelihood > 0.65 and previous likelihood <= 0.65 (Start condition)
        if current_likelihood > 0.65 >= previous_likelihood:
            start_time = pupil_data.iloc[i][('Timestamp', 'Seconds')]
            break

    # determine time diff in time between start of each trial
    diffs = []
    for starts in range(1, len(hmm_data['StartTime'])):
        diffs.append(hmm_data['StartTime'][starts] - hmm_data['StartTime'][starts - 1])

    # calculate the index of the starting time and the respective diameter in the pupil video and add it to a list
    first_timestamp = (pupil_data['Timestamp']['Seconds'] - start_time).abs().idxmin()
    first_diameter = pupil_data.loc[first_timestamp, ('EyeTotal', 'Diameter')]
    diameters = [first_diameter]
    time = start_time

    # find all the diameters that correspond to the actual trial times in the video, but based on the trial times from
    # the behavioural data file with the help of the diffs list
    for n in range(len(diffs)):
        time = time + diffs[n]
        closest_timestamp = (pupil_data['Timestamp']['Seconds'] - time).abs().idxmin()
        closest_diameter = pupil_data.loc[closest_timestamp, ('EyeTotal', 'Diameter')]
        diameters.append(closest_diameter)

    # Convert the trial_data list to a DataFrame
    hmm_data['Diameter'] = diameters

    return hmm_data


# Outlier removal of Pupils based on quartiles because it can't be assumed that the diameters are normally distributed.
# Also removing trials where not a single state has a p(State) > 0.8.
def drop_outliers(pupil_df, hmm_df):
    q1 = pupil_df['Diameter'].quantile(0.25)
    q3 = pupil_df['Diameter'].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    rows_to_drop_pupil = pupil_df[(pupil_df['Diameter'] < lower_bound) | (pupil_df['Diameter'] > upper_bound)].index
    pupil_df = pupil_df[(pupil_df['Diameter'] >= lower_bound) & (pupil_df['Diameter'] <= upper_bound)]
    hmm_df = hmm_df.drop(rows_to_drop_pupil)

    rows_to_drop_states = hmm_df[(hmm_df['State'] == 99)].index
    pupil_df = pupil_df.drop(rows_to_drop_states)
    hmm_df = hmm_df.drop(rows_to_drop_states)

    # resetting the indices because dropping rows leaves gaps in the df indices and would cause issues later
    pupil_df = pupil_df.reset_index(drop=True)
    hmm_df = hmm_df.reset_index(drop=True)

    return pupil_df, hmm_df


# Save some descriptive data about the pupil diameter dataset before and after outlier removal.
# Potential To-Do: Allow specification of target folder to save the file to.
def save_outlier_removal_data(prior_data, after_data, mice_and_date):
    outlier_removal_data = pd.DataFrame([prior_data, after_data], index=['Before', 'After'])
    outlier_removal_data.to_excel('outlier_removal_data_' + mice_and_date + '.xlsx')


# Create a new dataframe to store the current state, the probability of that state and the diameter per trial.
# This serves as the basis for plotting this relationship.
def create_state_pupil_diameter_data(hmm_data, best_num_states, posterior_probs, diameter_df):
    rows = []
    for trial in range(len(hmm_data)):
        for state in range(best_num_states):
            current_prob = posterior_probs[
                (posterior_probs["Trial"] == trial) & (posterior_probs["State"] == state)]['StateProbability'].iloc[0]
            pupil_diameter = diameter_df['Diameter'][trial]
            rows.append({
                'State': state,
                'StateProbability': current_prob,
                'PupilDiameter': pupil_diameter
            })
    state_pupil_data = pd.DataFrame(rows)

    return state_pupil_data


# Add a column to the pupil-state df that calculates the relative pupil dilation for each trial.
# Relative refers to each dilation being saved as a % of to the maximum dilation over the entire session.
# Potential To-Do: Merge this function with the "create_state_pupil_diameter_data" function.
def create_relative_diameter_df(state_pupil_df):
    max_pupil_diameter = state_pupil_df['PupilDiameter'].max()
    state_pupil_df['RelPupilDiameter'] = (state_pupil_df['PupilDiameter'] / max_pupil_diameter) * 100

    return state_pupil_df


# Create all the metrics for identifying the meaning of a given state and saving these metrics in Excel.
# "Lick Bias" is taking into account the amount of left/right stimuli and not only left/right responses. So it reflects
# actual bias and not just a dis-balance in the presentation of left/right stimuli.
# Potential To-Do: There has to be a way to make this code slimmer while maintaining functionality. Running the logic
# once for each state and then separately again for the entire thing doesn't look optimal to me.
# Potential To-Do: Allow specification of target folder to save the file to.
def create_state_identifier_excel(data, best_num_states, mice_and_date):
    states_performance_df = pd.DataFrame()
    states_performance_df['State'] = data['State']
    states_performance_df['Decision'] = data['Decision']
    states_performance_df['Direction'] = data['Direction']
    states_performance_df['Correct'] = data['Correct']

    result_dict = {}
    # Process the data for each state
    for state in range(best_num_states):
        state_data = states_performance_df[states_performance_df['State'] == state]
        total_trials = len(state_data)

        # Calculate misses rate
        misses = state_data[state_data['Decision'] == 0]
        misses_rate = len(misses) / total_trials if total_trials > 0 else 0

        # Calculate hit rates
        hit_rate = state_data['Correct'].sum() / total_trials if total_trials > 0 else 0
        left_hits = state_data[(state_data['Direction'] == 0) & (state_data['Correct'] == 1)]
        right_hits = state_data[(state_data['Direction'] == 1) & (state_data['Correct'] == 1)]
        left_hit_rate = len(left_hits) / len(state_data[state_data['Direction'] == 0]) if len(
            state_data[state_data['Direction'] == 0]) > 0 else 0
        right_hit_rate = len(right_hits) / len(state_data[state_data['Direction'] == 1]) if len(
            state_data[state_data['Direction'] == 1]) > 0 else 0

        # Calculate error rates
        errors = state_data[(state_data['Correct'] == 0) & (state_data['Decision'] != 0)]
        error_rate = len(errors) / total_trials if total_trials > 0 else 0
        left_errors = state_data[(state_data['Direction'] == 0) & (state_data['Correct'] == 0)]
        right_errors = state_data[(state_data['Direction'] == 1) & (state_data['Correct'] == 0)]
        left_error_rate = len(left_errors) / len(state_data[state_data['Direction'] == 0]) if len(
            state_data[state_data['Direction'] == 0]) > 0 else 0
        right_error_rate = len(right_errors) / len(state_data[state_data['Direction'] == 1]) if len(
            state_data[state_data['Direction'] == 1]) > 0 else 0

        # Calculate Lick Bias
        amount_left_trials = len(state_data[state_data['Direction'] == 0])
        amount_right_trials = len(state_data[state_data['Direction'] == 1])
        amount_left_licks = len(state_data[state_data['Decision'] == 1])
        amount_right_licks = len(state_data[state_data['Decision'] == 2])

        if amount_left_trials > 0:
            normalized_left_licks = amount_left_licks / amount_left_trials
        else:
            normalized_left_licks = 0

        if amount_right_trials > 0:
            normalized_right_licks = amount_right_licks / amount_right_trials
        else:
            normalized_right_licks = 0

        if normalized_right_licks > 0:
            lick_bias = normalized_left_licks / normalized_right_licks
        else:
            lick_bias = 999

        # Store in dictionary
        result_dict[state + 1] = {
            "Amount Left Trials": amount_left_trials,
            "Amount Right Trials": amount_right_trials,
            "Amount Left Licks": amount_left_licks,
            "Amount Right Licks": amount_right_licks,
            "Amount Misses": len(misses),
            "Lick Bias (left over right)": lick_bias,
            "Hit Rate": hit_rate,
            "Left Hit Rate": left_hit_rate,
            "Right Hit Rate": right_hit_rate,
            "Error Rate": error_rate,
            "Left Error Rate": left_error_rate,
            "Right Error Rate": right_error_rate,
            "Misses Rate": misses_rate
        }

    total_trials = len(states_performance_df['Decision'])

    # Calculate Lick Bias for the Entire Dataset
    amount_left_trials = len(states_performance_df[states_performance_df['Direction'] == 0])
    amount_right_trials = len(states_performance_df[states_performance_df['Direction'] == 1])
    amount_left_licks = len(states_performance_df[states_performance_df['Decision'] == 1])
    amount_right_licks = len(states_performance_df[states_performance_df['Decision'] == 2])

    if amount_left_trials > 0:
        normalized_left_licks = amount_left_licks / amount_left_trials
    else:
        normalized_left_licks = 0

    if amount_right_trials > 0:
        normalized_right_licks = amount_right_licks / amount_right_trials
    else:
        normalized_right_licks = 0

    if normalized_right_licks > 0:
        lick_bias = normalized_left_licks / normalized_right_licks
    else:
        lick_bias = 999

    # First calculate most relevant data for entire dataset
    result_dict['Total'] = {
        'Hit Rate': states_performance_df['Correct'].sum() / total_trials,
        'Error Rate': len(states_performance_df[(states_performance_df['Correct'] == 0) & (
                    states_performance_df['Decision'] != 0)]) / total_trials,
        'Misses Rate': len(states_performance_df[states_performance_df['Decision'] == 0]) / total_trials,
        'Lick Bias (left over right)': lick_bias
    }

    state_performance_df = pd.DataFrame.from_dict(result_dict, orient='index')
    # Save to Excel
    state_performance_df.to_excel('states_performance_' + mice_and_date + '.xlsx', index_label='State')


# Create bins for the relative pupil diameter and plot the average value per bin against the probability of each state
# per trial. A correlation is calculated between each state and the pupil diameter and the correlation line is plotted
# in the figure.
def plot_binned_correlation_diameter_state(best_num_states, state_pupil_df, bin_width, cols, mice_and_date):
    for state in range(best_num_states):
        # Filter the data for the current state
        state_data = state_pupil_df[state_pupil_df['State'] == state]

        # Create bins for RelPupilDiameter
        state_data['DiameterBin'] = pd.cut(state_data['RelPupilDiameter'],
                                           bins=np.arange(0, state_data['RelPupilDiameter'].max() + bin_width,
                                                          bin_width),
                                           labels=np.arange(0, state_data['RelPupilDiameter'].max(), bin_width))
        binned_data = state_data.groupby('DiameterBin').agg({
            'RelPupilDiameter': 'mean',  # Use the mean RelPupilDiameter for plotting
            'StateProbability': 'mean'  # Use the mean StateProbability for plotting
        }).reset_index()
        plt.scatter(binned_data['RelPupilDiameter'], binned_data['StateProbability'],
                    label=f'State = {state + 1}', marker='o', color=cols[state])
        binned_data = binned_data.dropna()
        plt.plot(binned_data['RelPupilDiameter'], binned_data['StateProbability'], color=cols[state], linestyle='-')
        model = LinearRegression()
        x = binned_data['RelPupilDiameter'].values.reshape(-1, 1)
        y = binned_data['StateProbability'].values
        model.fit(x, y)
        x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        plt.plot(x_line, y_line, linestyle='--', color=cols[state], label=f'Fit: State = {state + 1}')

        # Calculate and display correlation
        corr, _ = pearsonr(binned_data['RelPupilDiameter'], binned_data['StateProbability'])
        plt.text(0.05, 0.95 - 0.05 * state, f'Corr (State = {state + 1}): {corr:.2f}',
                 transform=plt.gca().transAxes, fontsize=15, color='black')

    plt.xlabel('Relative Pupil Diameter (%)')
    plt.ylabel('State Probability')
    plt.title('State Probability vs Relative Pupil Diameter')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(mice_and_date + '_binned_' + str(bin_width) + '.png', bbox_inches='tight')
    plt.close()


# deprecated as of now, doesn't actually give much meaningful data --> binned approach is preferred
def plot_correlation_diameter_state(best_num_states, state_pupil_df, mice_and_date):
    # Create the Pupil/States plot
    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Define colors for up to 6 states
    if best_num_states > len(colors):
        raise ValueError("Increase the number of colors to match best_num_states.")

    for state in range(best_num_states):
        # Filter the data for the current state
        state_data = state_pupil_df[state_pupil_df['State'] == state]
        plt.scatter(state_data['RelPupilDiameter'], state_data['StateProbability'],
                    label=f'State = {state + 1}', marker='o', color=colors[state])

        model = LinearRegression()
        x = state_data['RelPupilDiameter'].values.reshape(-1, 1)
        y = state_data['StateProbability'].values
        model.fit(x, y)
        x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        plt.plot(x_line, y_line, linestyle='--', color=colors[state], label=f'Fit: State = {state}')
        corr, _ = pearsonr(state_data['RelPupilDiameter'], state_data['StateProbability'])
        plt.text(0.05, 0.95 - 0.05 * state, f'Corr (State = {state + 1}): {corr:.2f}',
                 transform=plt.gca().transAxes, fontsize=15, color='black')

    plt.xlabel('Relative Pupil Diameter (%)')
    plt.ylabel('State Probability')
    plt.title('State Probability vs Relative Pupil Diameter')
    plt.legend()
    plt.savefig(mice_and_date + '_correlation.png')
    plt.close()
