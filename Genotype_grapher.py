import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import pingouin as pg
import seaborn as sns
from itertools import product
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import tukey_hsd
from scipy.stats import mannwhitneyu
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import re

def load_data(file_path, info_path):
    ### loads needed dataframes from file paths
    df = pd.read_csv(file_path)
    info_df = pd.read_csv(info_path)

    info_df = info_df.loc[(info_df['task'] != 'Reset')
                          & (info_df['task'] != 'Discrimination')
                          & (info_df['task'] != 'Holding')
                          & (info_df['task'] != 'Tsc2_LE')
                          & (info_df['analysis_type'] != 'Training - Oddball')
                          & (info_df['analysis_type'] != 'Training - Octave')]

    return df, info_df

def file_info(info_df):
    # Splits session file name into component parts for later organization
    df = info_df[['file_name','UUID']]
    file_name_list = list(zip(df['file_name'], df['UUID']))
    file_info_list = [(file_name.split('_'), uuid) for file_name,uuid in file_name_list]
    file_info = [(tup[:-1],tup[-1]) for tup in file_info_list]
    file_uuid_list = [t[1] for t in file_info]
    file_info_df = pd.DataFrame(file_info, columns=['file_info','UUID'])
    merged_df = pd.merge(file_info_df, info_df, on="UUID", how="right")
    file_info_df = merged_df[merged_df['UUID'].isin(file_uuid_list)]

    return file_info_df

def file_type_finder(df):

    # finds training files based on whether or not there is a ncatch in file name
    catch_pat = r'_\dcatch'

    catch_mask = df['file_name'].astype(str).str.contains(catch_pat, regex=True)

    training_uuids = df.loc[catch_mask, 'UUID']
    training_uuids_list = training_uuids.tolist()


    # find length of lock out time
    lo_pat = r'_\d+s\b'

    lo_matches = df['file_name'].apply(
    lambda x: re.search(lo_pat, str(x)).group(0) if re.search(lo_pat, str(x)) else None)

    lo_mask = lo_matches.notna()

    lo_df = pd.DataFrame({
        'UUID': df.loc[lo_mask, 'UUID'],
        'lo_time': lo_matches[lo_mask]
    })
    lo_df['lo_time'] = lo_df['lo_time'].str.extract(r'\b_(\d+)s').astype('Int64')

    lo_dict = dict(zip(lo_df['UUID'], lo_df['lo_time']))
    df['lo_time'] = df['UUID'].map(lo_dict).fillna(0).astype(int)

    # finds the dB interval for each session
    pattern = r'\b\d{2}-\d{2}dB\b'

    matches = df['file_info'].apply(
    lambda tup: next(
        (re.search(pattern, str(x)).group(0) for x in tup if re.search(pattern, str(x))), None))

    mask = matches.notna()

    dB_df = pd.DataFrame({
        'UUID': df.loc[mask, 'UUID'],
        'dB_range': matches[mask]
    })

    dB_df['low_dB'] = dB_df['dB_range'].str.extract(r'(\d{2})').astype(int)

    rxn_uuids = dB_df.loc[dB_df['low_dB'] > 20, 'UUID'].tolist()
    th_uuids = dB_df.loc[dB_df['low_dB'] <= 20, 'UUID'].tolist()

    df['new_task'] = df['UUID'].apply(lambda x: 'Training' if x in training_uuids_list else ('TH' if x in th_uuids else ('Rxn' if x in rxn_uuids else 'training_bad')))

    dB_range_set = sorted(set(dB_df['dB_range']))

    cldftrn = df.loc[(df['new_task'] == 'Training')]
    cldfrxn = df.loc[(df['new_task'] == 'Rxn')]
    cldfth = df.loc[(df['new_task'] == 'TH')]

    trnflset = set(cldftrn['UUID'])
    cldfrxnset = set(cldfrxn['UUID'])
    cldfthset = set(cldfth['UUID'])

    num_training = len(trnflset)
    num_rxn = len(cldfrxnset)
    num_th = len(cldfthset)

    return df, dB_df, lo_df, dB_range_set, num_training, num_rxn, num_th

def uuid_comparer(df, info_df):
    ### compares uuid (session ID) between dataframes and makes set of matched uuids
    valid_UUIDs_set = set(info_df['UUID'])
    data_UUIDs_set = set(df['UUID'])
    shared_UUIDs = valid_UUIDs_set.intersection(data_UUIDs_set)
    unshared_UUIDs = valid_UUIDs_set.symmetric_difference(data_UUIDs_set)

    return shared_UUIDs

def round(x):
    # used to find max and min delay times for each file
    return np.round(x * 2) / 2

def data_cleaner(df, info_df, shared_UUIDs, wanted_columns):
    ### filters for only valid data in dataframe and merges other needed dataframes. ie: shared uuids and completed blocks
    shared_UUIDs = list(shared_UUIDs)
    
    merged_df = pd.merge(df, info_df[wanted_columns], on="UUID", how="right")
    clean_df = merged_df[merged_df['UUID'].isin(shared_UUIDs)]
    clean_df = clean_df[clean_df['complete_block_number'] > 1]
    incomplete_blocks_df = clean_df[clean_df['complete_block_number'] == 1]

    cldftrn = clean_df.loc[(clean_df['task'] == 'Training')]
    cldfrxn = clean_df.loc[(clean_df['task'] == 'Rxn')]
    cldfth = clean_df.loc[(clean_df['task'] == 'TH')]

    trnflset = set(cldftrn['UUID'])
    cldfrxnset = set(cldfrxn['UUID'])
    cldfthset = set(cldfth['UUID'])

    num_training = len(trnflset)
    num_rxn = len(cldfrxnset)
    num_th = len(cldfthset)

    return clean_df, num_training, num_rxn, num_th
    
def delay_classifier(df):
    ### finds all delay intervals and each session's interval, also lists out important data set information
    wanted_data = df[['Delay (s)','UUID']]
    delay_max_min_by_UUID = wanted_data.groupby(['UUID']).agg(['max','min'])
    rounded_delays = delay_max_min_by_UUID['Delay (s)'].map(round)
    delay_intervals = rounded_delays.groupby(['max','min']).count()
    delay_interval_list = delay_intervals.index.tolist()
    df_with_classified_delays = pd.merge(df, rounded_delays, on='UUID', how='right')
    df_with_classified_delays = df_with_classified_delays.rename(columns={"max": "Max Delay (s)", "min": "Min Delay (s)"})
    rat_ids = df['rat_ID'].unique()
    dob_df = df[['rat_ID','DOB']].drop_duplicates()
    # dob_list = dob_df.tolist()
    gt_df = df['Genotype'].unique()
    gt_list = gt_df.tolist()
    return delay_interval_list, df_with_classified_delays, rat_ids, dob_df, gt_list



########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

def attempts_over_session(df, delay_int):

    df = df[['Trial_number','UUID','rat_ID','Genotype','Attempts_to_complete']]
    # i need to make a graph that shows attempts to complete across the session. I woould probably best do this by making a line graph for every rat in a dataset with the line designating the average attempts for each trial with error bars to show amount of error which should be pretty large around the front maybe do median so not to get skewed

    data = df.groupby(['rat_ID','Genotype','Trial_number']).agg(
                    attempt_med=('Attempts_to_complete','mean')
            ).reset_index()

    

def weight_var_finder(df):

    df = df[['weight','DOB','UUID','rat_ID','Genotype','Attempts_to_complete']]

    df['one_attempt'] = df['Attempts_to_complete'] == 1
             
    groups = df.groupby(['rat_ID','Genotype']).agg(
                    weight_var=('weight','var'),
                    # mean_weight=('weight','mean'),
                    trials_one_attempt=('one_attempt', 'sum'), 
                    total_trials=('Attempts_to_complete', 'count'))
    
    groups['prop_one_attempt'] = groups['trials_one_attempt'] / groups['total_trials']

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()

    data['weight_stability'] = data['weight_var'].apply(lambda x: 'Stable' if x < 300 else "Unstable")

    # sns.boxplot(x='weight_stability', y='prop_one_attempt', data=data, palette='Set2', width=0.5)
    # sns.swarmplot(x='weight_stability', y='prop_one_attempt', data=data, color='black', size=5)

    # plt.title(f"Differences in Patience Between Rats with Stable and Unstable Weights for Tsc2 Dataset")
    # plt.xlabel('Stability')
    # plt.ylabel("Proportion of Single Attempts")

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/fmr1_le_hit_rate_plots.png')
    # plt.show()

    # stable_data = data.loc[(data['weight_stability'] == 'Stable')]
    # unstable_data = data.loc[(data['weight_stability'] == 'Unstable')]

    # stat, p_value = mannwhitneyu(stable_data['prop_one_attempt'], unstable_data['prop_one_attempt'], alternative='two-sided')
    
    # print(f"Mann-Whitney U statistic: {stat}")
    # print(f"P-value: {p_value}")

    # # Interpretation at alpha = 0.05
    # alpha = 0.05
    # if p_value < alpha:
    #     print("Reject the null hypothesis: distributions differ.")
    # else:
    #     print("Fail to reject the null hypothesis: no significant difference.")


    # Initialize layout
    fig, ax = plt.subplots(figsize=(9, 9))

    # Add scatterplot
    ax.scatter('prop_one_attempt', 'weight_var', data=data, s=60, alpha=0.7, edgecolors="k")

    # Fit linear regression via least squares with numpy.polyfit
    # It returns an slope (b) and intercept (a)
    # deg=1 means linear fit (i.e. polynomial of degree 1)
    b, a = np.polyfit(data['prop_one_attempt'], data['weight_var'], deg=1)

    # Create sequence of 100 numbers from 0 to 100
    xseq = np.linspace(0, 1, num=100)

    # Plot regression line
    ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
    ax.set_title(f"Weight Variance vs Proportion of Single Attempts in Fmr1 Dataset")
    ax.set_ylabel("Weight Variance")
    ax.set_xlabel('Proportion of One Attempt')

    plt.show()

    return data

def training_periods(df, tasks, delay_int):

    df = df[['date','Genotype','rat_ID','task']]

    filtered = df.loc[(df['task'].isin(tasks))
                    # & (df['Max Delay (s)'] == delay_int[0])
                    # & (df['Min Delay (s)'] == delay_int[1])
            ]

    df = pd.DataFrame(filtered)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    rats = df["rat_ID"].unique()
    y_positions = {rat: i for i, rat in enumerate(rats)}

    color_map = {
        "Training": "royalblue",
        "Baseline": "orange",
        "Rxn": "red",
        "TH": "pink"
        }

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(
    y=df["rat_ID"].map(y_positions),
    left=df["date"],
    width=pd.Timedelta(days=1),
    color=df["task"].map(color_map),
    edgecolor="none")

    ax.set_yticks(range(len(rats)))
    ax.set_yticklabels(rats)
    ax.set_ylabel('Rat ID')
    ax.set_xlabel("Date")
    ax.set_title("Behavior Sessions by Rat Over Time for Fmr1 Dataset")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    legend_patches = [
        mpatches.Patch(color=color, label=task)
        for task, color in color_map.items()
        if task in df["task"].unique()   # only show tasks present in data
    ]
    ax.legend(handles=legend_patches, title="Task", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def age_differences(df,tasks,genotype_color,delay_int):
    
    df = df[['rat_ID','Genotype','task','analysis_type','DOB','UUID','date','Attempts_to_complete','Max Delay (s)','Min Delay (s)','Delay (s)']]

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["DOB"] = pd.to_datetime(df["DOB"], format="%Y%m%d")
    
    df['one_attempt'] = df['Attempts_to_complete'] == 1

    tsc2_rats = [156,306,316,157,160,311,315,319] # have more than 12 sessions for each time point
    fmr1_rats = [197,198,210,213,195,196,209,211]

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_int[0])
            & (df['Min Delay (s)'] == delay_int[1])
            & (df['Delay (s)'] > 2.5)
            & (df['task'].isin(tasks))
            # & (df['rat_ID'].isin(fmr1_rats))
            # & (df['analysis_type'].isin(analysis_types))
            ]

    target = filtered['DOB'] + pd.DateOffset(months=4)
    df_4m = filtered[
        (filtered['date'] >= target) &
        (filtered['date'] < target + pd.DateOffset(months=1))
    ]
    df_4m['age'] = '4'
    
    target = filtered['DOB'] + pd.DateOffset(months=8)
    df_8m = filtered[
        (filtered['date'] >= target) &
        (filtered['date'] < target + pd.DateOffset(months=1))
    ]
    df_8m['age'] = '8'

    df_m = pd.concat([df_4m, df_8m], axis=0)

    groups = df_m.groupby(['rat_ID','age','Genotype']).agg(
    num_sessions=('UUID','nunique'),
    trials_one_attempt=('one_attempt', 'sum'),
    total_trials=('Attempts_to_complete', 'count')).reset_index()

    groups['prop_one_attempt'] = groups['trials_one_attempt'] / groups['total_trials']
    
    # data = groups.sort_values(['Genotype', 'rat_ID'])
    # order = data['rat_ID'].tolist()
    
    # sns.barplot(x='rat_ID', y='num_sessions', data=data, hue="Genotype", order=order, palette=genotype_color)
    # plt.title(f"Session Totals per Genotype")
    # plt.ylabel("Number of Sessions")
    # plt.xlabel('Rat/Genotype')
    # plt.xticks(rotation=45)

    # plt.tight_layout()
    # # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/fmr1_le_rat_sessions_totals_plots.png')
    # plt.show()

    plus12_rat_df = groups.loc[
            (groups['num_sessions'] > 12)]

    data = plus12_rat_df.sort_values(by=['age']).reset_index()

    sns.boxplot(x='age', y='prop_one_attempt', data=data, width=0.5)
    sns.swarmplot(x='age', y='prop_one_attempt', data=data, color='red', size=5)

    plt.title(f"Proportion of Single Attempts Over Age")
    plt.ylabel("Proportion of Single Attempts")
    plt.xlabel('Age in Months')

    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/fmr1_age_diffs')
    plt.show()
    
    tukey = pairwise_tukeyhsd(endog=data['prop_one_attempt'], groups=data['age'], alpha=0.05)

    print(tukey)

    return data

def add_significance_bars(ax, group1, group2, p_value, x1, x2, y, significance_level=0.05):
    """Add significance bars to a boxplot."""
    ymax = ax.get_ylim()[1]
    y = ymax * 0.92
    y_offset = ymax * 0.03

    ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], color='black')
    
    # Annotate with asterisks based on p-value
    if p_value < significance_level:
        ax.text((x1 + x2) / 2, y + y_offset * 1.4, '*', fontsize=16, ha='center')
    if p_value < significance_level / 10:
        ax.text((x1 + x2) / 2, y + y_offset * 2.2, '*', fontsize=16, ha='center')
    if p_value < significance_level / 100:
        ax.text((x1 + x2) / 2, y + y_offset * 3.0, '*', fontsize=16, ha='center')

def short_long_comparison_graph(df,delay_interval,tasks,analysis_types):

    df = df[['Delay (s)','Max Delay (s)','Min Delay (s)','Genotype','task','analysis_type','Attempts_to_complete','rat_ID']]

    df['one_attempt'] = df['Attempts_to_complete'] == 1

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['task'].isin(tasks))
            & (df['analysis_type'].isin(analysis_types))
            ]
    
    filtered['delay_int'] = filtered['Delay (s)'].apply(lambda x: "1-1.5" if x < 1.5 else ('1.5-2' if 1.5 <= x < 2 else ('2-2.5' if 2 <= x < 2.5 else ('2.5-3' if 2.5 <= x < 3 else ('3-3.5' if 3 <= x < 3.5 else '3.5-4')))))
    
    groups = filtered.groupby(['Genotype','rat_ID','delay_int','task']).agg(
    trials_one_attempt=('one_attempt','sum'),
    total_trials=('Attempts_to_complete','count')).reset_index()

    groups[f'prop_one_attempt'] = groups[f'trials_one_attempt'] / groups['total_trials']

    data = groups.sort_values(by=['delay_int','Genotype']).reset_index()

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), sharey=True)

    for ax, task in zip(axes, tasks):

        subset = data[data["task"] == task]
        
        sns.boxplot(x="delay_int", y='prop_one_attempt', data=subset, width=0.5, ax=ax)
        # add_significance_bars(ax, 'A', 'B', .004, 0, 1, .95)
        ax.set_title(f'Proportion of 1-Attempt {task} Trials by Delay Interval')
        ax.set_ylabel('Proportion of Trials Completed in One Attempt')
        ax.set_xlabel('Delay Interval (s)')
    
    plt.tight_layout()
    plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/fmr1_le_delay_int_plots.png')
    plt.show()

    # baseline_above_data = data.loc[(data['task'] != 'Training')
    #                                & (data['delay_int'] == "Above")]
    # baseline_below_data = data.loc[(data['task'] != 'Training')
    #                                & (data['delay_int'] == "Below")]
    # training_above_data = data.loc[(data['task'] == 'Training')
    #                                & (data['delay_int'] == "Above")]
    # training_below_data = data.loc[(data['task'] == 'Training')
    #                                & (data['delay_int'] == "Below")]
    
    # stat, p_value = mannwhitneyu(baseline_above_data['prop_one_attempt'], baseline_below_data['prop_one_attempt'], alternative='two-sided')
    
    # stat, p_value = mannwhitneyu(training_above_data['prop_one_attempt'], training_below_data['prop_one_attempt'], alternative='two-sided')
    
    # print(f"Mann-Whitney U statistic: {stat}")
    # print(f"P-value: {p_value}")

    # # Interpretation at alpha = 0.05
    # alpha = 0.05
    # if p_value < alpha:
    #     print("Reject the null hypothesis: distributions differ.")
    # else:
    #     print("Fail to reject the null hypothesis: no significant difference.")



    return data

def attempts_to_delay_corr(df, delay_interval,tasks):

    df = df[['Delay (s)','Max Delay (s)','Min Delay (s)','Genotype','task','Attempts_to_complete','rat_ID']]

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['task'].isin(tasks))
            ]

    plt.scatter(filtered['Delay (s)'], filtered['Attempts_to_complete'])

    # Add labels and a title
    plt.xlabel('Delay (s)')
    plt.ylabel('Attempts')
    plt.title('Attempts vs. Delay')

    # Show the plot
    plt.show()

def file_diff_graph(df, tasks, analysis_types, delay_interval,genotype_color):

    df = df[['Genotype','task','analysis_type','Attempts_to_complete','rat_ID','Delay (s)',"Max Delay (s)","Min Delay (s)"]]

    df['one_attempt'] = df['Attempts_to_complete'] == 1
    df['more_than_one_attempt'] = df['Attempts_to_complete'] > 1

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['Delay (s)'] > 2.5)
            & (df['task'].isin(tasks))
            & (df['analysis_type'].isin(analysis_types))
            ]
    groups = filtered.groupby(['Genotype', 'task']).agg(
    trials_one_attempt=('one_attempt', 'sum'),
    trials_more_than_one_attempt=('more_than_one_attempt', 'sum'),
    total_trials=('Attempts_to_complete', 'count'))

    groups['prop_one_attempt'] = groups['trials_one_attempt'] / groups['total_trials']
    groups['prop_more_than_one'] = groups['trials_more_than_one_attempt'] / groups['total_trials']
    grouped_rats_df = groups.sort_values(by=['Genotype', 'task']).reset_index()
    
    prop_one_attempt_data = grouped_rats_df[['prop_one_attempt','Genotype','task']]
    data = prop_one_attempt_data.sort_values(by=['task','Genotype']).reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='task',
        y='prop_one_attempt',
        hue='Genotype',
        data=data,
        palette=genotype_color,
        edgecolor='black',
        errorbar='se'
    )
    
    plt.ylim(.5, 1)
    plt.title('Proportion of 1-Attempt Trials by Genotype and Task')
    plt.ylabel('Proportion of Trials Completed in One Attempt')
    plt.xlabel('Task')
    plt.legend(title='Genotype', loc='upper left')
    
    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_file_diff_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    tukey = pairwise_tukeyhsd(endog=data['prop_one_attempt'], groups=data['Genotype'], alpha=0.05)
    
    return data, tukey

def rat_session_totals(df,genotype_color,tasks,analysis_types,delay_interval):
    df = df[['rat_ID','UUID','Genotype']]

    filtered = df.loc[
            # (df['task'].isin(tasks))
            # & (df['analysis_type'].isin(analysis_types))
            (df['rat_ID'] != 745)
            # & (df['Max Delay (s)'] == delay_interval[0])
            # & (df['Min Delay (s)'] == delay_interval[1])
                      ]
    
    data = df.groupby(['rat_ID','Genotype']).agg(
                        num_UUIDs = ('UUID','nunique')).reset_index()
    
    data = data.sort_values(['Genotype', 'rat_ID'])
    order = data['rat_ID'].tolist()
    
    sns.barplot(x='rat_ID', y='num_UUIDs', data=data, hue="Genotype", order=order, palette=genotype_color)
    plt.title(f"Session Totals per Genotype")
    plt.ylabel("Number of Sessions")
    plt.xlabel('Rat/Genotype')
    plt.xticks(rotation=45)

    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/fmr1_le_rat_sessions_totals_plots.png')
    plt.show()

    return data
    
def file_type_distribution_graph(df):
    
    df = df[['UUID','task','analysis_type','rat_ID']]

    data = df.groupby(['analysis_type','task']).agg(
                        num_UUIDs = ('UUID','nunique')).reset_index()
    
    order = data['task'].tolist()
    
    sns.barplot(x='task', y='num_UUIDs', data=data, hue="analysis_type", order=order, palette='Set2')
    plt.title(f"Double Cross Session Totals per File Type")
    plt.ylabel("Number of Sessions")
    plt.xlabel('Task/Analysis Type')
    plt.xticks(rotation=45)

    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/tsc2_file_type_dist_plots.png')
    plt.show()

    return data

def trial_totals_graph(df,tasks,analysis_types,genotype_color):

    df = df[['Genotype','rat_ID','Response','task','analysis_type']]

    filtered = df.loc[
            # (df['task'].isin(tasks))
            # & (df['analysis_type'].isin(analysis_types))
             (df['rat_ID'] != 745)
            # & (df['Max Delay (s)'] == delay_interval[0])
            # & (df['Min Delay (s)'] == delay_interval[1])
                      ]

    groups = filtered.groupby(['rat_ID','Genotype','task']).agg(
                total_trials=('Response','count'))

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()

    order = data['rat_ID'].tolist()
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 5), sharey=True)

    for ax, task in zip(axes, tasks):

        subset = data[data["task"] == task]

        sns.barplot(x='rat_ID', y='total_trials', data=subset, hue="Genotype", order=order, palette=genotype_color, ci=None, ax=ax)
        ax.set_title(f"Total Trials in {task} files")
        ax.set_ylabel("Trials Completed")
        ax.set_xlabel('Rat/Genotype')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_trial_totals_plots.png')
    plt.show()

    return data 

def training_prop_graph(df, genotype_color):

    df = df[['task','rat_ID','Genotype']]

    df['training'] = df['task'] == "Training"

    groups = df.groupby(['rat_ID','Genotype']).agg(
        training_num=('training','sum'),
        total_num=('task','count'))
    
    groups['training_props'] = groups['training_num'] / groups['total_num']

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()

    sns.boxplot(x='Genotype', y='training_props', data=data, palette=genotype_color, width=0.5)
    sns.swarmplot(x='Genotype', y='training_props', data=data, color='green', size=2)

    plt.title(f"Proportion of Training Files Per Genotype")
    plt.ylabel("Proportion of Training Files")
    plt.xlabel('Genotype')

    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_training_props_plots.png')
    plt.show()
    
    tukey = pairwise_tukeyhsd(endog=data['training_props'], groups=data['Genotype'], alpha=0.05)

    return data, tukey

def d_prime_graph(df, tasks, analysis_types):

    df = df[['Response','rat_ID','Genotype','UUID','task','analysis_type']]

    df = df.loc[(df['task'].isin(tasks))
                & (df['analysis_type'].isin(analysis_types))
    ]

    df['hit'] = df['Response'] == "Hit"
    df['miss'] = df['Response'] == "Miss"
    df['cr'] = df["Response"] == "CR"
    df['fa'] = df["Response"] == "FA"

    groups = df.groupby(['rat_ID','UUID','Genotype','task','analysis_type']).agg(
        hit_num=('hit', 'sum'),
        miss_num=('miss', 'sum'),
        cr_num=('cr', 'sum'),
        fa_num=('fa', 'sum'))

    groups['hit_rate'] = groups['hit_num'] / (groups['miss_num'] + groups['hit_num'])
    groups['fa_rate'] = groups['fa_num'] / (groups['fa_num'] + groups['cr_num'])

    groups['z_hit'] = norm.ppf(groups['hit_rate'])
    groups['z_fa'] = norm.ppf(groups['fa_rate']) 

    groups['d_prime'] = groups['z_hit'] - groups['z_fa']
    
    # groups['criterion'] = (-(groups['z_hit'] + groups['z_fa'])/2)

    data = groups.sort_values(by=['Genotype','rat_ID','task','analysis_type']).reset_index()

    tukey = pairwise_tukeyhsd(endog=data['hit_rate'], groups=data['Genotype'], alpha=0.05)

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), sharey=True)

    for ax, task in zip(axes, tasks):

        subset = data[data["task"] == task]

        sns.boxplot(x='Genotype', y='hit_rate', data=subset, palette='Set2', width=0.5, ax=ax)
        sns.swarmplot(x='Genotype', y='hit_rate', data=subset, color='black', size=1, ax=ax)

        ax.set_title(f"{task} Hit Rate Differences Across Genotype")
        ax.set_xlabel('Genotype')
        ax.set_ylabel("Hit Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/fmr1_le_hit_rate_plots.png')
    plt.show()

    return data, tukey

def single_prop(df, delay_interval, tasks, analysis_types, genotype_color):

    if tasks == ['Training']:
        image_handle = 'training'
    else:
        image_handle = 'baseline'

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['one_attempt'] = df['Attempts_to_complete'] == 1
    df['two_attempt'] = df['Attempts_to_complete'] == 2
    df['three_attempt'] = df['Attempts_to_complete'] == 3
    df['more_than_three_attempt'] = df['Attempts_to_complete'] > 3

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['Delay (s)'] < 2.5)
            & (df['task'].isin(tasks))
            & (df['analysis_type'].isin(analysis_types))
            & (df['rat_ID'] != 745)
            ]

    groups = filtered.groupby(['rat_ID', 'Genotype']).agg(
    trials_one_attempt=('one_attempt','sum'),
    trials_two_attempt=('two_attempt','sum'),
    trials_three_attempt=('three_attempt','sum'),
    trials_more_than_three_attempt=('more_than_three_attempt','sum'),
    total_trials=('Attempts_to_complete','count')).reset_index()

    for n in ['two', 'three', 'more_than_three']:
        groups[f'prop_{n}_attempt'] = groups[f'trials_{n}_attempt'] / groups['total_trials']

    grouped_rats_df = groups.reset_index()
    
    nattempts = ['prop_two_attempt','prop_three_attempt','prop_more_than_three_attempt']

    # tukey = pairwise_tukeyhsd(endog=data['prop_one_attempt'], groups=data['Genotype'], alpha=0.05)

    # fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
    
    # for ax, nattempt in zip(axes, nattempts):
    #     sns.boxplot(x='Genotype', y=nattempt, data=grouped_rats_df, palette=genotype_color, width=0.5, ax=ax)
    #     sns.swarmplot(x='Genotype', y=nattempt, data=grouped_rats_df, color='green', size=5, ax=ax)

    #     ax.set_title(nattempt.replace('prop_', '').replace('_attempt', '').capitalize())
    #     ax.set_xlabel('Genotype')
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    #     ax.set_ylabel('Proportion of Trials Completed')

    # plt.suptitle(f'Proportion of {image_handle} Trials Completed Across Attempt Number')
    # plt.tight_layout()
    # # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_{image_handle}_n_prop_plots.png')
    # plt.show()

    groups['prop_one'] = groups['trials_one_attempt'] / groups['total_trials']

    sns.boxplot(x='Genotype', y='prop_one', data=groups, palette=genotype_color, width=0.5)
    sns.swarmplot(x='Genotype', y='prop_one', data=groups, color='green', size=5)

    plt.title('Proportion of Single Attempt Trials by Genotype')
    plt.xlabel('Genotype')
    plt.ylabel('Proportion of Trials')

    plt.show()

    return grouped_rats_df

def var_graph(df,delay_interval,tasks,genotype_color):
    # graph swarm plots for each rat and genotype each point being a sessions proportion of single attempt trials
        
    df = df[['rat_ID','Genotype','Max Delay (s)','Min Delay (s)','UUID','Attempts_to_complete','Delay (s)','task']]

    low_rats = ['328','898','903','906','947','330','331','900','901','905','948','745','743']

    filtered = df.loc[
                    (df['Max Delay (s)'] == delay_interval[0])
                    & (df['Min Delay (s)'] == delay_interval[1])
                    & (df['Delay (s)'] > 2.5)
                    & (df['task'].isin(tasks))
                    & (~df['rat_ID'].isin(low_rats))
                      ]
    
    filtered['one_attempt'] = filtered['Attempts_to_complete'] == 1

    groups = filtered.groupby(['Genotype','rat_ID','UUID']).agg(
    trials_one_attempt=('one_attempt','sum'),
    total_trials=('Attempts_to_complete','count')
    ).reset_index()
    
    groups[f'prop_one_attempt'] = groups['trials_one_attempt'] / groups['total_trials']

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()
    
    order = data['rat_ID'].tolist()
    
    sns.boxplot(x='rat_ID', y=groups['prop_one_attempt'], data=data, hue=groups['Genotype'], order=order, palette=genotype_color, width=0.5)
    sns.swarmplot(x='rat_ID', y=groups['prop_one_attempt'], data=data, order=order, color='green', size=1)
    plt.title(f"Variance of Single Attempt Trial Proportions")
    plt.ylabel("Proportion of Single Attmept Trials")
    plt.xlabel('Rat/Genotype')

    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_var_plots.png')
    plt.show()

    prop_var_per_rat = data.groupby('Genotype')['prop_one_attempt'].var().reset_index(name='var')
    prop_var_geno = prop_var_per_rat.groupby('Genotype')['var'].var()

    sns.barplot(x='Genotype', y='var', data=prop_var_per_rat, palette=genotype_color)
    plt.title(f"Differences in Genotype Behavioral Variability")
    plt.ylabel("Variance of Single Attempt Proportions Over Sessions")
    plt.xlabel('Genotype')

    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_geno_var_plots.png')
    plt.show()

    return prop_var_per_rat

def x_session_motivation_graph(df,delay_interval,tasks,lo_df,genotype_color):
    n_trials = 800
    df = df[['task','Time_since_file_start_(s)','Response','rat_ID','Genotype','UUID','analysis_type','Min Delay (s)','Max Delay (s)','Delay (s)','new_task']]
    
    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['task'].isin(tasks))
            ]

    lo_dict = dict(zip(lo_df['UUID'], lo_df['lo_time']))
    filtered['lo_time'] = filtered['UUID'].map(lo_dict).fillna(0)

    rat_data_list = []
    
    for rat in filtered['rat_ID'].unique():
        rat_df = filtered.loc[filtered['rat_ID'] == rat]
        genotype = rat_df['Genotype'].iloc[0]

        for task in rat_df['task'].unique():
            task_df = rat_df[rat_df['task'] == task]
            
            for uuid in task_df['UUID'].unique():
                session_df = rat_df.loc[rat_df['UUID'] == uuid].sort_values('Time_since_file_start_(s)')

                for trial_num, row in enumerate(session_df.itertuples(), start=1):
                    rat_data_list.append({
                        'rat_ID': rat,
                        'task': task,
                        'UUID': uuid,
                        'trial_number': trial_num,
                        'Time_since_file_start_(s)': row._2,
                        'Response': row.Response,
                        'Genotype': genotype,
                        'lo_time': row.lo_time
                })

    iti_df = pd.DataFrame(rat_data_list)

    iti_df['ITI'] = iti_df.groupby('UUID')['Time_since_file_start_(s)'].diff()

    iti_df = iti_df.dropna(subset=['ITI'])

    iti_df['response_prev'] = iti_df.groupby('UUID')['Response'].shift(1)

    specific_response = 'FA'
    iti_df['ITI_adjusted'] = iti_df['ITI'] - iti_df['lo_time'].where(iti_df['response_prev'] == specific_response, 0)

    n_bins = 10
    bin_edges = np.linspace(0, n_trials, n_bins + 1)
    bin_labels = (bin_edges[:-1] + bin_edges[1:]) / 2
    iti_df["bin"] = pd.cut(iti_df["trial_number"], bins=bin_edges, labels=False)

    per_rat = (
    iti_df
    .groupby(["rat_ID", "Genotype", "task", "bin"])["ITI"]
    .median()
    .reset_index()
)
    plot_df = (
    per_rat
    .groupby(["Genotype", "task", "bin"])["ITI"]
    .agg(['mean', 'sem'])
    .reset_index()
)
    tasks = plot_df["task"].unique()
    fig, axes = plt.subplots(1, (len(tasks)), figsize=(6 * len(tasks), 5), sharey=True)

    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        subset = plot_df[plot_df["task"] == task]
        for genotype, geno_df in subset.groupby("Genotype"):
            ax.errorbar(geno_df["bin"],
            geno_df["mean"],
            yerr=geno_df["sem"], 
            fmt='-o', 
            color=genotype_color[genotype], 
            label=f"{task} ({genotype})")

        if task == tasks[0]:
            ax.set_ylabel("Median inter-trial interval (s)")
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([int(x) for x in bin_labels])
        ax.set_xlabel("Approx. absolute trial number")

        y_min, y_max = geno_df['mean'].min() - 5, geno_df['mean'].max() + 5
        ax.set_ylim(y_min, y_max)
        
        ax.set_title(f"Motivation trend ({task})")
        ax.set_xlabel("Session progression (binned)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig('C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_xsession_motiv_plots.png')
    plt.show()
    
    return iti_df

def main():

### data paths and wanted info

    file_path="C:/Users/ckill/Documents/AuerbachLab/LabFiles/Fmr1-LE_data_exported_trials_20251015.csv"
    file_info_path="C:/Users/ckill/Documents/AuerbachLab/LabFiles/Fmr1-LE_data_exported_20251015.csv"

    # Tsc2-LE_data_exported_trials_20251015, Fmr1-LE_data_exported_trials_20251015

    wanted_columns_for_merge = ['new_task','date','UUID','rat_ID','DOB','file_name','Genotype','analysis_type','task','lo_time','weight']
    wanted_delay_interval = (4.0,1.0)
    training_tasks = ['Training']
    training_analysis_types = ['Training - BBN','BBN (Standard)','Tone (Single)']
    tasks = ['Rxn','TH']
    file_diff_tasks = ['Rxn','TH','Training']
    baseline_tasks = ['Rxn','TH']
    analysis_types = ["BBN (Standard)", "Tone (Standard)"]
    twox_task = ['Baseline','Training']
    twox_analysis_type = ['Training - BBN','Tone (Standard)','BBN (Standard)','Tone (Single)']
    genotype_color = {'Tsc2_LE_WT': 'k',
                        'Tsc2_LE_Het': 'b',
                        'Fmr1_WT-Tsc2_WT': 'k',
                        'Fmr1_WT-Tsc2_Het': 'b',
                        'Fmr1_KO-Tsc2_WT': 'r',
                        'Fmr1_KO-Tsc2_Het': 'm',
                        'Fmr1-LE_WT': 'k',
                        'Fmr1-LE_KO': 'r'    }
    
### data cleaning and organization
    df, info_df = load_data(file_path, file_info_path)
    info_df = file_info(info_df)
    info_df, dB_df, lo_df, dB_range_set, tst_num_trn, tst_num_rxn, tst_num_th = file_type_finder(info_df)
    shared_UUIDs = uuid_comparer(df, info_df)
    clean_df, num_trn, num_rxn, num_th = data_cleaner(df, info_df, shared_UUIDs, wanted_columns_for_merge)
    delay_interval_list, delay_df, rat_ids, dob_list, gt_list = delay_classifier(clean_df)

### data analysis graphs

## single_props

    short_long_data = short_long_comparison_graph(delay_df,wanted_delay_interval,file_diff_tasks,analysis_types)
    # att_to_delay_corr = attempts_to_delay_corr(delay_df,wanted_delay_interval,baseline_tasks) # not a great graph

    # training_single_prop_data = single_prop(delay_df,wanted_delay_interval,training_tasks,training_analysis_types,genotype_color)
    # baseline_single_prop_data = single_prop(delay_df,wanted_delay_interval,file_diff_tasks,analysis_types, genotype_color)

    #^ look for difference in variance between sessions for each rat
    # prop_var_geno = var_graph(delay_df,wanted_delay_interval,baseline_tasks,genotype_color)

    # file_diff_data, file_diff_tukey = file_diff_graph(delay_df, file_diff_tasks, analysis_types, wanted_delay_interval,genotype_color)

    # weight_diff_data = weight_diff_graph

## controls

    # file_dist_data = file_type_distribution_graph(delay_df)
    # trial_totals_data = trial_totals_graph(delay_df,file_diff_tasks,analysis_types,genotype_color)
    # training_time_props_data, training_props_tukey = training_prop_graph(delay_df, genotype_color)
    # age_data = age_differences(delay_df,baseline_tasks,genotype_color,wanted_delay_interval)
    # *
    
    # rat_session_total_data = rat_session_totals(delay_df,genotype_color)
    ####### exclude rats with a different trial number

    ####### y axis is rat id x axis is experiment run time bars change color as file changes from training to baseline
    # training_periods(info_df,file_diff_tasks,wanted_delay_interval)

    # how to find back to back days of training 
    # show training length for each rat on line graph
    
    ####### graph amounts of each 1-4 delay time

    #^ double check that training files are actually training files by file name 0catch 3catch
    #^ doesn't work for fmr1_le csv since training files arent designated by 0/3catch should work for twox

## other 

    # d_prime_data, d_prime_tukey = d_prime_graph(delay_df,file_diff_tasks,analysis_types)

    ### make graph for fa rate across genotype and task

    # weight loss and weight gain
    # weight_df = weight_var_finder(delay_df)
    # motivation_data = x_session_motivation_graph(delay_df,wanted_delay_interval,file_diff_tasks,lo_df, genotype_color)
    # attempts_over_session(delay_df)

    # show all sessions for each genotype and then do significance test for uniformity?
    # compare average height of each genotypes' graphs
    
    #^ need to control for false alarm time out if previous was FA exlude data

    ####### difference between hits, misses, and FA for attempt number

    ####### intertrial interval and fa rate

    ####### Need to help farheen with file organization
    
    ####### How do i fix the graphs to make them better? better weight graphs. showing not variance of weight but instead 

    ####### track changes in behavior over lifetime session_date-dob


### program testing
    rat_ID_info = delay_df.loc[(delay_df['rat_ID'] == 328)]
    rat_age = rat_ID_info['DOB'].unique()

    print(f'''
Data using Tones and BBN with all different durations
delay intervals: {delay_interval_list}
DOBs: {dob_list}
dB ranges: {dB_range_set}
Genotypes: {gt_list}
total rats in df: {len(rat_ids)}
shared UUIDs: {len(shared_UUIDs)}
number of trials: {len(clean_df)}
number of training sessions: {num_trn}
number of rxn sessions: {num_rxn}
number of th sessions: {num_th}
number of test training sessions: {tst_num_trn}
number of test rxn sessions: {tst_num_rxn}
number of test th sessions: {tst_num_th}
data: {age_data}
''')

if __name__ == "__main__":
    main()
