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
import re

def load_data(file_path, info_path):
    ### loads needed dataframes from file paths
    df = pd.read_csv(file_path)
    info_df = pd.read_csv(info_path)
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
    return clean_df
    
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
    dob_df = df['DOB'].unique()
    dob_list = dob_df.tolist()
    gt_df = df['Genotype'].unique()
    gt_list = gt_df.tolist()
    return delay_interval_list, df_with_classified_delays, rat_ids, dob_list, gt_list



##################################################################################################
##################################################################################################
##################################################################################################



def file_diff_graph(df, tasks, analysis_types, delay_interval):

    df = df[['task','analysis_type','Attempts_to_complete','rat_ID','Delay (s)',"Max Delay (s)","Min Delay (s)"]]

    df['one_attempt'] = df['Attempts_to_complete'] == 1
    df['more_than_one_attempt'] = df['Attempts_to_complete'] > 1

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['Delay (s)'] > 2.5)
            & (df['task'].isin(tasks))
            & (df['analysis_type'].isin(analysis_types))
            ]
    groups = filtered.groupby(['analysis_type', 'task']).agg(
    trials_one_attempt=('one_attempt', 'sum'),
    trials_more_than_one_attempt=('more_than_one_attempt', 'sum'),
    total_trials=('Attempts_to_complete', 'count'))

    groups['prop_one_attempt'] = groups['trials_one_attempt'] / groups['total_trials']
    groups['prop_more_than_one'] = groups['trials_more_than_one_attempt'] / groups['total_trials']
    grouped_rats_df = groups.sort_values(by=['analysis_type', 'task']).reset_index()
    
    prop_one_attempt_data = grouped_rats_df[['prop_one_attempt','analysis_type','task']]
    data = prop_one_attempt_data.sort_values(by=['task','analysis_type']).reset_index()
    
    tukey = pairwise_tukeyhsd(endog=data['prop_one_attempt'], groups=data['task'], alpha=0.05)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='task', y='prop_one_attempt', data=data, palette='Set2', width=0.5)

    sns.swarmplot(x='task', y='prop_one_attempt', data=data, color='black', size=5)

    plt.title('Proportion of 1-Attempt Trials by File Type')
    plt.ylabel('Proportion of Trials Completed in One Attempt')
    plt.xlabel('File Type')

    plt.tight_layout()
    plt.show()

    return data, tukey

def file_type_distribution_graph(df):
    
    df = df[['UUID','task','analysis_type','rat_ID']]

    uuids_list = []

    data = df.groupby(['analysis_type','task']).agg(
                        num_UUIDs = ('UUID','nunique')).reset_index()
    
    order = data['task'].tolist()
    
    sns.barplot(x='task', y='num_UUIDs', data=data, hue="analysis_type", order=order, palette='Set2')
    plt.title(f"Session Totals per File Type")
    plt.ylabel("Number of Sessions")
    plt.xlabel('Task/Analysis Type')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return data

def trial_totals_graph(df,tasks,analysis_types):

    df = df[['Genotype','rat_ID','Response','task','analysis_type']]

    filtered = df.loc[(df['task'].isin(tasks))
        & (df['analysis_type'].isin(analysis_types))
                      ]

    groups = filtered.groupby(['rat_ID','Genotype','task']).agg(
                total_trials=('Response','count'))

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()

    order = data['rat_ID'].tolist()

    sns.barplot(x='rat_ID', y='total_trials', data=data, hue="Genotype", order=order, palette='Set2')
    plt.title(f"Total Trials for each Genotype")
    plt.ylabel("Trials Completed")
    plt.xlabel('Rat/Genotype')

    plt.tight_layout()
    plt.show()

    return data 

def training_prop_graph(df):

    df = df[['task','rat_ID','Genotype']]

    df['training'] = df['task'] == "Training"

    groups = df.groupby(['rat_ID','Genotype']).agg(
        training_num=('training','sum'),
        total_num=('task','count'))
    
    groups['training_props'] = groups['training_num'] / groups['total_num']

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()

    sns.boxplot(x='Genotype', y='training_props', data=data, palette='Set2', width=0.5)
    sns.swarmplot(x='Genotype', y='training_props', data=data, color='black', size=2)

    plt.title(f"Proportion of Training Files Per Genotype")
    plt.ylabel("Proportion of Training Files")
    plt.xlabel('Genotype')

    plt.tight_layout()
    plt.show()

    return data

def d_prime_graph(df):

    df = df[['Response','rat_ID','Genotype','UUID']]

    df['hit'] = df['Response'] == "Hit"
    df['miss'] = df['Response'] == "Miss"
    df['cr'] = df["Response"] == "CR"
    df['fa'] = df["Response"] == "FA"

    groups = df.groupby(['rat_ID','UUID','Genotype']).agg(
        hit_num=('hit', 'sum'),
        miss_num=('miss', 'sum'),
        cr_num=('cr', 'sum'),
        fa_num=('fa', 'sum'))

    groups['hit_rate'] = groups['hit_num'] / (groups['miss_num'] + groups['hit_num'])
    groups['fa_rate'] = groups['fa_num'] / (groups['fa_num'] + groups['cr_num'])

    groups['z_hit'] = norm.ppf(groups['hit_rate'])  # Z(H)
    groups['z_fa'] = norm.ppf(groups['fa_rate'])  # Z(F)

    groups['d_prime'] = groups['z_hit'] - groups['z_fa']

    # groups['criterion'] = (-(groups['z_hit'] + groups['z_fa'])/2)

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    sns.boxplot(x='rat_ID', y='d_prime', data=data, hue='Genotype', palette='Set2', width=0.5, ax=ax1)
    sns.swarmplot(x='rat_ID', y='d_prime', data=data, color='black', size=1, ax=ax1)

    ax1.set_title("d' Differences Across Genotype/Rat")
    ax1.set_xlabel('Rat ID')
    ax1.set_ylabel("d'")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend_.remove()

    sns.boxplot(x='Genotype', y='d_prime', data=data, palette='Set2', width=0.5, ax=ax2)
    sns.swarmplot(x='Genotype', y='d_prime', data=data, color='black', size=1, ax=ax2)

    ax2.set_title("d' Average Differences Across Genotype")
    ax2.set_xlabel('Genotype')
    ax2.set_ylabel("")

    fig.legend(handles, labels, title='Genotype', loc='upper center', ncol=len(labels))

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend
    plt.show()

    return data

def single_prop(df, delay_interval, tasks, analysis_types):

    if tasks == ['Training']:
        label = 'Proportion of 1-Attempt Trials by Genotype Over Training Period'
    else:
        label = 'Proportion of 1-Attempt Trials by Genotype'

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['one_attempt'] = df['Attempts_to_complete'] == 1
    df['more_than_one_attempt'] = df['Attempts_to_complete'] > 1

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['Delay (s)'] > 2.5)
            & (df['task'].isin(tasks))
            & (df['analysis_type'].isin(analysis_types))
            ]
    
    groups = filtered.groupby(['rat_ID', 'Genotype']).agg(
    trials_one_attempt=('one_attempt', 'sum'),
    trials_more_than_one_attempt=('more_than_one_attempt', 'sum'),
    total_trials=('Attempts_to_complete', 'count'))

    groups['prop_one_attempt'] = groups['trials_one_attempt'] / groups['total_trials']
    groups['prop_more_than_one'] = groups['trials_more_than_one_attempt'] / groups['total_trials']
    grouped_rats_df = groups.sort_values(by=['Genotype', 'rat_ID']).reset_index()
    
    prop_one_attempt_data = grouped_rats_df[['prop_one_attempt','rat_ID','Genotype']]
    data = prop_one_attempt_data.sort_values(by=['Genotype','rat_ID']).reset_index()

    tukey = pairwise_tukeyhsd(endog=data['prop_one_attempt'], groups=data['Genotype'], alpha=0.05)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Genotype', y='prop_one_attempt', data=data, palette='Set2', width=0.5)

    sns.swarmplot(x='Genotype', y='prop_one_attempt', data=data, color='black', size=5)

    plt.title(label)
    plt.ylabel('Proportion of Trials Completed in One Attempt')
    plt.xlabel('Genotype')

    plt.tight_layout()
    plt.show()

    return data, tukey

def x_session_motivation_graph(df,delay_interval,tasks):

    df = df[['Time_since_file_start_(s)','rat_ID','Genotype','UUID','task','analysis_type','Min Delay (s)','Max Delay (s)']]

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['task'].isin(tasks))
            ]

    rat_data_list = []

    for rat in filtered['rat_ID'].unique():
        rat_df = filtered.loc[filtered['rat_ID'] == rat]
        genotype = rat_df['Genotype'].iloc[1]

        for uuid in rat_df['UUID'].unique():
            session_df = rat_df.loc[rat_df['UUID'] == uuid].sort_values('Time_since_file_start_(s)')

            itis = np.diff(session_df['Time_since_file_start_(s)'].values)
            n_trials = len(itis)
            if len(itis) == 0:
                continue
            relative_trial_pos = np.linspace(0, 1, n_trials, endpoint=False)

            for iti, pos in zip(itis, relative_trial_pos):
                rat_data_list.append({
                    'rat_ID': rat,
                    'Genotype': genotype,
                    'UUID': uuid,
                    'ITI': iti,
                    'relative_trial_pos': pos
            })

    iti_df = pd.DataFrame(rat_data_list)

    iti_df["bin"] = pd.cut(iti_df["relative_trial_pos"], bins=10, labels=False)
    iti_by_bin = iti_df.groupby(["rat_ID", 'Genotype', "UUID", "bin"])["ITI"].median().reset_index()

    genotypes = iti_by_bin["Genotype"].unique()
    fig, axes = plt.subplots(1, len(genotypes), figsize=(6 * len(genotypes), 5), sharey=True)

    if len(genotypes) == 1:
        axes = [axes]

    for ax, genotype in zip(axes, genotypes):
        subset = iti_by_bin[iti_by_bin["Genotype"] == genotype]
        mean_iti = subset.groupby("bin")["ITI"].mean()
        sem_iti = subset.groupby("bin")["ITI"].sem()

        ax.errorbar(mean_iti.index, mean_iti, yerr=sem_iti, fmt='-o', label=genotype)
        ax.set_title(f"Motivation trend ({genotype})")
        ax.set_xlabel("Session progression (binned)")
        ax.set_ylabel("Median inter-trial interval (s)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    return iti_df

def main():

### data paths and wanted info

    file_path="C:/Users/ckill/Documents/neuroscience_sterf/AuerbachLab/Fmr1-LE_data_exported_trials_20251015.csv"
    file_info_path="C:/Users/ckill/Documents/neuroscience_sterf/AuerbachLab/Fmr1-LE_data_exported_20251015.csv"
    wanted_columns_for_merge = ['date','UUID','weight','rat_ID','DOB','file_name','Genotype','task','analysis_type']
    wanted_delay_interval = (4.0,1.0)
    wanted_month = (2025,1)
    wanted_age = (2024,7)
    training_tasks = ['Training']
    tasks = ['Rxn','TH']
    file_diff_tasks = ['Rxn','TH','Training']
    analysis_types = ["BBN (Standard)", "Tone (Single)"]
    
### data cleaning and organization
    df, info_df = load_data(file_path, file_info_path)
    info_df = file_info(info_df)
    shared_UUIDs = uuid_comparer(df, info_df)
    clean_df = data_cleaner(df, info_df, shared_UUIDs, wanted_columns_for_merge)
    delay_interval_list, delay_df, rat_ids, dob_list, gt_list = delay_classifier(clean_df)

### data analysis graphs

## single_props

    # training_single_prop_data, training_prop_tukey = single_prop(delay_df,wanted_delay_interval,training_tasks,analysis_types)
    # baseline_single_prop_data, baseline_prop_tukey = single_prop(delay_df,wanted_delay_interval,tasks,analysis_types)

    # file_diff_data, file_diff_tukey = file_diff_graph(delay_df, file_diff_tasks, analysis_types, wanted_delay_interval)

    # weight_diff_data = weight_diff_graph

## controls

    # file_dist_data = file_type_distribution_graph(delay_df)
    # trial_totals_data = trial_totals_graph(delay_df,tasks,analysis_types)
    # training_time_props_data = training_prop_graph(delay_df)

## other 

    # d_prime_data = d_prime_graph(delay_df)
    # need to separate by task
    motivation_data = x_session_motivation_graph(delay_df,wanted_delay_interval,tasks)
    # need to separate by task

### program testing
    print(f'''
Data using Tones and BBN with all different durations
delay intervals: {delay_interval_list}
DOBs: {dob_list}
Genotypes: {gt_list}
total rats in df: {len(rat_ids)}
shared UUIDs: {len(shared_UUIDs)}
number of trials: {len(clean_df)}
data: {motivation_data}
''')
    
if __name__ == "__main__":
    main()
