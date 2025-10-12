import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import pingouin as pg
import seaborn as sns
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

def file_type_distribution_graph(df):
    pass
    # df = 

    # data = 

    # return data

def trial_totals_graph(df):

    df = df[['Genotype','rat_ID','Response','task','analysis_type']]

def training_prop_graph(df):

    df = df[['task','rat_ID','Genotype']]

    df['training'] = df['task'] == "Training"

    groups = df.groupby(['rat_ID','Genotype']).agg(
        training_num=('training','sum'),
        total_num=('task','count'))
    
    groups['training_props'] = groups['training_num'] / groups['total_num']

    sns.boxplot(x='Genotype', y='training_props', data=groups, palette='Set2', width=0.5)
    sns.swarmplot(x='Genotype', y='training_props', data=groups, color='black', size=2)

    plt.title(f"Proportion of Training Files Per Genotype")
    plt.ylabel("Proportion of Training Files")
    plt.xlabel('Genotype')

    plt.tight_layout()
    plt.show()

    return groups

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

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    sns.boxplot(x='rat_ID', y='d_prime', data=data, hue='Genotype', palette='Set2', width=0.5, ax=ax1)
    sns.swarmplot(x='rat_ID', y='d_prime', data=data, hue='Genotype', color='black', size=2, ax=ax1)

    plt.title(f"d' Differences Across Genotype/Rat")
    plt.ylabel("d'")
    plt.xlabel('Rat/Genotype')

    sns.boxplot(x='Genotype', y='d_prime', data=data, palette='Set2', width=0.5, ax=ax2)
    sns.swarmplot(x='Genotype', y='d_prime', data=data, color='black', size=2, ax=ax2)

    plt.title(f"d' Average Differences Across Genotype")
    plt.ylabel("d'")
    plt.xlabel('Genotype')

    plt.tight_layout()
    plt.show()

    return data

def single_prop_over_training(df, delay_interval):

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['one_attempt'] = df['Attempts_to_complete'] == 1
    df['more_than_one_attempt'] = df['Attempts_to_complete'] > 1

    filtered = df.loc[
            (df['Max Delay (s)'] == delay_interval[0])
            & (df['Min Delay (s)'] == delay_interval[1])
            & (df['Delay (s)'] > 2.5)
            & (df['task'] == 'Training')
            ]
    print(filtered)
    groups = filtered.groupby(['rat_ID', 'Genotype']).agg(
    trials_one_attempt=('one_attempt', 'sum'),
    trials_more_than_one_attempt=('more_than_one_attempt', 'sum'),
    total_trials=('Attempts_to_complete', 'count'))

    groups['prop_one_attempt'] = groups['trials_one_attempt'] / groups['total_trials']
    groups['prop_more_than_one'] = groups['trials_more_than_one_attempt'] / groups['total_trials']
    grouped_rats_df = groups.sort_values(by=['Genotype', 'rat_ID']).reset_index()
    
    prop_one_attempt_data = grouped_rats_df[['prop_one_attempt','rat_ID','Genotype']]
    data = prop_one_attempt_data.sort_values(by=['Genotype','rat_ID']).reset_index()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Genotype', y='prop_one_attempt', data=data, palette='Set2', width=0.5)

    sns.swarmplot(x='Genotype', y='prop_one_attempt', data=data, color='black', size=5)

    plt.title('Proportion of 1-Attempt Trials by Genotype Over Training Period')
    plt.ylabel('Proportion of Trials Completed in One Attempt')
    plt.xlabel('Genotype')

    plt.tight_layout()
    plt.show()

    return data

def main():
    ### data paths and wanted info
    file_path="C:/Users/ckill/Documents/neuroscience_sterf/AuerbachLab/FXS x TSC_archive.csv"
    file_info_path="C:/Users/ckill/Documents/neuroscience_sterf/AuerbachLab/FXS x TSC_data_exported_20250801.csv"
    wanted_columns_for_merge = ['date','UUID','weight','rat_ID','DOB','file_name','Genotype','task','analysis_type']
    wanted_delay_interval = (4.0,1.0)
    wanted_month = (2025,1)
    wanted_age = (2024,7)

    ### data cleaning and organization
    df, info_df = load_data(file_path, file_info_path)
    info_df = file_info(info_df)
    shared_UUIDs = uuid_comparer(df, info_df)
    clean_df = data_cleaner(df, info_df, shared_UUIDs, wanted_columns_for_merge)
    delay_interval_list, delay_df, rat_ids, dob_list, gt_list = delay_classifier(clean_df)

    ### data analysis
    file_dist_data = file_type_distribution_graph(df)
    # training_props_data = training_prop_graph(delay_df)
    # d_prime_data = d_prime_graph(delay_df)
    # prop_data = single_prop_over_training(delay_df,wanted_delay_interval) 

    ### program testing
    print(f'''
Data using Tones and BBN 
delay intervals: {delay_interval_list}
DOBs: {dob_list}
Genotypes: {gt_list}
total rats in df: {len(rat_ids)}
shared UUIDs: {len(shared_UUIDs)}
number of trials: {len(clean_df)}
rat data: {file_dist_data}
''')
    
if __name__ == "__main__":
    main()
