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

    info_df = info_df.loc[(info_df['task'] != 'Reset')
                          & (info_df['task'] != 'Discrimination')
                          & (info_df['task'] != 'Holding')
                          & (info_df['task'] != 'Tsc2_LE')]

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

def weight_drop_finder(df):

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    merged_df = df.sort_values(['rat_ID', 'date'])

    # Compute daily % weight change per rat
    merged_df = merged_df.groupby(['rat_ID','UUID'])['weight'].pct_change(fill_method=None) * 100

    # Flag significant drops (>7%)
    merged_df['sig_drop_7'] = merged_df < -7

    print(merged_df)

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
    dob_df = df['DOB'].unique()
    dob_list = dob_df.tolist()
    gt_df = df['Genotype'].unique()
    gt_list = gt_df.tolist()
    return delay_interval_list, df_with_classified_delays, rat_ids, dob_list, gt_list



########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################



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
    plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_file_diff_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    tukey = pairwise_tukeyhsd(endog=data['prop_one_attempt'], groups=data['Genotype'], alpha=0.05)
    
    return data, tukey

def rat_session_totals(df,genotype_color):
    df = df[['rat_ID','UUID','Genotype']]
    
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
    plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_rat_sessions_totals_plots.png')
    plt.show()

    return data
    
def file_type_distribution_graph(df):
    
    df = df[['UUID','task','analysis_type','rat_ID']]

    data = df.groupby(['analysis_type','task']).agg(
                        num_UUIDs = ('UUID','nunique')).reset_index()
    
    order = data['task'].tolist()
    
    sns.barplot(x='task', y='num_UUIDs', data=data, hue="analysis_type", order=order, palette='Set2')
    plt.title(f"Session Totals per File Type")
    plt.ylabel("Number of Sessions")
    plt.xlabel('Task/Analysis Type')
    plt.xticks(rotation=45)

    plt.tight_layout()
    # plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/tsc2_file_type_dist_plots.png')
    plt.show()

    return data

def trial_totals_graph(df,tasks,analysis_types,genotype_color):

    df = df[['Genotype','rat_ID','Response','task','analysis_type']]

    filtered = df.loc[(df['task'].isin(tasks))
        & (df['analysis_type'].isin(analysis_types))
                      ]

    groups = filtered.groupby(['rat_ID','Genotype','task']).agg(
                total_trials=('Response','count'))

    data = groups.sort_values(by=['Genotype','rat_ID']).reset_index()

    order = data['rat_ID'].tolist()
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), sharey=True)

    for ax, task in zip(axes, tasks):

        subset = data[data["task"] == task]

        sns.barplot(x='rat_ID', y='total_trials', data=subset, hue="Genotype", order=order, palette=genotype_color, ci=None, ax=ax)
        ax.set_title(f"Total Trials for each Genotype in {task} files")
        ax.set_ylabel("Trials Completed")
        ax.set_xlabel('Rat/Genotype')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_trial_totals_plots.png')
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
    plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_training_props_plots.png')
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
            & (df['Delay (s)'] > 2.5)
            & (df['task'].isin(tasks))
            & (df['analysis_type'].isin(analysis_types))
            ]

    groups = filtered.groupby(['rat_ID', 'Genotype']).agg(
    trials_one_attempt=('one_attempt','sum'),
    trials_two_attempt=('two_attempt','sum'),
    trials_three_attempt=('three_attempt','sum'),
    trials_more_than_three_attempt=('more_than_three_attempt','sum'),
    total_trials=('Attempts_to_complete','count'))

    for n in ['one', 'two', 'three', 'more_than_three']:
        groups[f'prop_{n}_attempt'] = groups[f'trials_{n}_attempt'] / groups['total_trials']

    grouped_rats_df = groups.reset_index()
    
    nattempts = ['prop_one_attempt','prop_two_attempt','prop_three_attempt','prop_more_than_three_attempt']

    # tukey = pairwise_tukeyhsd(endog=data['prop_one_attempt'], groups=data['Genotype'], alpha=0.05)

    fig, axes = plt.subplots(1, len(nattempts), figsize=(6 * len(nattempts), 5), sharey=True)
    for ax, nattempt in zip(axes, nattempts):
        sns.boxplot(x='Genotype', y=nattempt, data=grouped_rats_df, palette=genotype_color, width=0.5, ax=ax)

        sns.swarmplot(x='Genotype', y=nattempt, data=grouped_rats_df, color='black', size=5, ax=ax)

        ax.set_title(nattempt.replace('prop_', '').replace('_attempt', '').capitalize())
        ax.set_ylabel('Proportion of Trials Completed')
        ax.set_xlabel('Genotype')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.suptitle('Proportion of Trials Completed Across Attempt Number')
    plt.tight_layout()
    plt.savefig(f'C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_{image_handle}_n_prop_plots.png')
    plt.show()

    return grouped_rats_df

def x_session_motivation_graph(df,delay_interval,tasks,lo_df,genotype_color):
    n_trials = 800
    df = df[['task','Time_since_file_start_(s)','Response','rat_ID','Genotype','UUID','analysis_type','Min Delay (s)','Max Delay (s)','new_task']]
    
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

    iti_by_bin = iti_df.groupby(["rat_ID", 'task', "UUID", "bin",'Genotype'])["ITI"].median().reset_index()

    task = iti_by_bin["task"].unique()

    fig, axes = plt.subplots(1, len(task), figsize=(6 * len(task), 5), sharey=True)

    if len(task) == 1:
        axes = [axes]

    for ax, task in zip(axes, task):
        subset = iti_by_bin[iti_by_bin["task"] == task]
        for genotype, geno_df in subset.groupby("Genotype"):
            mean_iti = geno_df.groupby("bin")["ITI"].mean()
            sem_iti = geno_df.groupby("bin")["ITI"].sem()

            ax.errorbar(mean_iti.index, mean_iti, yerr=sem_iti, fmt='-o', color=genotype_color[genotype], label=f"{task} ({genotype})")

        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([int(x) for x in bin_labels])
        ax.set_xlabel("Approx. absolute trial number")

        y_min, y_max = mean_iti.min() - 5, mean_iti.max() + 5
        ax.set_ylim(y_min, y_max)
        
        ax.set_title(f"Motivation trend ({task})")
        ax.set_xlabel("Session progression (binned)")
        ax.set_ylabel("Median inter-trial interval (s)")
        ax.legend()
    plt.tight_layout()
    plt.savefig('C:/Users/ckill/OneDrive/Documents/GitHub/o_behavior_data_analysis_fall2025/figures/twox_xsession_motiv_plots.png')
    plt.show()
    
    return iti_df

def main():

### data paths and wanted info

    file_path="C:/Users/ckill/Documents/neuroscience_sterf/AuerbachLab/FXS x TSC_archive.csv"
    file_info_path="C:/Users/ckill/Documents/neuroscience_sterf/AuerbachLab/FXS x TSC_data_exported_20250801.csv"
    wanted_columns_for_merge = ['new_task','date','UUID','rat_ID','DOB','file_name','Genotype','analysis_type','task','lo_time','weight']
    wanted_delay_interval = (4.0,1.0)
    training_tasks = ['Training']
    training_analysis_types = ['Training - BBN','BBN (Standard)','Tone (Single)']
    tasks = ['Rxn','TH']
    file_diff_tasks = ['Rxn','TH','Training']
    analysis_types = ["BBN (Standard)", "Tone (Single)"]
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

    # training_single_prop_data = single_prop(delay_df,wanted_delay_interval,training_tasks,training_analysis_types,genotype_color)
    # baseline_single_prop_data = single_prop(delay_df,wanted_delay_interval,file_diff_tasks,analysis_types, genotype_color)

    ####### look for difference in variance between sessions for each rat

    # file_diff_data, file_diff_tukey = file_diff_graph(delay_df, twox_task, analysis_types, wanted_delay_interval,genotype_color)

    # weight_diff_data = weight_diff_graph

## controls

    # file_dist_data = file_type_distribution_graph(delay_df)
    # trial_totals_data = trial_totals_graph(delay_df,twox_task,analysis_types,genotype_color)
    # training_time_props_data, training_props_tukey = training_prop_graph(delay_df, genotype_color)
    # *
    
    # rat_session_total_data = rat_session_totals(delay_df,genotype_color)
    ####### exclude rats with a different trial number
    
    ####### graph amounts of each 1-4 delay time

    #^ double check that training files are actually training files by file name 0catch 3catch
    #^ doesn't work for fmr1_le csv since training files arent designated by 0/3catch should work for twox

## other 

    # d_prime_data, d_prime_tukey = d_prime_graph(delay_df,file_diff_tasks,analysis_types)

    ### make graph for fa rate across genotype and task

    # weight loss and weight gain
    # weight_drop_finder(delay_df)
    # motivation_data = x_session_motivation_graph(delay_df,wanted_delay_interval,twox_task,lo_df, genotype_color)

    # show all sessions for each genotype and then do sognificance test for uniformity?
    # compare average height of each genotypes' graphs
    
    #^ need to control for false alarm time out if previous was FA exlude data

    ####### difference between hits, misses, and FA for attempt number

    ####### intertrial interval and fa rate

    
    ### FGX = red tsc = blue 2cross = purple wt = black

### program testing
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
data: {file_diff_tukey}
''')
    
if __name__ == "__main__":
    main()
