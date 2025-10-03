import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
import pingouin as pg
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

def rat_info(df, delay_interval, date): 
    # returns a dataframe with each genotypes rats and their counts of single attempts and greater than 1 attempts
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['one_attempt'] = df['Attempts_to_complete'] == 1
    df['more_than_one_attempt'] = df['Attempts_to_complete'] > 1
    
    dfs = []
    for month in range(1,13):
        filtered = df.loc[
                # (df['Max Delay (s)'] == delay_interval[0])
                # & (df['Min Delay (s)'] == delay_interval[1])
                # & (df['Delay (s)'] > 2.5)
                (df['date'].dt.year == date[0])
                & (df['date'].dt.month == month)
                ]
        if filtered.empty:
            continue  # Skip empty months

        groups = filtered.groupby(['rat_ID', 'Genotype']).agg(
        trials_one_attempt=('one_attempt', 'sum'),
        trials_more_than_one_attempt=('more_than_one_attempt', 'sum'),
        total_trials=('Attempts_to_complete', 'count'))

        groups['prop_one_attempt'] = groups['trials_one_attempt'] / groups['total_trials']
        groups['prop_more_than_one'] = groups['trials_more_than_one_attempt'] / groups['total_trials']
        grouped_rats_df = groups.sort_values(by=['Genotype', 'rat_ID']).reset_index()
        grouped_rats_df['month'] = month

        prop_one_attempt_data = grouped_rats_df[['prop_one_attempt','rat_ID','Genotype','month']]
        prop_one_attempt_data = prop_one_attempt_data.sort_values(by=['Genotype', 'rat_ID', 'month']).reset_index()

        dfs.append(prop_one_attempt_data)
    
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df

def genotype_bargraph(data):
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    
    sns.lineplot(
        data=data,
        x='month',
        y='prop_one_attempt',
        hue='Genotype',
        units='rat_ID',
        estimator=None,
        alpha=0.2,              # Faded lines
        linewidth=1,
        palette='muted',        # Softer colors
        legend=False
        )

    sns.lineplot(
        data=data,
        x='month',
        y='prop_one_attempt',
        hue='Genotype',
        estimator='mean',
        errorbar='se',
        marker='o',
        linewidth=3,
        palette='deep'          # Bold colors
    )

    plt.title('Proportion of Single-Attempt Trials Over Time by Genotype')
    plt.xlabel('Month')
    plt.ylabel('Mean Proportion (± SE)')
    plt.xticks(range(1, 8))
    plt.grid(True)
    plt.legend(title='Genotype', loc='upper right')
    plt.tight_layout()
    plt.show()
    # sns.set(style="whitegrid")

    # # plt.figure(figsize=(8, 6))
    # sns.boxplot(x='Genotype', y='prop_one_attempt', data=data, palette='Set2', width=0.5)

    # sns.swarmplot(x='Genotype', y='prop_one_attempt', data=data, color='black', size=5)

    # plt.title('Proportion of 1-Attempt Trials by Genotype')
    # plt.ylabel('Proportion of Trials Completed in One Attempt')
    # plt.xlabel('Genotype')

    # plt.tight_layout()
    # plt.show()

# # Example data
# data = pd.DataFrame({
#     'value': [23, 45, 67, 89, 12, 34, 56, 78, 10, 30, 50, 70],
#     'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
# })

# # Example residuals
# residuals = np.random.normal(0, 1, 100)

# # Q-Q plot
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.show()

# # Perform one-way ANOVA
#     anova = pg.anova(data=data, dv='trials_one_attempt', between='Genotype')

#     print(anova)

    # kruskal_result = pg.kruskal(data=data, dv='prop_one_attempt', between='Genotype')

    # # Display the result
    # print(kruskal_result)


def main():
    ### data paths and wanted info
    file_path="C:/Users/ckill/Documents/neuroscience_sterf/AuerbachLab/FXS x TSC_archive.csv"
    file_info_path="C:/Users/ckill/Documents/neuroscience_sterf/AuerbachLab/FXS x TSC_data_exported_20250801.csv"
    wanted_columns_for_merge = ['date','UUID','weight','rat_ID','DOB','file_name','Genotype']
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
    graph_data = rat_info(delay_df,wanted_delay_interval,wanted_month)
    genotype_bargraph(graph_data) 

    ### program testing
    print(f'''
delay intervals: {delay_interval_list}
DOBs: {dob_list}
Genotypes: {gt_list}
total rats in df: {len(rat_ids)}
shared UUIDs: {len(shared_UUIDs)}
number of trials: {len(clean_df)}
rat data: {graph_data}
''')
    ### extra stuff
# clean df: {clean_df[['UUID','Attempts_to_complete','Delay (s)','weight','rat_ID']]}
# info df: {info_df[['date','DOB','Sex','weight','Genotype','UUID']]}
# clean_delay_date_df: {delay_df[['Delay (s)','Max Delay (s)','Min Delay (s)','UUID','complete_block_number','Attempts_to_complete','weight','rat_ID','date']]}

if __name__ == "__main__":
    main()
