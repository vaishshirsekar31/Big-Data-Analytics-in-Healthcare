import utils
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    try:
        events = pd.read_csv(filepath + 'events.csv', parse_dates=['timestamp'])
        events = events.sort_values('timestamp')
    except IOError:
        events = None

    #patient_id,timestamp,label
    try:
        mortality = pd.read_csv(filepath + 'mortality_events.csv', parse_dates=['timestamp'])
        mortality = mortality.sort_values('timestamp')
    except IOError:
        mortality = None

    #Columns in event_feature_map.csv - idx,event_id
    try:
        feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
    except IOError:
        feature_map = None

    return events, mortality, feature_map

def agg_events(df):

    event_name = df['event_id'].iloc[0]
    
    if 'LAB' in event_name:
        return df['event_id'].count()

    elif 'DIAG' in event_name or 'DRUG' in event_name:
        return df['value'].sum()

    else:
        print("Event name wrong!")


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''

    patient_ids = events['patient_id'].unique()
    dead_ids    = mortality['patient_id'].unique()
    alive_ids   = pd.Series(list(set(patient_ids) - set(dead_ids)))

    alive_events = events[events['patient_id'].isin(alive_ids)]
    dead_events =  events[events['patient_id'].isin(dead_ids)]

    def compute_index_dead(df):
        patientTimeStamp = mortality[mortality['patient_id'] == df.iloc[0, :]['patient_id']].iloc[0, :].timestamp
        patientTimeStamp = patientTimeStamp - pd.Timedelta(days=30)
        return patientTimeStamp

    def compute_index_alive(df):
        patientTimeStamp = df.iloc[-1, :].timestamp
        return patientTimeStamp

    dead_indexes = pd.DataFrame(dead_events.groupby('patient_id').apply(compute_index_dead))

    alive_indexes = pd.DataFrame(alive_events.groupby('patient_id').apply(compute_index_alive))

    indx_date = alive_indexes.append(dead_indexes).sort_index()
    indx_date = indx_date.reset_index()
    indx_date.columns = ['patient_id', 'indx_date']

    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', index=False)

    return indx_date

def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''

    indx_date = indx_date.set_index('patient_id')
    events_join_indx = events.join(indx_date, on='patient_id', how='outer')

    observation_window = events_join_indx['indx_date'] - events_join_indx['timestamp']
    mask = np.logical_and(observation_window >= pd.Timedelta(days=0), observation_window <= pd.Timedelta(days=2000))

    filtered_events = events_join_indx[mask]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''

    # 1. Replace event_id's with index available in event_feature_map.csv

    joined_df = filtered_events_df.join(feature_map_df.set_index('event_id'), on='event_id')
    filtered_events_df['feature_id'] = joined_df['idx']

    # 2. Remove events with n/a values
    filtered_events_df = filtered_events_df.dropna()

    #3. Aggregate events using sum and count to calculate feature value

    aggregated_events = filtered_events_df.groupby(['patient_id', 'event_id', 'feature_id'])


    lab_scores = filtered_events_df[filtered_events_df['event_id'].str.contains('LAB')]
    lab_scores = lab_scores.groupby(['patient_id', 'event_id', 'feature_id'])['patient_id'].count()

    no_lab_scores = filtered_events_df[np.logical_or(filtered_events_df['event_id'].str.contains('DIAG'), 
        filtered_events_df['event_id'].str.contains('DRUG'))]

    no_lab_scores = no_lab_scores.groupby(['patient_id', 'event_id', 'feature_id'])['value'].sum()

    aggregated_events = pd.concat([no_lab_scores, lab_scores]).reset_index()
    aggregated_events.columns = ['patient_id', 'event_id', 'feature_id', 'feature_value']
    aggregated_events = aggregated_events[['patient_id', 'feature_id', 'feature_value']]

    #4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)

    pivoted_df = aggregated_events.pivot(index='patient_id', columns='feature_id', values='feature_value')
    normed_df = pivoted_df / pivoted_df.max()
    normed_df = normed_df.reset_index()
    aggregated_events = pd.melt(normed_df, id_vars='patient_id', value_name='feature_value').dropna()

    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''

    patientID_tuple = aggregated_events.groupby('patient_id').apply(lambda x: list(x.sort_values('feature_id').apply(lambda y: (y['feature_id'], y['feature_value']), axis=1)))
    patient_features = patientID_tuple.to_dict()

    mortality_labels = [(id, int(id in list(mortality['patient_id']))) for id in list(aggregated_events['patient_id'].unique())]
    mortality = dict(mortality_labels)

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    #deliverable1.write(bytes((""),'UTF-8')); #Use 'UTF-8'
    #deliverable2.write(bytes((""),'UTF-8'));

    for patients in patient_features:

        features = patient_features[patients]

        features = pd.DataFrame(features).sort_values(0)

        features = features.values.tolist()

        deliverable1.write(bytes(("{} {} \n".format(mortality.get(patients, 0), utils.bag_to_svmlight(features))),'UTF-8'))
        deliverable2.write(bytes(("{} {} {} \n".format(int(patients), mortality.get(patients, 0), utils.bag_to_svmlight(features))),'UTF-8'))

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()