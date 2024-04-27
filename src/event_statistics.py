import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''

    #Cpatient_id,event_id,event_description,timestamp,value
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

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''

    patient_ids = events['patient_id'].unique()
    dead_ids    = mortality['patient_id'].unique()
    alive_ids   = pd.Series(list(set(patient_ids) - set(dead_ids)))

    alive_events = events[events['patient_id'].isin(alive_ids)]
    dead_events =  events[events['patient_id'].isin(dead_ids)]

    alive_counts = alive_events.groupby('patient_id').event_id.count()
    dead_counts  = dead_events.groupby('patient_id').event_id.count()

    avg_dead_event_count = dead_counts.mean()
    max_dead_event_count = dead_counts.max()
    min_dead_event_count = dead_counts.min()
    avg_alive_event_count = alive_counts.mean()
    max_alive_event_count = alive_counts.max()
    min_alive_event_count = alive_counts.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''

    patient_ids = events['patient_id'].unique()
    dead_ids    = mortality['patient_id'].unique()
    alive_ids   = pd.Series(list(set(patient_ids) - set(dead_ids)))

    alive_events = events[events['patient_id'].isin(alive_ids)]
    dead_events =  events[events['patient_id'].isin(dead_ids)]

    encounter_labels = ['DRUG', 'LAB', 'DIAG']

    alive_encounters = alive_events[pd.Series(np.any([alive_events['event_id'].str.contains(x) for x in encounter_labels]), index=alive_events.index)]
    dead_encounters  = dead_events[pd.Series(np.any([dead_events['event_id'].str.contains(x) for x in encounter_labels]), index=dead_events.index)]

    alive_counts = alive_encounters.groupby('patient_id').apply(lambda x: x.timestamp.unique().size)
    dead_counts  = dead_encounters.groupby('patient_id').apply(lambda x: x.timestamp.unique().size)

    avg_dead_encounter_count = dead_counts.mean()
    max_dead_encounter_count = dead_counts.max()
    min_dead_encounter_count = dead_counts.min()
    avg_alive_encounter_count = alive_counts.mean()
    max_alive_encounter_count = alive_counts.max()
    min_alive_encounter_count = alive_counts.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''

    patient_ids = events['patient_id'].unique()
    dead_ids    = mortality['patient_id'].unique()
    alive_ids   = pd.Series(list(set(patient_ids) - set(dead_ids)))

    alive_events = events[events['patient_id'].isin(alive_ids)]
    dead_events =  events[events['patient_id'].isin(dead_ids)]

    alive_lengths =  alive_events.groupby('patient_id').apply(lambda x: (x.timestamp.iloc[-1] - x.timestamp.iloc[0]).days)
    dead_lengths  =  dead_events.groupby('patient_id').apply(lambda x: (x.timestamp.iloc[-1] - x.timestamp.iloc[0]).days)

    avg_dead_rec_len = dead_lengths.mean()
    max_dead_rec_len = dead_lengths.max()
    min_dead_rec_len = dead_lengths.min()
    avg_alive_rec_len = alive_lengths.mean()
    max_alive_rec_len = alive_lengths.max()
    min_alive_rec_len = alive_lengths.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
