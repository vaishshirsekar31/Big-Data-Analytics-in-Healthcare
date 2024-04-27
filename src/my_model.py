import utils
import etl
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import *


#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def classification_metrics(Y_pred, Y_true):
    #TODO: Calculate the above mentioned metrics
    #NOTE: It is important to provide the output in the same order
    return (accuracy_score(Y_true, Y_pred),
            roc_auc_score(Y_true, Y_pred),
            precision_score(Y_true, Y_pred),
            recall_score(Y_true, Y_pred),
            f1_score(Y_true, Y_pred))

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
    print("______________________________________________")
    print(("Classifier: "+classifierName))
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
    print(("Accuracy: "+str(acc)))
    print(("AUC: "+str(auc_)))
    print(("Precision: "+str(precision)))
    print(("Recall: "+str(recall)))
    print(("F1-score: "+str(f1score)))
    print("______________________________________________")
    print("")

def aggregate_test_events(filtered_events_df, feature_map_df):
    
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

    return aggregated_events

def generate_kaggle_submission(svmlight_with_ids_file, Y_pred):
    f = open(svmlight_with_ids_file)
    lines = f.readlines()
    target = open('../my_kaggle_predictions.csv', 'w')
    target.write("%s,%s\n" %("patient_id","label"))

    print(len(Y_pred))
    print(len(lines))
    for i in range(len(lines)):

        #print(str(lines[i].split()[0]))
        #print(str(Y_pred[i]))
        target.write("%s,%s\n" %(str(lines[i].split()[0]),str(Y_pred[i])))

def my_features():

    '''
    You may generate your own features over here.
    Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
    IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
    patient_id followed by a space and the corresponding feature in sparse format.
    Eg of a line:
    60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
    Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

    Save the file as "test_features.txt" and save it inside the folder deliverables

    input:
    output: X_train,Y_train,X_test
    '''


    events_train, mortality_train, feature_map = etl.read_csv('../data/train/')

    patient_features_train, mortality = etl.create_features(events_train, mortality_train, feature_map)
    etl.save_svmlight(patient_features_train, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')
    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")

    try:
        events_test = pd.read_csv('../data/test/' + 'events.csv', parse_dates=['timestamp'])
        events_test = events_test.sort_values('timestamp')
    except IOError:
        events_test = None

    aggregated_events_test = aggregate_test_events(events_test, feature_map)

    patientID_test_tuple  = aggregated_events_test.groupby('patient_id').apply(lambda x: list(x.sort_values('feature_id').apply(lambda y: (y['feature_id'], y['feature_value']), axis=1)))
    patient_features_test = patientID_test_tuple.to_dict()

    deliverable1 = open("../features_svmlight.test", 'wb')
    deliverable2 = open("../deliverables/test_features.txt", 'wb')
    
    #deliverable1.write(bytes((""),'UTF-8')); #Use 'UTF-8'
    #deliverable2.write(bytes((""),'UTF-8'));

    for patients in patient_features_test:

        features = patient_features_test[patients]

        features = pd.DataFrame(features).sort_values(0)

        features = features.values.tolist()

        deliverable1.write(bytes(("{} {} \n".format(str(1), utils.bag_to_svmlight(features))),'UTF-8'))
        deliverable2.write(bytes(("{} {} \n".format(int(patients), utils.bag_to_svmlight(features))),'UTF-8'))

    deliverable1.close()
    deliverable2.close()

    print("Number of Patients in patient_features_test")
    print(len(patient_features_test))

    X_test, _ = utils.get_data_from_svmlight("../features_svmlight.test")

    print("Dim of X_test")
    print(X_test.shape)

    #TODO: complete this
    return X_train, Y_train, X_test

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
    #TODO: complete this

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 3000, num = 30)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 150, num = 15)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6, 8, 10, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6, 8, 10, 20]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    clf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = clf, 
        param_distributions = random_grid, 
        n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, Y_train)
    best_clf = rf_random.best_estimator_
    #

    # Evaluation Train set

    display_metrics("My classifier Train: ", best_clf.predict(X_train), Y_train)

    # Evaluation Validation set

    X_validation, Y_validation = utils.get_data_from_svmlight("../data/features_svmlight.validate")
    display_metrics("My classifier Validation: ", best_clf.predict(X_validation), Y_validation)

    # Soft label for Kaggle

    print("Dim of X_test for clf")
    print(X_test.shape)

    print("Dim of Y_test for clf")
    print(best_clf.predict_proba(X_test)[:,1].shape)

    generate_kaggle_submission("../deliverables/test_features.txt", best_clf.predict_proba(X_test)[:,1])

    return best_clf.predict(X_test).astype(int)

def main():
    X_train, Y_train, X_test = my_features()
    Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
    utils.generate_submission("../deliverables/test_features.txt",Y_pred)
    #The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

    