import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean
import numpy as np

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
    #TODO:First get the train indices and test indices for each iteration
    #Then train the classifier accordingly
    #Report the mean accuracy and mean auc of all the folds
    acc_array = []

    kf = KFold(k)
    for train, test in kf.split(X):
        X_train, Y_train = (X[train, :], Y[train])
        X_test, Y_test = (X[test, :], Y[test])

        Y_pred = models_partc.logistic_regression_pred(X_train, Y_train, X_test)
        acc, auc, precision, recall, f1score = models_partc.classification_metrics(Y_pred, Y_test)
        acc_array.append([acc, auc])

    acc_array = np.array(acc_array)
    return np.mean(acc_array[:, 0]), np.mean(acc_array[:, 1])


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
    #TODO: First get the train indices and test indices for each iteration
    #Then train the classifier accordingly
    #Report the mean accuracy and mean auc of all the iterations
    
    acc_array = []

    rs = ShuffleSplit(n_splits=iterNo, test_size=test_percent, random_state=0)
    for train, test in rs.split(X):
        X_train, Y_train = (X[train, :], Y[train])
        X_test, Y_test = (X[test, :], Y[test])

        Y_pred = models_partc.logistic_regression_pred(X_train, Y_train, X_test)
        acc, auc, precision, recall, f1score = models_partc.classification_metrics(Y_pred, Y_test)
        acc_array.append([acc, auc])

    acc_array = np.array(acc_array)
    return np.mean(acc_array[:, 0]), np.mean(acc_array[:, 1])


def main():
    X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    print("Classifier: Logistic Regression__________")
    acc_k,auc_k = get_acc_auc_kfold(X,Y)
    print(("Average Accuracy in KFold CV: "+str(acc_k)))
    print(("Average AUC in KFold CV: "+str(auc_k)))
    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
    print(("Average Accuracy in Randomised CV: "+str(acc_r)))
    print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
    main()

