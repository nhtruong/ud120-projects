#!/usr/bin/python

import sys
sys.path.append("../tools/")
sys.path.append("../final_project/")
import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest as SKB
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

#----------------------------------------------------------
#   Load the dictionary containing the dataset
#----------------------------------------------------------

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#----------------------------------------------------------
#   Replace missing values with 0's
#----------------------------------------------------------

count_na = {}
for k in data_dict:
    for f in data_dict[k]:
        if f not in count_na:
            count_na[f] = 0
        if data_dict[k][f] == 'NaN':
            data_dict[k][f] = 0
            count_na[f] += 1
print "Number of Missing Values per Feature:"
for f in count_na:
    print count_na[f],f

#----------------------------------------------------------
#   Create new features
#----------------------------------------------------------
for k in data_dict:
    val = data_dict[k]
    data_dict[k]['grand_total'] = val['total_payments'] + val['total_stock_value']
    data_dict[k]['from_poi_ratio'] = \
        1. * val['from_poi_to_this_person'] / val['to_messages'] if val['to_messages'] > 0 else 0
    data_dict[k]['to_poi_ratio'] = \
        1. * + val['from_this_person_to_poi'] / val['from_messages'] if val['from_messages'] > 0 else 0
# Store to my_dataset for easy export below.
my_dataset = data_dict

#----------------------------------------------------------
#   Function That Builds a Classification Model
#----------------------------------------------------------

def build_model(selected_features, classifier, parameters, use_kbest = False, use_pca = False):
    # Extract features and labels from dataset
    data = featureFormat(my_dataset, selected_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # Add pipeline prefix for parameters
    parameters = {'classifier__'+k:parameters[k] for k in parameters}

    transformer = [('scaler',MMS())]

    if use_kbest:
        transformer += [('selector',SKB())]
        parameters['transformer__selector__k'] = range(1,len(selected_features)-1)

    if use_pca:
        transformer += [("pca",PCA())]
        parameters['transformer__pca__n_components'] = range(1,len(selected_features))

    transformer = FeatureUnion(transformer)
    pipeline = Pipeline([("transformer",transformer),("classifier",classifier)])
    grid = GridSearchCV(pipeline, parameters, scoring='f1', n_jobs=-1, cv=10)
    grid.fit(features, labels)
    clf = grid.best_estimator_

    print '\n-----------------------------------------------\n'
    test_classifier(clf,data_dict,selected_features)
    print '-----------------------------------------------\n'

    print 'Best f1-score:', grid.best_score_
    print 'Best parameters:', grid.best_params_, '\n'

    return clf

#----------------------------------------------------------
#   Compare the Performance of 3 Different Algorithms
#----------------------------------------------------------

features_list = ['poi',
                 'total_stock_value', 'total_payments', 'grand_total',
                 'from_this_person_to_poi','from_poi_to_this_person','shared_receipt_with_poi',
                 'from_poi_ratio','to_poi_ratio']

build_model(features_list, SVC(), {})
build_model(features_list, DTC(), {})
build_model(features_list, KNC(), {})

#----------------------------------------------------------
#   Decide which algorithm to use
#----------------------------------------------------------



features_list = ['poi',
                 'from_this_person_to_poi','from_poi_to_this_person','shared_receipt_with_poi',
                 'from_poi_ratio','to_poi_ratio']

features_list = ['poi',
                 'total_stock_value', 'total_payments', 'grand_total']

clf = build_model(features_list, DTC(),
                  {'criterion':['gini','entropy'],
                   'max_features':[None,'sqrt','log2'],
                   'max_depth':[1,2,3],
                   'min_samples_split':[2,3,4]},
                  use_kbest = False,
                  use_pca= True)


features_list = ['poi',
                 'shared_receipt_with_poi','from_poi_ratio','to_poi_ratio']

clf = build_model(features_list, DTC(),
                  {'criterion':['gini','entropy'],
                   'max_features':[None],
                   'max_depth':[1,2,3],
                   'min_samples_split':[2,3,4]},
                  use_kbest = False,
                  use_pca= True)



#----------------------------------------------------------
### Dump Classifier and Data
#----------------------------------------------------------


dump_classifier_and_data(clf, my_dataset, features_list)