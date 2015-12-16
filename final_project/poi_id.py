#!/usr/bin/python

import sys
sys.path.append("../tools/")
sys.path.append("../final_project/")
import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

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

#----------------------------------------------------------
#   Function That Builds a Classification Model
#----------------------------------------------------------

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

def build_model(selected_features, classifier, parameters={}, scoring = 'f1', use_kbest = False, use_pca = False):
    # Extract features and labels from dataset
    data = featureFormat(data_dict, selected_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # Add pipeline prefix for parameters
    parameters = {'classifier__'+k:parameters[k] for k in parameters}

    transformer = []

    if use_kbest:
        transformer += [('kbest',SelectKBest())]
        parameters['transformer__kbest__k'] = range(1,len(selected_features)-1)

    if use_pca:
        transformer += [('scaler',MinMaxScaler()),("pca",PCA())]
        parameters['transformer__pca__n_components'] = range(1,len(selected_features))

    if len(transformer) <> 0:
        transformer = FeatureUnion(transformer)
        pipeline = Pipeline([("transformer",transformer),("classifier",classifier)])
    else:
        pipeline = Pipeline([("classifier",classifier)])

    grid = GridSearchCV(pipeline, parameters, scoring=scoring, n_jobs=-1, cv=10)
    grid.fit(features, labels)
    clf = grid.best_estimator_

    print '\n-----------------------------------------------\n'
    test_classifier(clf,data_dict,selected_features)
    print '-----------------------------------------------\n'

    print 'Best '+scoring+'-score:', grid.best_score_
    print 'Best Parameters:', grid.best_params_, '\n'

    return clf

#----------------------------------------------------------
#   Compare the Performance of 3 Different Algorithms
#----------------------------------------------------------

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC

# select all numerical features
all_features = ['poi']+[k for k in data_dict[data_dict.keys()[0]] if k not in ['poi','email_address']]

build_model(all_features, SVC())
build_model(all_features, DTC())
build_model(all_features, KNC())

# Decicion Tree takes the cake here!!!

#----------------------------------------------------------
#   Select Features
#----------------------------------------------------------

data = featureFormat(data_dict, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Calculate Feature Scores:
skb = SelectKBest(k='all')
skb.fit(features,labels)
scores = zip(skb.scores_, all_features[1:])
print "Feature Scores:"
for x in sorted(scores, key=lambda x: -x[0]):
    print x


# Pick the top features that are not dominated by NA values
features_list = ['poi',
                 'to_poi_ratio', 'shared_receipt_with_poi', 'from_poi_to_this_person',
                 'from_poi_ratio','from_this_person_to_poi','to_messages']

# Use SelectKBest in GridSearchCV to find the best number of features
clf1 = build_model(features_list, DTC(), {}, use_kbest = True)

# Best number of features is 2
features_list = ['poi',
                 'to_poi_ratio', 'shared_receipt_with_poi']


#----------------------------------------------------------
#   Tuning Parameters for the Decision Tree
#----------------------------------------------------------


# With PCA
clf2 = build_model(features_list, DTC(max_features=None),
                  {'criterion':['gini','entropy'],
                   'max_depth':[1,2,3],
                   'min_samples_split':[2,3,4]},
                  use_pca= True)

# Without PCA
clf3 = build_model(features_list, DTC(max_features=None),
                  {'criterion':['gini','entropy'],
                   'max_depth':[1,2,3],
                   'min_samples_split':[2,3,4]})



#----------------------------------------------------------
#    Dump Classifier and Data
#----------------------------------------------------------
clf = clf2
my_dataset = data_dict
dump_classifier_and_data(clf, my_dataset, features_list)