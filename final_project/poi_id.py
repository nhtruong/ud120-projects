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

F_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
              'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
              'long_term_incentive', 'restricted_stock', 'director_fees']

E_Features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
              'shared_receipt_with_poi']

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

na_counts = zip(count_na.values(), count_na.keys())
na_counts = sorted(na_counts, key=lambda x:-x[0])
print "Number of Missing Values per Feature:"
for f in na_counts:
    print f[0],f[1]


#----------------------------------------------------------
#   Handling Outliers
#----------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np


def plot_feature(name):
    data = [data_dict[k][name] for k in data_dict if data_dict[k][name] <> 0]
    position = np.array(range(0,len(data)))
    labels = [k.title() for k in data_dict if data_dict[k][name] <> 0]

    plt.clf()
    plt.bar(position,data)
    plt.xticks( position+0.6, labels, rotation='vertical',fontsize=10)
    plt.title(name)
    plt.show()

# plot_feature('salary')
# plot_feature('total_stock_value')
# plot_feature('total_payments')
#
# plot_feature('to_messages')
# plot_feature('from_messages')
# plot_feature('shared_receipt_with_poi')

# Remove outlier
data_dict.pop('TOTAL')

# Check the data again
# plot_feature('salary')
# plot_feature('total_stock_value')
# plot_feature('total_payments')

plt.close()

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
#   (Function) Validate a Classification Model
#----------------------------------------------------------

from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score


def validate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    precision = precision_score(y_test,preds)
    recall = recall_score(y_test,preds)
    f1 = f1_score(y_test,preds)
    print "(VALIDATION) Precision: {:.2}   Recall: {:.2}   F1-Score: {:.2}".format(precision,recall,f1)

#----------------------------------------------------------
#   Function) Build a Classification Model
#----------------------------------------------------------

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion


def build_model(selected_features, classifier, parameters={}, use_scaler = False, use_kbest = False, use_pca = False):

    feature_count = len(selected_features) - 1

    # Extract features and labels from dataset
    data = featureFormat(data_dict, selected_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    X_train, X_test, y_train, y_test = train_test_split(features,labels,train_size=0.7,random_state=10)

    # Add Prefix for classifier parameters
    parameters = {'classifier__'+k:parameters[k] for k in parameters}

    # Transform Features
    transformer = []
    if use_scaler or use_pca:
        transformer += [('scaler',MinMaxScaler())]
    if use_kbest:
        transformer += [('kbest',SelectKBest())]
        parameters['transformer__kbest__k'] = range(1,feature_count+1)

    if use_pca:
        transformer += [("pca",PCA())]
        parameters['transformer__pca__n_components'] = range(1,feature_count+1)

    # Build Pipeline
    if len(transformer) <> 0:
        transformer = FeatureUnion(transformer)
        pipeline = Pipeline([("transformer",transformer),("classifier",classifier)])
    else:
        pipeline = Pipeline([("classifier",classifier)])

    # Search for best fit:
    grid = GridSearchCV(pipeline, parameters, scoring='f1', n_jobs=-1, cv=10)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print 'Best Parameters:'
    for k in grid.best_params_:
        print k,":",grid.best_params_[k]
    validate_model(clf,X_test,y_test)

    if use_kbest:
        best_features = get_best_features(X_train, y_train, k = grid.best_params_['transformer__kbest__k'])
    else:
        best_features = selected_features

    return clf, best_features


def get_best_features(features,labels,k=None):
    if k is None:
        k = len(features[0])
    print k
    skb = SelectKBest(k='all')
    skb.fit(features,labels)
    scores = zip(skb.scores_, all_features[1:])
    scores = sorted(scores, key=lambda s: -s[0])
    print "\nBest Features:"
    best_features = []
    for i in range(0,k):
        print scores[i][0], scores[i][1]
        best_features += [scores[i][1]]
    print
    return ['poi'] + best_features

#----------------------------------------------------------
#   Compare the Performance of 3 Different Algorithms
#----------------------------------------------------------

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC

# select all numerical features
all_features = ['poi']+[k for k in data_dict[data_dict.keys()[0]] if k not in ['poi','email_address']]

build_model(all_features, SVC(), use_scaler=True)
build_model(all_features, KNC(), use_scaler=True)
build_model(all_features, DTC())

# Decicion Tree takes the cake here!!!

#----------------------------------------------------------
#   Select Features
#----------------------------------------------------------

_ , features_list = build_model(all_features, DTC(), use_kbest=True)

#----------------------------------------------------------
#   Tuning Parameters for the Decision Tree
#----------------------------------------------------------

clf, features_list = build_model(features_list, DTC(max_features=None),
                                  {'criterion':['gini','entropy'],
                                   'max_depth':[2,3,4,5,6],
                                   'min_samples_split':[2,3]},
                                 use_kbest=True)

clf, _ = build_model(features_list, DTC(max_features=None),
                                  {'criterion':['gini','entropy'],
                                   'max_depth':[2,3,4,5,6],
                                   'min_samples_split':[2,3]},
                                 use_pca=True)



test_classifier(clf, data_dict, features_list, folds = 1000)

#----------------------------------------------------------
#    Dump Classifier and Data
#----------------------------------------------------------
dump_classifier_and_data(clf, data_dict, features_list)