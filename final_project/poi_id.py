#!/usr/bin/python

import sys

sys.path.append("../tools/")
sys.path.append("../final_project/")
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

# ----------------------------------------------------------
#   Load the dictionary containing the dataset
# ----------------------------------------------------------

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

F_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
              'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
              'long_term_incentive', 'restricted_stock', 'director_fees']

E_Features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
              'shared_receipt_with_poi']

# ----------------------------------------------------------
#   Replace missing values with 0's
# ----------------------------------------------------------

count_na = {}
for k in data_dict:
    for f in data_dict[k]:
        if f not in count_na:
            count_na[f] = 0
        if data_dict[k][f] == 'NaN':
            data_dict[k][f] = 0
            count_na[f] += 1

na_counts = zip(count_na.values(), count_na.keys())
na_counts = sorted(na_counts, key=lambda x: -x[0])
print "Number of Missing Values per Feature:"
for f in na_counts:
    print f[0], f[1]

# ----------------------------------------------------------
#   Handling Outliers
# ----------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np


def plot_feature(name):
    data = [data_dict[k][name] for k in data_dict if data_dict[k][name] <> 0]
    position = np.array(range(0, len(data)))
    labels = [k.title() for k in data_dict if data_dict[k][name] <> 0]

    plt.clf()
    plt.bar(position, data)
    plt.xticks(position + 0.6, labels, rotation='vertical', fontsize=10)
    plt.title(name)
    plt.show()


# Uncomment the following lines to view the plots.

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

# ----------------------------------------------------------
#   Create new features
# ----------------------------------------------------------

for k in data_dict:
    val = data_dict[k]
    data_dict[k]['grand_total'] = val['total_payments'] + val['total_stock_value']
    data_dict[k]['from_poi_ratio'] = \
        1. * val['from_poi_to_this_person'] / val['to_messages'] if val['to_messages'] > 0 else 0
    data_dict[k]['to_poi_ratio'] = \
        1. * + val['from_this_person_to_poi'] / val['from_messages'] if val['from_messages'] > 0 else 0

# select all numerical features
all_features = ['poi'] + [k for k in data_dict[data_dict.keys()[0]] if k not in ['poi', 'email_address']]

# ----------------------------------------------------------
#   (Function) Validate a Classification Model
# ----------------------------------------------------------

from sklearn.cross_validation import cross_val_score
from sklearn import metrics

def validate_model(model, features, labels):
    accuracy = cross_val_score(model, features, labels, scoring='accuracy', cv=4).mean()
    precision = cross_val_score(model, features, labels, scoring='precision', cv=4).mean()
    recall = cross_val_score(model, features, labels, scoring='recall', cv=4).mean()
    f1 = cross_val_score(model, features, labels, scoring='f1', cv=4).mean()
    print "\n(METRICS) Accuracy: {:.3f}   Precision: {:.3f}   Recall: {:.3f}   F1-Score: {:.3f}".\
        format(accuracy,precision, recall, f1)


# ----------------------------------------------------------
#   Function) Build a Classification Model
# ----------------------------------------------------------

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion


def build_model(selected_features, classifier, parameters={},
                use_scaler=True, use_pca=False, n=None, use_kbest=False, k=None):

    feature_range = range(1, len(selected_features))

    # Extract features and labels from dataset
    data = featureFormat(data_dict, selected_features, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    # Add Prefix for classifier parameters
    parameters = {'classifier__' + k: parameters[k] for k in parameters}

    # Transform Features
    transformer = []
    if use_scaler or use_pca:
        transformer += [('scaler', MinMaxScaler())]
    if use_kbest:
        transformer += [('kbest', SelectKBest())]
        parameters['transformer__kbest__k'] = feature_range if k is None else k
    if use_pca:
        transformer += [("pca", PCA())]
        parameters['transformer__pca__n_components'] = feature_range if n is None else n

    # Build Pipeline
    if len(transformer) <> 0:
        transformer = FeatureUnion(transformer)
        pipeline = Pipeline([("transformer", transformer), ("classifier", classifier)])
    else:
        pipeline = Pipeline([("classifier", classifier)])

    # Search for best fit:
    grid = GridSearchCV(pipeline, parameters, scoring='f1', n_jobs=-1, cv=4)
    grid.fit(features, labels)
    clf = grid.best_estimator_
    print 'Best Parameters:'
    for k in grid.best_params_:
        print k, ":", grid.best_params_[k]
    validate_model(clf, features, labels)

    if use_kbest:
        best_features = get_best_features(features, labels, selected_features,
                                          k=grid.best_params_['transformer__kbest__k'])
    else:
        best_features = selected_features

    return clf, best_features


def get_best_features(features, labels, selected_features, k=None):
    if k is None or k == 'all':
        k = len(features[0])
    skb = SelectKBest(k='all')
    skb.fit(features, labels)
    scores = zip(skb.scores_, selected_features[1:])
    scores = sorted(scores, key=lambda s: -s[0])
    print "\nSelected Features:"
    best_features = []
    for i in range(0, k):
        print " {:6.2f} ".format(scores[i][0]), scores[i][1]
        best_features += [scores[i][1]]
    print
    return ['poi'] + best_features


# ----------------------------------------------------------
#   Compare the Performance of 3 Different Algorithms
# ----------------------------------------------------------

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC

'''

#### NOTE: Each of the 3 statements below could take 1 to 5 minutes to run.
#### I commented them out so that you can execute this file quickly.

# Support Vector Machine
clf, features_list = build_model(all_features, SVC(),
                                 {'C': [pow(10,i) for i in range(1,10)]},
                                 use_kbest=True,
                                 use_scaler=True)

# Decision Tree
clf, features_list = build_model(all_features, DTC(max_features=None),
                                 {'criterion': ['gini', 'entropy'],
                                  'max_depth': [2, 3, 4, 5, 6, 7],
                                  'min_samples_split': [2, 3, 4]},
                                 use_kbest=True,
                                 use_scaler=False)

# Nearest Neighbors
clf, features_list = build_model(all_features, KNC(),
                                 {'n_neighbors': [2, 3, 4, 5],
                                  'weights': ['uniform', 'distance'],
                                  'leaf_size': [2, 3, 4, 5, 6],
                                  'p': [1, 2, 3]},
                                 use_kbest=True,
                                 use_scaler=True)
'''

# The best estimator found
estimator = KNC(n_neighbors=3, weights='uniform', leaf_size=2, p=3)

clf, features_list = build_model(all_features, estimator, {},
                                 use_kbest=True, k=[14],
                                 use_scaler=True)

# ----------------------------------------------------------
#   Assess New Features
# ----------------------------------------------------------

original_features = [f for f in features_list if f not in ['grand_total','to_poi_ratio','from_poi_ratio']]

# Without new features
_ = build_model(original_features, estimator, {}, use_kbest=True, k=['all'], use_scaler=True)

# With grand_total
_ = build_model(original_features + ['grand_total'], estimator, {}, use_kbest=True, k=['all'], use_scaler=True)

# With from_poi_ratio
_ = build_model(original_features + ['from_poi_ratio'], estimator, {}, use_kbest=True, k=['all'], use_scaler=True)

# With to_poi_ratio
_ = build_model(original_features + ['to_poi_ratio'], estimator, {}, use_kbest=True, k=['all'], use_scaler=True)


# ----------------------------------------------------------
#   Final Model
# ----------------------------------------------------------

final_model, final_features = build_model(original_features + ['grand_total'],
                                          estimator, {},
                                          use_kbest=True,
                                          use_scaler=True)

test_classifier(final_model, data_dict, final_features, folds=1000)

# ----------------------------------------------------------
#    Dump Classifier and Data
# ----------------------------------------------------------

dump_classifier_and_data(final_model, data_dict, final_features)
