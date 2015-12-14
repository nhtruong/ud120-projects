#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

sys.path.append("../final_project/")
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_payments','total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers and Handle NA values

for k in data_dict:
    for f in data_dict[k]:
        if data_dict[k][f] == 'NaN':
            data_dict[k][f] = 0


### Task 3: Create new feature(s)
my_feature_list = ['grand_total','from_poi_ratio','to_poi_ratio']
features_list += my_feature_list

for k in data_dict:
    val = data_dict[k]
    data_dict[k]['grand_total'] = val['total_payments'] + val['total_stock_value']
    data_dict[k]['from_poi_ratio'] = \
        1. * val['from_poi_to_this_person'] / val['to_messages'] if val['to_messages'] > 0 else 0
    data_dict[k]['to_poi_ratio'] = \
        1. * + val['from_this_person_to_poi'] / val['from_messages'] if val['from_messages'] > 0 else 0



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

import sklearn.metrics as met
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.3,random_state=10)
no_features = len(features_list)-1

scl =  MinMaxScaler()
pca = PCA(n_components=no_features*9/10)

for c in [GNB(),DTC(),SVC()]:
    clf = make_pipeline(scl,pca,c)
    print '-----------------------'
    test_classifier(clf,data_dict,features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

trans = FeatureUnion([("scale",MinMaxScaler()),("pca",PCA())])
grid = GridSearchCV(Pipeline([("transformer",trans),("classifier",DTC())]),
                    {'transformer__pca__n_components':range(1,no_features+1),
                     'classifier__criterion':['gini','entropy'],
                     'classifier__max_depth':[1,2,3,4,5,6,7],
                     'classifier__min_samples_split':[2,3,4]},
                    scoring='f1',
                    n_jobs=-1,
                    cv=10)
grid.fit(features, labels)
clf = grid.best_estimator_

print grid.best_score_
print grid.best_params_
print

test_classifier(clf,data_dict,features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)