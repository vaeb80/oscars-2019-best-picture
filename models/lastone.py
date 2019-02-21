import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.958927738927739
exported_pipeline = make_pipeline(
    Normalizer(norm="l2"),
    StackingEstimator(estimator=LinearSVC(C=0.001, dual=True, loss="hinge", penalty="l2", tol=0.1)),
    Normalizer(norm="l1"),
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    StandardScaler(),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.2, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
