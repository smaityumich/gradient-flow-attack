import numpy as np
from adult_data import preprocess_adult_data
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity, TruePositiveRateDifference, ErrorRateRatio, EqualizedOdds
from metrics import group_metrics
constraints = {'TPRD': TruePositiveRateDifference,
               'ERR': ErrorRateRatio,
               'DP': DemographicParity,
               'EO': EqualizedOdds}

# Adult data processing
seed = 17
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)

all_train, all_test = dataset_orig_train.features, dataset_orig_test.features
y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')

x_train = np.delete(all_train, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)
x_test = np.delete(all_test, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)

group_train = dataset_orig_train.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]
group_test = dataset_orig_test.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]
group_train_cross = group_train[:,0] + group_train[:,1]*2
group_test_cross = group_test[:,0] + group_test[:,1]*2

## Train reductions
eps = 0.01
c = 'EO'
constraint = constraints[c]()
classifier = LogisticRegression(solver='liblinear', fit_intercept=True)
mitigator = ExponentiatedGradient(classifier, constraint, eps=eps)
mitigator.fit(x_train, y_train, sensitive_features=group_train_cross)
y_pred_mitigated = mitigator.predict(x_test)
print('\nFair on all test')
_ = group_metrics(y_test, y_pred_mitigated, group_test[:,0], label_protected=0, label_good=1)


ens_weights = []
coefs = []
intercepts = []

for t, w_t in enumerate(mitigator._weights.index):
    if mitigator._weights[w_t] > 0:
        coefs.append(mitigator._predictors[t].coef_.flatten())
        intercepts.append(mitigator._predictors[t].intercept_[0])
        ens_weights.append(mitigator._weights[w_t])
