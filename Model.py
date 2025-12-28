################################################################################
############################### INITIALIZE #####################################
################################################################################
import numpy as np
import pandas as pd
import os
import warnings
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# Remove warning iteritems in Pool
warnings.simplefilter(action='ignore', category=FutureWarning)

# Warning "A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead" removing:
pd.options.mode.chained_assignment = None

pd.set_option('display.max_rows', 50, 'display.max_columns', None)

# Read data
print(os.listdir('./DATA'))
train = pd.read_csv('./DATA/Train.csv')
test = pd.read_csv('./DATA/OoT.csv')

################################################################################
################################# FEATURE ######################################
############################### ENGINEERING ####################################
################################################################################
# Feature types
Features = train.dtypes.reset_index()
Categorical = Features.loc[Features[0] == 'object', 'index']

del Features

# Categorical to the beginning
cols = train.columns.tolist()
pos = 0
for col in Categorical:
    cols.insert(pos, cols.pop(cols.index(col)))
    pos += 1
train = train[cols]
test = test[cols]

del col, cols, pos

# 1) Missings
################################################################################
# Function to print columns with at least n_miss missings
def miss(ds, n_miss):
    tot = len(ds)
    for col in list(ds):
        miss = ds[col].isna().sum()
        if miss>= n_miss:
            print(col,' ',str(miss),' (',round(miss/tot*100,1), '%)', sep='')

# Which columns have 1 missing at least...
print('\n################## TRAIN ##################')
miss(train, 1)
print('\n################## TEST ##################')
miss(test, 1)

# Missings in categorical features (fix it with an "NA" string)
for col in Categorical:
    train.loc[train[col].isna(), col] = 'NA'
    test.loc[test[col].isna(), col] = 'NA'

del col

# 2) Correlations
################################################################################
# Let's see if certain columns are correlated
# or even that are the same with a "shift"
thresholdCorrelation = 0.999

def InspectCorrelated(df):
    corrMatrix = df.corr().abs()  # Correlation Matrix
    upperMatrix = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))
    correlColumns = []
    for col in upperMatrix.columns:
        correls = upperMatrix.loc[upperMatrix[col] > thresholdCorrelation, col].keys()
        if len(correls) >= 1:
            correlColumns.append(col)
            print('\n', col, '->', end=' ')
            for i in correls:
                print(i, end=' ')
    print('\nSelected columns to drop:\n', correlColumns)
    return correlColumns

# Look at correlations in the original features
correlColumns = InspectCorrelated(train.iloc[:, len(Categorical):-1])

# If we are ok, throw them:
train = train.drop(correlColumns, axis=1)
test = test.drop(correlColumns, axis=1)

del correlColumns, thresholdCorrelation

# 3) Constants
################################################################################
# Let's see if there is some constant column:
def InspectConstant(df):
    consColumns = []
    for col in list(df):
        if len(df[col].unique()) < 2:
            print(df[col].dtypes, '\t', col, len(df[col].unique()))
            consColumns.append(col)
    print('\nSelected columns to drop:\n', consColumns)
    return consColumns

consColumns = InspectConstant(train.iloc[:, len(Categorical):-1])

# If we are ok, throw them:
train = train.drop(consColumns, axis=1)
test = test.drop(consColumns, axis=1)

del consColumns

################################################################################
################################ MODEL CATBOOST ################################
################################################################################
pred = list(train)[1:-1]
X_train = train[pred].reset_index(drop=True)
Y_train = train['TARGET'].reset_index(drop=True)
X_test = test[pred].reset_index(drop=True)
Y_test = test['TARGET'].reset_index(drop=True)

# 1) Train with OoT set as overfitting detector
################################################################################
def print_gini(Pred, TARGET):
    print(round(2 * roc_auc_score(TARGET, Pred) - 1, 4))

RS = 1234  # Seed for partitions (train/test) and model random part
esr = 100  # Early stopping rounds (when validation does not improve in these rounds, stops)

# Categorical positions for catboost
Pos = list()
As_Categorical = Categorical.tolist()
As_Categorical.remove('ID')
for col in As_Categorical:
    Pos.append((X_train.columns.get_loc(col)))

# To Pool Class (for catboost only)
pool_train = Pool(X_train, Y_train, cat_features=Pos)
pool_test = Pool(X_test, Y_test, cat_features=Pos)

# "By-hand" hyper-parameter tuning. A grid-search is expensive
# We test different combinations
# See hyper-parameter options here:
# "https://catboost.ai/en/docs/references/training-parameters/"
model_catboost = CatBoostClassifier(
    eval_metric='AUC', # Evaluation metric in validation set to stop training
    iterations=2000,  # High value, to be sure to find the optimum
    od_type='Iter',  # Overfitting detector set to "iterations" (number of trees)
    random_seed=RS,  # Random seed for reproducibility
    verbose=100)  # Shows train/test metric every "verbose" trees

# Training hyper-parameters of the model:
params = {'objective': 'Logloss',
          'learning_rate': 0.05,
          'depth': 5,
          'min_data_in_leaf': 50,
          'l2_leaf_reg': 15,
          'rsm': 0.7,
          'subsample': 0.7,
          'random_seed': RS}

model_catboost.set_params(**params)

print('\nCatboost Fit...\n')
model_catboost.fit(X=pool_train,
                   eval_set=pool_test,
                   early_stopping_rounds=esr)

# Raw Prediction of the model in test
test['Pred'] = model_catboost.predict_proba(X_test)[:, 1]

print('Final Model Gini on OoT:')
print_gini(test['Pred'], Y_test)

del As_Categorical, Categorical, col, esr
del RS, Pos, X_train, Y_train, X_test, Y_test, pred, pool_train, pool_test, params

################################################################################
################################# CALIBRATION ##################################
################################################################################
def lodds(p):
    return np.log(p / (1 - p))

def sigmoid(s, const, beta):
    return 1 / (1 + np.exp(-(beta * s + const)))

print('\nOoT Prediction CT:', round(test['Pred'].mean(), 4))
print('OoT Real CT:', round(test['TARGET'].mean(), 4))

###################
target_ct = 0.08
###################
print('OoT Target CT:', target_ct)

# 1) OoT - Train Platt scaling
################################################################################
lr = LogisticRegression(penalty=None)
lr.fit(pd.DataFrame(lodds(test['Pred'])), test['TARGET'])
beta_calib = lr.coef_[0][0]
const_calib = lr.intercept_[0]

test['p_calib'] = sigmoid(lodds(test['Pred']), const_calib, beta_calib)

del lr

# 2) OoT - Bayes (central trend correction)
################################################################################
test['p_calib_bayes'] = test['p_calib']
w = target_ct / test['p_calib_bayes'].mean()
n_iterations = 30
for i in range(n_iterations):
    test['p_calib_bayes'] = test['p_calib_bayes'] / (test['p_calib_bayes'] + (1 - test['p_calib_bayes']) / w)
    w = target_ct / test['p_calib_bayes'].mean()

del w, n_iterations, i

# 3) OoT Final calibration regression
################################################################################
lin = LinearRegression()
lin.fit(pd.DataFrame(lodds(test['Pred'])), lodds(test['p_calib_bayes']))
beta_calib2 = lin.coef_[0]
const_calib2 = lin.intercept_

# We can check that the output of the regression produces the same probabilities
# test['p_calib_bayes_check'] = sigmoid(lodds(test['Pred']), const_calib2, beta_calib2)
# test.drop(columns=['p_calib_bayes_check'], inplace=True)

del lin, LinearRegression

print('Corrected OoT Prediction CT:', round(test['p_calib_bayes'].mean(), 4))

del const_calib, beta_calib, target_ct

print('\nFinal Calibration Beta:', beta_calib2)
print('Final Calibration Constant:', const_calib2)

################################################################################
################################ APPENDIX: MS ##################################
################################################################################
###################
x_0 = 0.0015
step = 2
###################

master_scale = list()
while x_0 < 1:
    master_scale.append(x_0)
    x_0 = step * x_0

master_scale.insert(0, 0)
master_scale.append(1)

print('\nMaster Scale in', len(master_scale) - 1, 'RATING CLASSES:\n')
print('RC 1\t', round(master_scale[0], 4), ' \t\t -->\t ', round(master_scale[1], 4))
for i in range(1, len(master_scale) - 1):
    print('RC', i + 1, '\t', round(master_scale[i], 4), ' \t -->\t ', round(master_scale[i + 1], 4))

del i, x_0, step

MS = pd.DataFrame(
    {'RATING CLASS': ['RC '+str(i) for i in range(1,len(master_scale))],
     'PD min': master_scale[:-1],
     'PD max': master_scale[1:]
     })

del master_scale

################################################################################
################################ FILE OUTPUTS ##################################
################################################################################
# Predicted probabilities to .csv
test[['ID','TARGET','Pred','p_calib','p_calib_bayes']].to_csv('Output/Model_Outputs.csv', index=False)
# Save model artifact to make predictions
model_catboost.save_model('./Output/catboost_classifier')
# Save Master Scale
MS.to_csv('Output/Master_Scale.csv', index=False)
