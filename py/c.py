import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import GridSearchCV, cross_val_score

df = pd.read_excel("C:\\Users\\L.A\\Desktop\\adult_income.xlsx")

le = preprocessing.LabelEncoder()
df['workclass'] = le.fit_transform(df['workclass'])
df['education'] = le.fit_transform(df['education'])
df['marital-status'] = le.fit_transform(df['marital-status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['sex'] = le.fit_transform(df['sex'])
df['native-country'] = le.fit_transform(df['native-country'])

x = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  # targets
train_x = x.iloc[:24130]
train_y = y.iloc[:24130]
test_x = x.iloc[24130:]
test_y = y.iloc[24130:]

# decision tree classification
clf = DecisionTreeClassifier(random_state= 1)
clf = clf.fit(train_x, train_y)

pred_y = clf.predict(test_x)

print("Accuracy for Decision Tree before optimizing : ", metrics.accuracy_score(test_y, pred_y))

param_grid = {'criterion': ['gini'],
              'max_depth': [4, 6, 8, 10, 12, 14, 16],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
              }

gsDCT = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
gsDCT.fit(x, y)

print(gsDCT.best_estimator_)

pred_y = gsDCT.predict(test_x)

print("Accuracy for Decision Tree after optimizing : ", metrics.accuracy_score(test_y, pred_y))

# RandomForest Classifier
# Create the model with 100 trees
rfc = RandomForestClassifier(n_estimators=100, random_state=1, max_features = 'sqrt', n_jobs=-1, verbose = 1)

rfc.fit(train_x, train_y) #fitting
n_nodes = []
max_depths = []

for ind_tree in rfc.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average max depth {int(np.mean(max_depths))}')

train_rf_predictions = rfc.predict(train_x)
rf_predictions = rfc.predict(test_x)

print("Accuracy for Random Forest before optimizing : ",accuracy_score(test_y,rf_predictions))

param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini']
}

rfc_gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
rfc_gs.fit(train_x, train_y)
y_pred=rfc_gs.predict(test_x)
print("Accuracy for Random Forest after optimizing: ",accuracy_score(test_y, y_pred))


print("\n************************* Regression *************************\n")

# decision tree regression
x = df.loc[:, ["age","workclass", "marital-status", "relationship","race" , "sex","capital-gain","capital-loss", "hours-per-week", "native-country"]]  # Features
y = df.loc[:, "education-num"]  # targets
train_x = x.iloc[:24130]
train_y = y.iloc[:24130]
test_x = x.iloc[24130:]
test_y = y.iloc[24130:]

rgrsr = DecisionTreeRegressor(random_state= 1)
rgrsr = rgrsr.fit(train_x, train_y)

y_pred = rgrsr.predict(test_x)
print("MSE for Decision Tree before optimization:", metrics.mean_squared_error(test_y, y_pred))

param_grid = {'criterion': ['mse'],
              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
              }


# Create a grid search object
gsDCT = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
gsDCT.fit(train_x, train_y) # fitting the gs
print(gsDCT.best_estimator_)

y_pred = gsDCT.predict(test_x)
# model MSE
print("MSE for Decision Tree after optimization:", metrics.mean_squared_error(test_y, y_pred))

# random forest regression

rfr = RandomForestRegressor(random_state= 1)
rfr = rfr.fit(train_x, train_y)
y_pred = rfr.predict(test_x)
print("MSE for Random Forest before optimization:", metrics.mean_squared_error(test_y, y_pred))

param_grid = {'criterion': ['mse'],
              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
              }


# Create a grid search object
rf_gs = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
rf_gs.fit(train_x, train_y) # fitting the gs

y_pred = rf_gs.predict(test_x)
# model MSE
print("MSE for Random Forest after optimization:", metrics.mean_squared_error(test_y, y_pred))

