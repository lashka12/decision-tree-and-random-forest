import math
from sklearn import preprocessing
from a import RandomForestRegressor
from a import DecisionTreeClassifier
from a import DecisionTreeRegressor
from a import RandomForestClassifier
from a import GridSearchCV
from a import Calculation
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np


# reading data file
print()
print('uploading data file ... ')
print()
start = time.time()
df = pd.read_excel("C:\\Users\\L.A\\Desktop\\adult_income.xlsx")
end = time.time()
print('data was uploaded successfully , ' + "%.2f" % (end - start) + ' seconds')
print()
print('---------------------------------------------------')

# preparing data for the algorithm , and convert string values
le = preprocessing.LabelEncoder()
n_unique_values_treshold = 15
for feature in df.columns:
    if feature != df.columns[len(df.columns) - 1]:
        unique_values = df[feature].unique()
        example_value = unique_values[0]

        if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
            df[feature] = le.fit_transform(df[feature])

################################################ Tree Classification #################################################

train_df = df.iloc[:19034]
val_df = df.iloc[19035:24130]
test_df = df.iloc[24130:]

print()
print('starting decision tree classifier on the data ... ')
print()

start = time.time()
tree = DecisionTreeClassifier(data=train_df, max_depth=2, min_samples_split=5)
end = time.time()

real_vals = test_df.iloc[:, -1].values

c = 0
for i in range(len(test_df)):
    if real_vals[i] == tree.predict(test_df.iloc[i]):
        c += 1

print('accuracy : ' + str(c / len(test_df)))
print('time : ' + "%.2f" % (end - start) + ' seconds')
print()
print('tuning parameters to optimize model ... ')

param_grid = {
    'max_depth': [4, 6, 8, 10, 12, 14, 16],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
}

gsDCT = GridSearchCV(DecisionTreeClassifier(), train_data=train_df, validation_data=val_df, param_grid=param_grid,
                     scoring='accuracy')

print('best estimation : ' + str(gsDCT.best_estimation))
print()
print('building new tree model by the best parameters ( train+validation )...')

train_df = df.iloc[:24130]

best_depth = gsDCT.best_estimation.get('max_depth')
best_min_split = gsDCT.best_estimation.get('min_samples_split')

start = time.time()
tree = DecisionTreeClassifier(data=train_df, max_depth=best_depth, min_samples_split=best_min_split)
end = time.time()

real_vals = test_df.iloc[:, -1].values

c = 0
for i in range(len(test_df)):
    if real_vals[i] == tree.predict(test_df.iloc[i]):
        c += 1

print('accuracy : ' + str(c / len(test_df)))
print('time : ' + "%.2f" % (end - start) + ' sec')
print()
print('---------------------------------------------------')

######################################## Random Forest Classification #################################################

print()
print('starting random forest classifier on the data ... ')
print()

max_features = int(np.ceil(math.sqrt(len(df.columns))))
start = time.time()
forest = RandomForestClassifier(train_df, n_estimators=2, max_depth=3, max_features=max_features, max_samples=len(train_df),
                                min_samples_split=5)

end = time.time()

real_vals = test_df.iloc[:, -1].values
c = 0
for i in range(len(test_df)):
    if real_vals[i] == forest.predict(test_df.iloc[i]):
        c += 1

print('accuracy : ' + str(c / len(test_df)))
print('time : ' + "%.2f" % (end - start) + ' seconds')
print()
print('tuning parameters to optimize model ... ')

param_grid = {
    'n_estimators': [15],
    'max_depth': [15],
    'min_samples_split': [2, 4]
}

start = time.time()
gsDCT = GridSearchCV(RandomForestClassifier(), train_data=train_df, validation_data=val_df, param_grid=param_grid,
                     scoring='accuracy')
end = time.time()

print('optimizing time : ' + "%.2f" % ((end - start)/60) + ' min')
print('best estimation : ' + str(gsDCT.best_estimation))
print()
print('building new forest model by the best parameters ( train + validation ) ...')
print()

train_df = df.iloc[:24130]

best_depth = gsDCT.best_estimation.get('max_depth')
best_min_split = gsDCT.best_estimation.get('min_samples_split')
best_n_estimators = gsDCT.best_estimation.get('n_estimators')

start = time.time()
forest = RandomForestClassifier(train_df, n_estimators=best_n_estimators, max_depth=best_depth,
                                max_features=max_features, max_samples=len(train_df),
                                min_samples_split=best_min_split)

end = time.time()
real_vals = test_df.iloc[:, -1].values
pred = []
for i in range(len(test_df)): pred.append(forest.predict(test_df.iloc[i]))
acc = Calculation.accuracy(values=real_vals, predictions=pred)
print('accuracy : ' + str(acc))
print('time : ' + "%.2f" % (end - start) + ' seconds')
print()
print('---------------------------------------------------')

################################################ Tree Regression #################################################

df["label"] = df[['education-num']]
df = df.drop('>50K', axis=1)
df = df.drop('education', axis=1)
df = df.drop('occupation', axis=1)
df = df.drop('education-num', axis=1)

train_df = df.iloc[:19035]
val_df = df.iloc[19035:24130]
test_df = df.iloc[24130:]

print()
print('starting decision tree regression on the data ... ')
print()

start = time.time()
tree = DecisionTreeRegressor(data=train_df, max_depth=2, min_samples_split=7)
end = time.time()

pred = []
vals = test_df.iloc[:, -1].values
for i in range(len(test_df)): pred.append(tree.predict(test_df.iloc[i]))

print('MSE : ' + str(Calculation.mse(test_df.iloc[:, -1].values, pred)))
print('time : ' + "%.2f" % (end - start) + ' seconds')
print()
print('tuning parameters to optimize model ... ')
print()

param_grid = {  # 8 and 9 are the best
    'max_depth': [2,  7, 8],
    'min_samples_split': [3, 4, 8, 9]
}

gsDCT = GridSearchCV(DecisionTreeRegressor(), train_data=train_df, validation_data=val_df, param_grid=param_grid,
                     scoring='mse')

print('best estimation : ' + str(gsDCT.best_estimation))
print('building new tree model with the best parameters (train+validation) ...')
print()

train_df = df.iloc[:24130]

best_depth = gsDCT.best_estimation.get('max_depth')
best_min_split = gsDCT.best_estimation.get('min_samples_split')

start = time.time()
tree = DecisionTreeRegressor(data=train_df, max_depth=best_depth, min_samples_split=best_min_split)
end = time.time()

pred = []
vals = test_df.iloc[:, -1].values
for i in range(len(test_df)): pred.append(tree.predict(test_df.iloc[i]))

print('MSE : ' + str(Calculation.mse(test_df.iloc[:, -1].values, pred)))
print('time : ' + "%.2f" % (end - start) + ' seconds')
print()
print('---------------------------------------------------')

####################################### Random Forest Regression #################################################



first = time.time()
print('starting random forest regression on the data ... ')
print()

max_features = int(np.ceil(math.sqrt(len(df.columns))))

start = time.time()
forest = RandomForestRegressor(train_df, n_estimators=2, max_depth=3,
                               max_features=max_features, max_samples=len(train_df),
                               min_samples_split=4)

end = time.time()


real_vals = test_df.iloc[:, -1].values
pred = []

for i in range(len(real_vals)): pred.append(forest.predict(test_df.iloc[i]))
mse = Calculation.mse(real=test_df.iloc[:, -1].values, predictions=pred)

print('MSE : ' + str(mse))
print('time : ' + "%.2f" % (end - start) + ' seconds')
print()
print('tuning parameters to optimize model ... ')
print()



param_grid = {
    'n_estimators': [2, 19],
    'max_depth': [12, 13],
    'min_samples_split': [6, 7]
}

start = time.time()
gsDCT = GridSearchCV(RandomForestRegressor(), train_data=train_df, validation_data=val_df, param_grid=param_grid,
                     scoring='mse')
end = time.time()

print('optimizing time : ' + "%.2f" % ((end - start) / 60) + ' min')
print('best estimation : ' + str(gsDCT.best_estimation))
print()
print('building new forest model by the best parameters ( train + validation ) ...')
print()

train_df = df.iloc[:24130]

best_depth = gsDCT.best_estimation.get('max_depth')
best_min_split = gsDCT.best_estimation.get('min_samples_split')
best_n_estimators = gsDCT.best_estimation.get('n_estimators')

start = time.time()
forest = RandomForestRegressor(train_df, n_estimators=best_n_estimators, max_depth=best_depth,
                               max_features=max_features, max_samples=len(train_df),
                               min_samples_split=best_min_split)

end = time.time()
real_vals = test_df.iloc[:, -1].values
pred = []

for i in range(len(real_vals)): pred.append(forest.predict(test_df.iloc[i]))

mse = Calculation.mse(real=test_df.iloc[:, -1].values, predictions=pred)

print('mse : ' + str(mse))
print('time : ' + "%.2f" % (end - start) + ' seconds')


A = real_vals[:120]
B = pred[:120]
l1, = plt.plot(A, 'b')
l2, = plt.plot(B, 'r')
plt.legend(['real values', 'predicted values'], loc='upper left')
plt.show()
