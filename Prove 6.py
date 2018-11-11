from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.preprocessing import StandardScaler

le = preprocessing.LabelEncoder()

# Breast Cancer

X = cancer['data']
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print('The accuracy score for the Cancer data set is: ', accuracy_score(y_test, predictions))


# MPG

mpg_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
mpg = pd.read_csv(mpg_data, delim_whitespace=True)
mpg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'mYear', 'origin', 'carName']

for row in mpg:
    le.fit(mpg[row])
    mpg[row] = (le.transform(mpg[row]))

y = mpg['mpg']
X = mpg.iloc[:, 1:7]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

rgr = MLPRegressor()
rgr.fit(X_train, y_train)
predictions = rgr.predict(X_test)

print('The accuracy score for the MPG data set is: ', rgr.score(X_test, y_test))


# Forest Fires

fire = pd.read_csv('forestfires.csv')
fire.columns = ['x', 'y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']

#print(mpg)

for row in fire:
    le.fit(fire[row])
    fire[row] = (le.transform(fire[row]))

y = fire['area']
X = fire.iloc[:, 0:12]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
    beta_2=0.999, early_stopping=False, epsilon=1e-08,
    hidden_layer_sizes=(10, 10, 10), learning_rate='constant',
    learning_rate_init=0.001, max_iter=500, momentum=0.5,
    nesterovs_momentum=True, power_t=0.5, random_state=None,
    shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
    verbose=False, warm_start=False)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print('The accuracy score for the Forest Fire data set is: ', accuracy_score(y_test, predictions))


# student-por

student = pd.read_csv('student-por.csv', sep=';')
student.columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                   'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                   'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                   'absences', 'G1', 'G2', 'G3']

#print(mpg)

for row in student:
    le.fit(student[row])
    student[row] = (le.transform(student[row]))

y = student['G3']

for i in range(len(y)):
    if y[i] >= 10:
        y[i] = 1
    else:
        y[i] = 0

X = student.iloc[:, 0:31]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(200, 200, 200), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=.8,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
#
# print(y_test)
# print(predictions)

print(confusion_matrix(y_test, predictions))
print('The accuracy score for the Student data set is: ', accuracy_score(y_test, predictions))