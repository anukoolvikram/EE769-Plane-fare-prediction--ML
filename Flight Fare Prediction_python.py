

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

train_data=pd.read_excel('/content/Data_Train.xlsx')

#ot display all the columns
pd.set_option('display.max_columns', None)

train_data.head()

train_data.info()

train_data.shape

#calculating the no of similar duration data
train_data['Duration'].value_counts()

#dropping the nan values
train_data.dropna(inplace=True)

train_data.shape

"""No nan values"""

train_data.isnull().sum()

"""EDA : Exploratory Data Analysis"""

train_data['Journey_day']=pd.to_datetime(train_data.Date_of_Journey, format='%d/%m/%Y').dt.day

train_data['Journey_month']=pd.to_datetime(train_data['Date_of_Journey'], format='%d/%m/%Y').dt.month

"""Now we have extracted features like day and month from the given data

"""

train_data.head()

#since we have acquired data of month and date, so we can drop the column of Data of Journey
train_data.drop(['Date_of_Journey'],axis=1, inplace=True)

#extracting values like departure hour and departure minute from the departure time

train_data['Dep_hour']=pd.to_datetime(train_data['Dep_Time'],format='%H:%M').dt.hour

train_data['Dep_min']=pd.to_datetime(train_data['Dep_Time'],format='%H:%M').dt.minute

#now removing the column for dep_time
train_data.drop(['Dep_Time'],axis=1, inplace=True)

train_data.head()

#now extracting arrival hour and arrival min from arrival time
train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time'], format='mixed').dt.hour
train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time'], format='mixed').dt.minute

#dropping the arrival time column from the data
train_data.drop(['Arrival_Time'], axis=1, inplace=True)

train_data.head()

#time taken by plane to reach destination is Duration

duration=list(train_data['Duration'])

for i in range(len(duration)):
  if len(duration[i].split())!=2: #check if duration contains only hour or mins
    if "h" in duration[i]:
      duration[i]=duration[i]+' 0m' #adds 0 minute
    else:
      duration[i]='0h'+duration[i] #adds 0 hour

duration_hours = []
duration_mins = []

for dur in duration:
    # Split the duration string into segments
    segments = dur.split()
    hours, mins = 0, 0
    # Iterate over segments to extract hours and minutes
    for segment in segments:
        if 'h' in segment:
            hours = int(segment.split('h')[0])
        elif 'm' in segment:
            mins = int(segment.split('m')[0])
    duration_hours.append(hours)
    duration_mins.append(mins)

#adding the columns of duration hours and duration minutes

train_data['Duration_hours']=duration_hours
train_data['Duration_mins']=duration_mins

train_data.head()

#dropping the duration column

train_data.drop(['Duration'], axis=1, inplace=True)

train_data.head()

"""Handing Categorical Data"""

train_data['Airline'].value_counts()

#dropping the jet airways since it is not working any more

train_data = train_data[train_data['Airline'] != 'Jet Airways']
train_data['Airline'].value_counts()

train_data = train_data[train_data['Airline'] != 'Jet Airways Business']
train_data['Airline'].value_counts()

#plotting the airline vs price graph

sns.catplot(y='Price', x='Airline', data=train_data.sort_values('Price', ascending=False), kind='boxen', height=6, aspect=3 )

#as airline is Nominal categorical data we will perform onehotEncoding

Airline=train_data[['Airline']]
Airline=pd.get_dummies(Airline, drop_first=True)
Airline.head()

#finding unique values from the source

train_data['Source'].value_counts()

#plotting source vs price

sns.catplot(y = "Price", x = "Source", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)
plt.show()

"""Some of the outliers are present in Banglore"""

#As source is Nominal Categorical data we will perform OneHotEncoding
Source=train_data[['Source']]
Source=pd.get_dummies(Source, drop_first=True)

Source.head()

train_data['Destination'].value_counts()

#As destination is Nominal Categorical data we will perform OneHotEncoding

Destination=train_data[['Destination']]
Destination=pd.get_dummies(Destination, drop_first=True)

Destination.head()

train_data['Route']

#additional info contains almost 80% no_info
#route and total_stops are related to each other

train_data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

train_data.head()

train_data['Total_Stops'].value_counts()

#as this is case of ordinal categorical type we perform labelencoder
#here values are assigned with corresponding keys

train_data.replace({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4 }, inplace=True)

train_data.head()

#concatenate dataframe -->train_data + Airline + Source + Destination
#combining all the datasets

data_train=pd.concat([train_data, Airline, Source, Destination], axis=1)

data_train.head()

data_train.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)

data_train.head()

data_train.shape

"""**Test Set**"""

test_data=pd.read_excel('/content/Test_set.xlsx')

test_data.head()

# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)

data_test.head()

"""**Feature Selection**"""

data_train.shape

data_train.columns

X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()

#now taking our dependent feature

y=data_train.iloc[:,1]
y.head()

#finding correlation between Independent and dependent attributes

# Select only numeric columns
numeric_columns = train_data.select_dtypes(include=['int64', 'float64'])

# Generate correlation matrix and plot heatmap
plt.figure(figsize=(18, 18))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='RdYlGn')
plt.show()

"""green side : highly correlated

red side: negative correlated
"""

#important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection=ExtraTreesRegressor()

selection.fit(X,y)

print(selection.feature_importances_)

#plotting the graph of feature importances for better visualization

plt.figure(figsize=(12,8))
feat_importances=pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

"""**Fitting model using Random Forest**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)

y_pred = reg_rf.predict(X_test)

reg_rf.score(X_train, y_train)

reg_rf.score(X_test, y_test)

sns.histplot(y_test-y_pred)
plt.show()

#scatter plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Y_test')
plt.ylabel('y_pred')
plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

metrics.r2_score(y_test, y_pred)

"""**Hyperparameter Tuning**"""

from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Random search of parameters, using 5 fold cross validation,
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)

rf_random.fit(X_train, y_train)

rf_random.best_params_

prediction=rf_random.predict(X_test)

plt.figure(figsize=(8,8))
sns.histplot(y_test-prediction)
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(y_test, prediction, alpha=0.5)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

"""**Save the model to reuse it again**"""

import pickle
# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)

model=open('/content/flight_rf.pkl', 'rb')
forest=pickle.load(model)

y_prediction=forest.predict(X_test)

metrics.r2_score(y_test, y_prediction)