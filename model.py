# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('heart_processed.csv')

X = dataset.drop(["Unnamed: 0","HeartDisease"], axis=1)

y = dataset['HeartDisease']

X = X.values
y = y.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LogisticRegression


regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(X_train,y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[40, 130, 214,0,108,1,0,1,1,0,1,1,1,0,0]]))