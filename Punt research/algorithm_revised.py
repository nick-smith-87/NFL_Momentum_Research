import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import warnings
warnings.filterwarnings("ignore")
'''
Nick Smith
Algorithm for determing whether a team should go for a blocked punt or a big return
Developed for Sports Analytics and Research Club at Johns Hopkins Fall 2022
Project Members: Nick Smith, Owen Hartman, Drew Amunategui
'''

##reading in data and assigning to certain data points
data = input("Enter each data value with a space in between: \n precipitation level (0: none, 1: moderate, 2: heavy)\n subfreezing temperatures 15mph+ wind (0:neither, 1: one of them, 2:both)\n field position \n")
data_array = data.split(' ')
precipitation = int(data_array[0])
cold_wind = int(data_array[1])
field_position = int(data_array[2])

row = [field_position, precipitation, cold_wind]

df = pd.read_csv("combinedblockandreturn.csv")
df = df.drop(['Unnamed: 0'], axis = 1)
X = df.drop(['Blocked'], axis = 1)
y = df['Blocked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)

model = LogisticRegression(random_state=0, multi_class='multinomial', 
                           solver='newton-cg').fit(X_train, y_train)
model.fit(X, y)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs=-1)

# predict the class label
yhat = model.predict([row])
# summarize the predicted class
print('Predicted Class: %d' % yhat[0])
#predict probability
yhat1 = model.predict_proba([row])
#predict 
#print('Predicted Probabilities: %s' % yhat1[0])
blocked = ""
prob = 0
if (yhat[0] == 0):
    blocked = "won't be blocked"
    prob = yhat1[0][0]
    percent = prob * 100
else:
    blocked = "will be blocked"
    prob = yhat1[0][1]
    percent = prob * 100
    
    
print("Our model predicts with %.2f%%" % percent, "certainity that the punt", blocked)


