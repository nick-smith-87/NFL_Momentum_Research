from dash import Dash, html, dcc
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import warnings

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv("combinedblockandreturn.csv")
df = df.drop(['Unnamed: 0'], axis = 1)
X = df.drop(['Blocked'], axis = 1)
y = df['Blocked']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
model = LogisticRegression(random_state=0, multi_class='multinomial', 
                           solver='newton-cg').fit(X, y)
#model.fit(X, y)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs=-1)

app.layout = html.Div(children=[
    html.H1(children='Punt Block analyzer',
           style = {
               'textAlign' : 'center',
               'background' : 'lightblue',
               'margin-left' : '-1em',
               'margin-right' : '-1em',           
           }),

   dcc.Dropdown(
        options=[
            {'label': 'No Precipitation', 'value': 'no_rain'},
            {'label': 'Light Precipitation', 'value': 'moderate_rain'},
            {'label': 'Moderate Precipitation', 'value': 'heavy_rain'}
        ],
        placeholder="Select Precipitation level",
        id='precipitation-dropdown',
        searchable=False,
        style=dict(
            width="50%",
        )),
    dcc.Dropdown(
        options=[
            {'label': 'No Extreme Weather', 'value': 'no_weather'},
            {'label': 'Subfreezing or 15+mph wind', 'value': 'one_weather'},
            {'label': 'Both subfreezing and 15+mph wind', 'value': 'both_weather'}
        ],
        placeholder="Select Weather",
        id='weather-dropdown',
        searchable=False,
        style=dict(
            width="50%",
        )),
    dcc.Input(
            id="input-yardline",
            type="number",
            placeholder="Input yardline 1-50",
            min=1, max=50,
        ),
    
    
        html.Div(id = 'result'),
])
   
@app.callback(
    Output('result', 'children'),
    [Input('input-yardline', 'value'),
    Input('precipitation-dropdown', 'value'),
    Input('weather-dropdown', 'value'),
    ]
)
def show_result(yardline_value, precipitation_value, weather_value):
    if yardline_value is None or precipitation_value is None or weather_value is None:
        return ""

    try:
        row = [1, 1, 1]
        if precipitation_value == "no_rain":
            precipitation_value = 0
        elif precipitation_value == "moderate_rain":
            precipitation_value = 1
        else:
            precipitation_value = 2

        if weather_value == "no_weather":
            weather_value = 0
        elif weather_value == "one_weather":
            weather_value = 1
        else:
            weather_value = 2
            
        row = [yardline_value, precipitation_value, weather_value]
        yhat1 = model.predict_proba([row])
        blocked = ""
        prob = 0
        if (yhat1[0][0] > yhat1[0][1]):
            blocked = "won't be blocked"
            prob = yhat1[0][0]
            percent = prob * 100
        else:
            blocked = "will be blocked"
            prob = yhat1[0][1]
            percent = prob * 100
       
        str_result = "Our model predicts with %.2f%% " % percent, "certainty that the punt ", blocked
        
        pie_chart = np.array([yhat1[0][0], yhat1[0][1]])
        mylabels = ["Probability punt won't be blocked", "Probability it will be blocked"]
        fig = plt.pie(pie_chart, labels = mylabels)
    
        return html.Div([
            html.Div(str_result),
            html.Div([dcc.graph(id = "g1", figure=fig)]),
                    ])
    
    
    except Exception as e:
        print(e)
        return ["An error occurred. Please check your input values."]




if __name__ == '__main__':
    app.run_server(debug=True)