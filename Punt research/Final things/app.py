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
import plotly.tools as tls
import plotly.graph_objects as go
import warnings

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv("combinedblockandreturn.csv")
df = df.drop(['Unnamed: 0'], axis = 1)
X = df.drop(['Blocked'], axis = 1)
y = df['Blocked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
model = LogisticRegression(random_state=0, multi_class='multinomial', 
                           solver='newton-cg').fit(X_train, y_train)
#model.fit(X, y)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X_test, y_test, scoring = 'accuracy', cv = cv, n_jobs=-1)

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
    dcc.Input(
            id="input-seconds-left",
            type="number",
            placeholder="Input time left (seconds)",
            min=0, max=3600,
         ),
    dcc.Input(
            id="input-deficit",
            type="number",
            placeholder="input deficit (0 if winning)",
            min=0, max=35,
         ),
    dcc.Dropdown(
        options=[
            {'label': 'True', 'value': 'true'},
            {'label': 'False', 'value': 'false'},
        ],
        placeholder="Top 5 Returner?",
        id='t5returner-dropdown',
        searchable=False,
        style=dict(
            width="50%",
        )),
    dcc.Dropdown(
        options=[
            {'label': 'True', 'value': 'true'},
            {'label': 'False', 'value': 'false'},
        ],
        placeholder="Top 5 Punter?",
        id='t5punter-dropdown',
        searchable=False,
        style=dict(
            width="50%",
        )),
    dcc.Dropdown(
        options=[
            {'label': 'True', 'value': 'true'},
            {'label': 'False', 'value': 'false'},
        ],
        placeholder="Facing Elite QB?",
        id='QB-against-dropdown',
        searchable=False,
        style=dict(
            width="50%",
        )),
    dcc.Dropdown(
        options=[
            {'label': 'True', 'value': 'true'},
            {'label': 'False', 'value': 'false'},
        ],
        placeholder="Have an Elite QB?",
        id='QB-for-dropdown',
        searchable=False,
        style=dict(
            width="50%",
        )),
    dcc.Dropdown(
        options=[
            {'label': 'True', 'value': 'true'},
            {'label': 'False', 'value': 'false'},
        ],
        placeholder="Injuries to opposing Specialists?",
        id='spec-dropdown',
        searchable=False,
        style=dict(
            width="50%",
        )),
    dcc.Dropdown(
        options=[
            {'label': 'True', 'value': 'true'},
            {'label': 'False', 'value': 'false'},
        ],
        placeholder="Change to the personal protector?",
        id='pp-dropdown',
        searchable=False,
        style=dict(
            width="50%",
        )),
    
    
    
    
    
    
        html.Div(id = 'result'),
])
   
@app.callback(
    Output('result', 'children'),
    [Input('input-yardline', 'value'),
    Input('precipitation-dropdown', 'value'),
    Input('weather-dropdown', 'value'),
    Input('input-seconds-left', 'value'),
    Input('input-deficit', 'value'),
    Input('t5returner-dropdown', 'value'),
    Input('t5punter-dropdown', 'value'),
    Input('QB-against-dropdown', 'value'),
    Input('QB-for-dropdown', 'value'),
    Input('spec-dropdown', 'value'),
    Input('pp-dropdown', 'value')
    ]
)
def show_result(yardline_value, precipitation_value, weather_value, time_left, deficit, t5ret, t5punter, elite_qb_against, elite_qb_for, injured_spec, protection_change):
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
            
        str_to_return = "Non Quantifiable Factors of Importance: \n"
        possesions_needed = deficit//8
        if(yardline_value < 5):
            str_to_return += "-Backed up Punt improves chances of a block \n"
        if(precipitation_value == 2):
            str_to_return += "-High Precipitation levels, make a punt return difficult \n"
        if(weather_value == 2):
            str_to_return += "-High Winds and Freezing temperaturs make a punt return difficult \n"
        if(time_left < possesions_needed * 120):
            str_to_return += "-Minimal time left so block may be neccessary \n"
        if(t5ret):
            str_to_return += "-Top 5 returner sways more towards going for a return \n"
        if(t5punter):
            str_to_return += "-Top 5 Punter sways more towards going for a block or safe return \n"
        if(elite_qb_against == 0):
            str_to_return += "-Risk of block not worth giving the ball back to other teams QB \n"
        if(elite_qb_for):
            str_to_return += "-Elite QB can make up for the need of a punt block \n"
        if(injured_spec):
            str_to_return += "-Slower operation time likely, block more favorable \n"
        if(protection_change):
            str_to_return += "-Personal Protector change can cause for miscommunication in the protection \n"
        
        str_result = "Our model predicts with %.2f%% " % percent, "certainty that the punt ", blocked
        
        values = [yhat1[0][0], yhat1[0][1]]
        labels = ["Probability punt won't be blocked", "Probability it will be blocked"]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

    
        return html.Div([
            html.Div("If you commit to a punt block, and send six or more rushers, this is what our algorithm predicts"),
            html.Div(str_result),
            html.Div([dcc.Graph(id = "g1", figure=fig)]),
            html.Div(str_to_return, style={"white-space": "pre"}),
        ])
    
    
    except Exception as e:
        print(e)
        return ["An error occurred. Please check your input values."]




if __name__ == '__main__':
    app.run_server(debug=True)