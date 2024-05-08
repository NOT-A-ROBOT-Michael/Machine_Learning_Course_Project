import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd

model = joblib.load('../artifacts/final_model_dt.pkl')

coefficients = model.named_steps["randomforestclassifier"].feature_importances_
features = model.named_steps['randomforestclassifier'].feature_names_in_

feat_imp = pd.Series(
    np.exp(coefficients), index=features
).sort_values(ascending=True)

def check_eligibility(*args):
    return "Eligible" if model.predict() > 2 else "Not Eligible"

gender_options = [{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}]
marital_status_options = [{'label': 'Single', 'value': 'Single'}, {'label': 'Married', 'value': 'Married'}]
education_options = [{'label': 'Graduate', 'value': 'Graduate'}, {'label': 'Not Graduate', 'value': 'Not Graduate'}]
credit_history_options = [{'label': 'Good', 'value': 'Good'}, {'label': 'Bad', 'value': 'Bad'}]
self_employed_options = [{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}]
property_area_options = [{'label': 'Urban', 'value': 'Urban'}, {'label': 'Semiurban', 'value': 'Semiurban'}, {'label': 'Rural', 'value': 'Rural'}]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Loan Eligibility Checker", className='title'),
    html.Div([
        html.Div([
            html.Label("Gender", className='label'),
            dcc.Dropdown(
                id='gender-dropdown',
                options=gender_options,
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Marital Status", className='label'),
            dcc.Dropdown(
                id='marital-status-dropdown',
                options=marital_status_options,
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Education", className='label'),
            dcc.Dropdown(
                id='education-dropdown',
                options=education_options,
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Number of Dependents", className='label'),
            dcc.Input(
                id='dependents-input',
                type='number',
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Income", className='label'),
            dcc.Input(
                id='income-input',
                type='number',
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Loan Amount", className='label'),
            dcc.Input(
                id='loan-amount-input',
                type='number',
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Credit History", className='label'),
            dcc.Dropdown(
                id='credit-history-dropdown',
                options=credit_history_options,
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Self Employed", className='label'),
            dcc.Dropdown(
                id='self-employed-dropdown',
                options=self_employed_options,
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Coapplicant Income", className='label'),
            dcc.Input(
                id='coapplicant-income-input',
                type='number',
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Loan Amount Term", className='label'),
            dcc.Input(
                id='loan-amount-term-input',
                type='number',
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Label("Property Area", className='label'),
            dcc.Dropdown(
                id='property-area-dropdown',
                options=property_area_options,
                value=None,
                className='form-control'
            )
        ], className='form-group'),
        html.Div([
            html.Button('Check Eligibility', id='submit-button', n_clicks=0, className='btn btn-primary mt-3'),
        ], className='form-group'),
        html.Div(id='output-container-button', className='output')
    ], style={'width': '50%', 'margin': 'auto', 'padding': '40px', 'background-color': '#f59e0b'})
])

@app.callback(
    Output('output-container-button', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('gender-dropdown', 'value'),
     dash.dependencies.State('marital-status-dropdown', 'value'),
     dash.dependencies.State('education-dropdown', 'value'),
     dash.dependencies.State('dependents-input', 'value'),
     dash.dependencies.State('income-input', 'value'),
     dash.dependencies.State('loan-amount-input', 'value'),
     dash.dependencies.State('credit-history-dropdown', 'value'),
     dash.dependencies.State('self-employed-dropdown', 'value'),
     dash.dependencies.State('coapplicant-income-input', 'value'),
     dash.dependencies.State('loan-amount-term-input', 'value'),
     dash.dependencies.State('property-area-dropdown', 'value')]
)
def update_output(n_clicks, gender, marital_status, education, dependents, income, loan_amount, credit_history, self_employed, coapplicant_income, loan_amount_term, property_area):
    if n_clicks > 0:
        result = check_eligibility(1, gender, marital_status, dependents, education, self_employed, income, coapplicant_income, loan_amount, credit_history, loan_amount_term, property_area)
        return html.Div(f"Loan Eligibility: {result}", className='result')
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)