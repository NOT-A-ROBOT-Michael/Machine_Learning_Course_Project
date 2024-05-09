import csv
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import prepare_data
import train_models
import feature_engineering

feature_engineering.dump_data()

model = joblib.load('../artifacts/model_2.pkl')

coefficients = model.named_steps["randomforestclassifier"].feature_importances_
features = model.named_steps['randomforestclassifier'].feature_names_in_

feat_imp = pd.Series(
    np.exp(coefficients), index=features
).sort_values(ascending=True)

def check_eligibility(loan_ID, gender, marital_status, education, dependents, income, loan_amount, credit_history, self_employed, coapplicant_income, loan_amount_term, property_area):
    input_data = pd.DataFrame({
        'Loan_ID': [loan_ID],
        'Gender': [1 if gender == 'Female' else 0],
        'Married': [1 if marital_status == 'Married' else 0],
        'Education': [1 if education == 'Graduate' else 0],
        'Dependents': [dependents],
        'ApplicantIncome': [income],
        'LoanAmount': [loan_amount],
        'CreditHistory': [1 if credit_history == 'Good' else 0],
        'SelfEmployed': [1 if self_employed == 'Yes' else 0],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmountTerm': [loan_amount_term],
        'PropertyArea_Rural': [1 if property_area == 'Rural' else 0],
        'PropertyArea_Semiurban': [1 if property_area == 'Semiurban' else 0],
        'PropertyArea_Urban': [1 if property_area == 'Urban' else 0]
    })       

    temp_df = prepare_data.prepare_data_final(input_data)
    
    Xtrain = train_models.modeling(temp_df)

    prediction = model.predict(Xtrain)

    return "Eligible" if np.round(prediction[0] == 1) else "Not Eligible"

def write_to_csv(inputs):
    with open('../data/web_app.csv', mode='a', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(inputs)

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
def update_output(n_clicks, gender, marital_status, dependents, education, income, loan_amount, credit_history, self_employed, coapplicant_income, loan_amount_term, property_area):
    if n_clicks > 0:
        result = check_eligibility(1, gender, marital_status, dependents, education, self_employed, income, coapplicant_income, loan_amount, credit_history, loan_amount_term, property_area)
        inputs = [1, gender, marital_status, dependents, education, income, loan_amount, credit_history, self_employed, coapplicant_income, loan_amount_term, property_area]
        write_to_csv(inputs)
        return html.Div(f"Loan Eligibility: {result}", className='result')
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)