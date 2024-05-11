import csv
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd

gender_options = [{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}]
marital_status_options = [{'label': 'Single', 'value': 'Single'}, {'label': 'Married', 'value': 'Married'}]
education_options = [{'label': 'Graduate', 'value': 'Graduate'}, {'label': 'Not Graduate', 'value': 'Not Graduate'}]
credit_history_options = [{'label': 'Good', 'value': 'Good'}, {'label': 'Bad', 'value': 'Bad'}]
self_employed_options = [{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}]
property_area_options = [{'label': 'Urban', 'value': 'Urban'}, {'label': 'Semiurban', 'value': 'Semiurban'}, {'label': 'Rural', 'value': 'Rural'}]
    
    
    
    
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  


def check_eligibility(gender, marital_status, education, dependents,  income, loan_amount, credit_history, self_employed, coapplicant_income, loan_amount_term, property_area):
    input_data = pd.DataFrame({
        'applicant_income':[income],
        'coapplicant_income':[coapplicant_income],
        'loan_amount':[loan_amount],
        'loan_amount_term':[loan_amount_term],
        'credit_history':[1 if credit_history == 'Good' else 0],
        'gender_Female':[1 if gender == 'Female' else 0],
        'gender_Male':[1 if gender == 'Male' else 0],
        'married_No':[1 if marital_status == 'Single' else 0],
        'married_Yes':[1 if marital_status == 'Married' else 0],
        'dependents_0':[1 if dependents == 0 else 0],
        'dependents_1':[1 if dependents == 1 else 0],
        'dependents_2':[1 if dependents == 2 else 0],
        'dependents_3+':[1 if dependents >= 3 else 0],
        'education_Graduate':[1 if education == 'Graduate' else 0],
        'education_Not Graduate':[1 if education == 'Not Graduate' else 0],
        'self_employed_No':[1 if self_employed == 'No' else 0],
        'self_employed_Yes':[1 if self_employed == 'Yes' else 0],
        'property_area_Rural':[1 if property_area == 'Rural' else 0],
        'property_area_Semiurban':[1 if property_area == 'Semiurban' else 0],
        'property_area_Urban':[1 if property_area == 'Urban' else 0],
        'total_income': [income + coapplicant_income]
        
    })       

    model = joblib.load('../artifacts/model_2.pkl')
    #Xtrain = train_models.modeling(temp_df)

    prediction = model.predict(input_data)

    return "Eligible" if np.round(prediction[0] == 1) else "Not Eligible"



def write_to_csv(inputs):
    with open('../data/web_app.csv', mode='a', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(inputs)





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
def update_output(n_clicks, gender, marital_status, education, dependents,  income, loan_amount, credit_history, self_employed, coapplicant_income, loan_amount_term, property_area):
    if n_clicks > 0:
        result = check_eligibility( gender, marital_status, education, dependents, income, loan_amount , credit_history,  self_employed,  coapplicant_income,   loan_amount_term, property_area)
        print("working")
        inputs = [0, gender, marital_status,  education, dependents, income, loan_amount, credit_history, self_employed, coapplicant_income, loan_amount_term, property_area]
        write_to_csv(inputs)
        return html.Div(f"Loan Eligibility: {result}", className='result')
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)