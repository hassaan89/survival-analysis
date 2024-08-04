# code to install the following libararies before use
# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install seaborn
# !pip install lifelines
# !pip install scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SurvSet.data import SurvLoader
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import concordance_index_censored

# Load the dataset
def load_dataset(ds_name):
    loader = SurvLoader()
    df, ref = loader.load_dataset(ds_name)
    return df

# Preprocess the data
def preprocess_data(df):
    # Convert categorical variables to dummy variables
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(df[['fac1', 'fac2', 'fac3']])
    df = pd.concat([df.drop(['fac1', 'fac2', 'fac3'], axis=1), pd.DataFrame(encoder.transform(df[['fac1', 'fac2', 'fac3']]), columns=encoder.categories_[0])], axis=1)

    # Scale numerical variables
    scaler = StandardScaler()
    df[['num1', 'num2', 'num3']] = scaler.fit_transform(df[['num1', 'num2', 'num3']])

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[['num1', 'num2', 'num3']] = imputer.fit_transform(df[['num1', 'num2', 'num3']])

    # Split the data into training and testing sets
    X = df.drop(['pid', 'event', 'time'], axis=1)
    y = df[['event', 'time']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# Fit a Cox proportional hazards model
def fit_cox_model(X_train, y_train):
    cph = CoxPHFitter()
    cph.fit(X_train, duration_col='time', event_col='event')
    return cph

# Evaluate the model using concordance index
def evaluate_model(cph, X_test, y_test):
    c_index = concordance_index_censored(cph.predict_partial_hazard(X_test), X_test['event'], X_test['time'])
    print('Concordance index:', c_index)

# Plot the survival curves for a subset of the data
def plot_survival_curves(kmf, subset_df):
    kmf.fit(subset_df['time'], event_observed=subset_df['event'])
    kmf.plot()
    plt.show()

# Implement the main function
def main():
    # Load the dataset
    ds_name = 'ovarian_cancer'
    df = load_dataset(ds_name)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Fit the Cox model
    cph = fit_cox_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(cph, X_test, y_test)

    # Plot the survival curves for a subset of the data
    subset_df = X_test.iloc[:10, :]
    kmf = KaplanMeierFitter()
    plot_survival_curves(kmf, subset_df)

# Call the main function
main()