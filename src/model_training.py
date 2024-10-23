import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def custom_mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

data = pd.read_csv('data/feature_engineered_bus_demand.csv')

data['HolidayEffect'].fillna(False, inplace=True)
data['Friday'].fillna(False, inplace=True)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=False)

X = data.drop(columns=['RequiredBuses', 'Date','MaxPassengers','MinPassengers','AvgPassengers'])  # Drop the target and date columns
y = data['RequiredBuses']

if X.isnull().values.any() or y.isnull().values.any():
    print("There are still missing values in the features or target.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/scaler.pkl')

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    joblib.dump(lin_reg, 'models/linear_regression_model.pkl')

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train_scaled)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    joblib.dump(poly_reg, 'models/polynomial_regression_model.pkl')
    joblib.dump(poly, 'models/poly_transform.pkl')

    ridge_reg = Ridge(alpha=1.0)
    ridge_reg.fit(X_train_scaled, y_train)
    joblib.dump(ridge_reg, 'models/ridge_regression_model.pkl')

    print("Models trained and saved successfully.")

    models = {
        'Linear Regression': lin_reg,
        'Polynomial Regression': poly_reg,
        'Ridge Regression': ridge_reg
    }

    results = []

    for model_name, model in models.items():
        if model_name == 'Polynomial Regression':
            X_test_transformed = poly.transform(X_test_scaled)
            y_pred = model.predict(X_test_transformed)
        else:
            y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mse = custom_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = (abs(y_test - y_pred) / y_test <= 0.1).mean() * 100

        results.append({
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'R²': r2,
            'Accuracy (%)': accuracy
        })

    results_df = pd.DataFrame(results)

    print("Model Evaluation Results:")
    print(results_df)

    best_model = results_df.loc[results_df['R²'].idxmax()]
    print(f"Best Model: {best_model['Model']} with R²: {best_model['R²']}")
