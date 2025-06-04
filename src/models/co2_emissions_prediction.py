import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
from io import StringIO
import os

"""
CO2 Emissions Prediction Model for SDG 13: Climate Action
This project aims to predict CO2 emissions based on various economic and social factors,
contributing to the United Nations Sustainable Development Goal 13: Climate Action.

The model uses real-world data from the World Bank API to predict CO2 emissions per capita
based on factors such as GDP, energy consumption, and population.

Ethical Considerations:
1. Data Bias: The model may be biased towards countries with better data collection systems
2. Fairness: Predictions should be interpreted in the context of each country's development stage
3. Transparency: Model decisions are explainable through feature importance analysis
"""

def fetch_world_bank_data():
    """Fetch real-world data from World Bank API"""
    # World Bank API endpoints for relevant indicators
    indicators = {
        'EN.ATM.CO2E.PC': 'CO2_emissions',  # CO2 emissions (metric tons per capita)
        'NY.GDP.PCAP.CD': 'GDP_per_capita',  # GDP per capita (current US$)
        'EG.USE.PCAP.KG.OE': 'Energy_use',  # Energy use (kg of oil equivalent per capita)
        'SP.POP.TOTL': 'Population'  # Total population
    }
    
    data = {}
    
    for indicator, name in indicators.items():
        try:
            url = f"http://api.worldbank.org/v2/country/all/indicator/{indicator}?date=2019&format=csv"
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            if response.text.strip():  # Check if response is not empty
                df = pd.read_csv(StringIO(response.text), skiprows=4)
                if not df.empty:
                    data[name] = df
                else:
                    print(f"Warning: No data received for {name}")
            else:
                print(f"Warning: Empty response for {name}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {name}: {str(e)}")
        except pd.errors.EmptyDataError:
            print(f"Error: No data to parse for {name}")
        except Exception as e:
            print(f"Unexpected error for {name}: {str(e)}")
    
    if not data:
        print("No data could be fetched. Using sample data instead.")
        return generate_sample_data()
    
    return data

def generate_sample_data():
    """Generate sample data when API is unavailable"""
    np.random.seed(42)
    n_samples = 100
    
    # Generate realistic ranges for the features
    gdp_per_capita = np.random.uniform(1000, 50000, n_samples)
    energy_use = np.random.uniform(500, 10000, n_samples)
    population = np.random.uniform(1000000, 1000000000, n_samples)
    
    # Generate CO2 emissions with a realistic relationship to other features
    co2_emissions = (
        0.0001 * gdp_per_capita + 
        0.0002 * energy_use + 
        0.00001 * population + 
        np.random.normal(0, 2, n_samples)
    )
    
    # Create DataFrame
    data = {
        'CO2_emissions': pd.DataFrame({'value': co2_emissions}),
        'GDP_per_capita': pd.DataFrame({'value': gdp_per_capita}),
        'Energy_use': pd.DataFrame({'value': energy_use}),
        'Population': pd.DataFrame({'value': population})
    }
    
    return data

def preprocess_data(data):
    """Preprocess and combine the World Bank data"""
    try:
        # Combine all indicators into a single DataFrame
        combined_data = pd.DataFrame()
        
        for name, df in data.items():
            if name == 'CO2_emissions':
                combined_data['CO2_emissions'] = df['value']
            else:
                combined_data[name] = df['value']
        
        # Handle missing values
        combined_data = combined_data.fillna(method='ffill')
        
        # Remove any remaining rows with NaN values
        combined_data = combined_data.dropna()
        
        if combined_data.empty:
            raise ValueError("No valid data after preprocessing")
            
        return combined_data
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

def train_model(X, y):
    """Train the Random Forest model"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R2): {r2:.2f}")
        
        return y_pred
        
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        raise

def plot_results(y_test, y_pred, feature_importance, model):
    """Create visualizations of model results"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual CO2 Emissions')
        plt.ylabel('Predicted CO2 Emissions')
        plt.title('Actual vs Predicted CO2 Emissions')
        plt.grid(True)
        plt.savefig('output/co2_emissions_plot.png')
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance in CO2 Emissions Prediction')
        plt.savefig('output/feature_importance_plot.png')
        
        plt.show()
        
    except Exception as e:
        print(f"Error in plotting results: {str(e)}")
        raise

def main():
    try:
        # Fetch and preprocess data
        print("Fetching data from World Bank API...")
        data = fetch_world_bank_data()
        processed_data = preprocess_data(data)
        
        # Prepare features and target
        features = ['GDP_per_capita', 'Energy_use', 'Population']
        X = processed_data[features]
        y = processed_data['CO2_emissions']
        
        # Train model
        print("\nTraining model...")
        model, X_test, y_test = train_model(X, y)
        
        # Evaluate model
        print("\nEvaluating model performance...")
        y_pred = evaluate_model(model, X_test, y_test)
        
        # Plot results
        print("\nGenerating visualizations...")
        plot_results(y_test, y_pred, model.feature_importances_, model)
        
        # Save model
        joblib.dump(model, 'output/co2_emissions_model.joblib')
        print("\nModel saved as 'output/co2_emissions_model.joblib'")
        
        # Print ethical considerations
        print("\nEthical Considerations:")
        print("1. Data Bias: The model may be biased towards countries with better data collection systems")
        print("2. Fairness: Predictions should be interpreted in the context of each country's development stage")
        print("3. Transparency: Model decisions are explainable through feature importance analysis")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()

