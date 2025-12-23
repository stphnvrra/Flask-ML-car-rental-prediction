# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# --- Load the dataset first, then model ---
MODEL_PATH = 'car_rental_model.pkl'
FEATURES_PATH = 'car_rental_model_features.json'
DATA_PATH = 'CarRentalData.csv'

# Load the dataset first (needed for fallback model creation)
try:
    df_data = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded from {DATA_PATH}")
    print(f"Dataset shape: {df_data.shape}")
    print(f"Target variable (rate.daily) stats: {df_data['rate.daily'].describe()}")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_PATH}. Please ensure it's in the same directory or provide full path.")
    exit()

# Load model with fallback
try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Model type: {type(ml_model)}")
    print(f"Model has predict method: {hasattr(ml_model, 'predict')}")
    
    if hasattr(ml_model, 'predict'):
        print("✅ Model loaded successfully")
    else:
        print("❌ Model object doesn't have predict method")
        print(f"Model contents: {str(ml_model)[:200]}")
        raise ValueError("Invalid model object")
        
except Exception as e:
    print(f"❌ Error loading model: {type(e).__name__}: {str(e)}")
    print("🔄 Creating fallback dummy model for testing...")
    
    # Create a simple fallback model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create and train a simple model on the existing data
    print("Training a simple fallback model...")
    fallback_data = df_data.dropna()
    
    # Use only numeric features that are definitely available
    potential_features = ['rating', 'renterTripsTaken', 'reviewCount', 
                         'location.latitude', 'location.longitude', 'vehicle.year']
    
    # Check which features actually exist in the data
    available_features = [f for f in potential_features if f in fallback_data.columns]
    print(f"Available numeric features for fallback model: {available_features}")
    
    if len(available_features) == 0:
        print("❌ No suitable numeric features found for fallback model")
        exit()
    
    X_fallback = fallback_data[available_features].copy()
    y_fallback = fallback_data['rate.daily']
    
    # Train simple model
    ml_model = RandomForestRegressor(n_estimators=10, random_state=42)
    ml_model.fit(X_fallback, y_fallback)
    
    # Update features list for fallback model
    features = available_features
    
    print("✅ Fallback model created successfully")
    print(f"Fallback model features: {features}")

# Load features list (only if not using fallback model)
if 'features' not in locals():
    try:
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        print(f"Features loaded from {FEATURES_PATH}: {features}")
    except FileNotFoundError:
        print(f"Error: Features file not found at {FEATURES_PATH}. Please ensure it's in the same directory or provide full path.")
        # Use basic features as fallback
        features = ['rating', 'renterTripsTaken', 'reviewCount', 
                   'location.latitude', 'location.longitude', 'vehicle.year']
        print(f"Using fallback features: {features}")

print(f"Final features being used: {features}")

# Dataset already loaded above

@app.route('/')
def home():
    # Get unique values for dropdowns
    fuel_types = sorted(df_data['fuelType'].dropna().unique().tolist())
    cities = sorted(df_data['location.city'].dropna().unique().tolist())
    countries = sorted(df_data['location.country'].dropna().unique().tolist())
    states = sorted(df_data['location.state'].dropna().unique().tolist())
    makes = sorted(df_data['vehicle.make'].dropna().unique().tolist())
    models = sorted(df_data['vehicle.model'].dropna().unique().tolist())
    types = sorted(df_data['vehicle.type'].dropna().unique().tolist())
    years = sorted(df_data['vehicle.year'].dropna().unique().tolist(), reverse=True)
    
    return render_template('index.html', 
                         fuel_types=fuel_types,
                         cities=cities,
                         countries=countries,
                         states=states,
                         makes=makes,
                         models=models,
                         types=types,
                         years=years)

@app.route('/test', methods=['GET'])
def test_connection():
    """Simple test endpoint to verify backend is working"""
    return jsonify({
        'status': 'success',
        'message': 'Backend is working',
        'model_features': features,
        'data_shape': df_data.shape
    })

@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    """Get a sample row from the dataset for testing"""
    sample_row = df_data.iloc[0].to_dict()
    return jsonify({
        'sample_data': sample_row,
        'message': 'Use this data to test the prediction'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug: Print all form data received
        print("=== DEBUG: Form data received ===")
        for key, value in request.form.items():
            print(f"{key}: {value}")
        print("================================")
        
        # Get form data with better error handling
        fuel_type = request.form.get('fuelType')
        if not fuel_type:
            raise ValueError("Fuel type is required")
            
        rating_str = request.form.get('rating', '0')
        rating = float(rating_str) if rating_str else 0
        
        renter_trips_str = request.form.get('renterTripsTaken', '0')
        renter_trips = int(renter_trips_str) if renter_trips_str else 0
        
        review_count_str = request.form.get('reviewCount', '0')
        review_count = int(review_count_str) if review_count_str else 0
        
        city = request.form.get('location.city')
        if not city:
            raise ValueError("City is required")
            
        country = request.form.get('location.country')
        if not country:
            raise ValueError("Country is required")
            
        latitude_str = request.form.get('location.latitude', '0')
        latitude = float(latitude_str) if latitude_str else 0
        
        longitude_str = request.form.get('location.longitude', '0')
        longitude = float(longitude_str) if longitude_str else 0
        
        state = request.form.get('location.state')
        if not state:
            raise ValueError("State is required")
            
        make = request.form.get('vehicle.make')
        if not make:
            raise ValueError("Vehicle make is required")
            
        vehicle_model = request.form.get('vehicle.model')
        if not vehicle_model:
            raise ValueError("Vehicle model is required")
            
        vehicle_type = request.form.get('vehicle.type')
        if not vehicle_type:
            raise ValueError("Vehicle type is required")
            
        year_str = request.form.get('vehicle.year', '2020')
        year = int(year_str) if year_str else 2020
        
        print(f"=== Processed values ===")
        print(f"fuel_type: {fuel_type}, rating: {rating}, renter_trips: {renter_trips}")
        print(f"city: {city}, state: {state}, country: {country}")
        print(f"make: {make}, model: {vehicle_model}, type: {vehicle_type}, year: {year}")
        print("========================")
        
        # Create input dataframe - include all possible features first
        all_input_data = {
            'fuelType': fuel_type,
            'rating': rating,
            'renterTripsTaken': renter_trips,
            'reviewCount': review_count,
            'location.city': city,
            'location.country': country,
            'location.latitude': latitude,
            'location.longitude': longitude,
            'location.state': state,
            'vehicle.make': make,
            'vehicle.model': vehicle_model,
            'vehicle.type': vehicle_type,
            'vehicle.year': year
        }
        
        # Only include features that the model actually needs
        filtered_input_data = {}
        for feature in features:
            if feature in all_input_data:
                filtered_input_data[feature] = [all_input_data[feature]]
            else:
                print(f"⚠️  Feature '{feature}' not available in input data")
        
        input_data = pd.DataFrame(filtered_input_data)
        
        print(f"=== Input data created ===")
        print(f"Required features: {features}")
        print(f"Available features: {list(input_data.columns)}")
        print(input_data.head())
        print("=========================")
        
        # Ensure the columns are in the same order as the model expects
        try:
            input_data = input_data[features]
            print(f"=== Input data after reordering ===")
            print(input_data.head())
            print("==================================")
        except KeyError as e:
            available_features = list(input_data.columns)
            missing_features = [f for f in features if f not in available_features]
            raise ValueError(f"Missing required features: {missing_features}. Available: {available_features}")
        
        # Check for any NaN values
        if input_data.isnull().any().any():
            print("WARNING: NaN values detected in input data")
            print(input_data.isnull().sum())
        
        # Handle categorical variables - encode them to numeric values
        print("=== Encoding categorical variables ===")
        all_categorical_columns = ['fuelType', 'location.city', 'location.country', 'location.state', 
                                 'vehicle.make', 'vehicle.model', 'vehicle.type']
        
        # Only check categorical columns that are actually in our feature set
        categorical_features = [col for col in all_categorical_columns if col in features and col in input_data.columns]
        print(f"Categorical features being used: {categorical_features}")
        
        # For each categorical column, encode it to numeric values
        for col in categorical_features:
            input_value = input_data[col].iloc[0]
            unique_values = df_data[col].dropna().unique()
            print(f"{col}: '{input_value}' - exists in training data: {input_value in unique_values}")
            
            # If the value doesn't exist in training data, use the most common value
            if input_value not in unique_values:
                most_common = df_data[col].mode().iloc[0] if not df_data[col].mode().empty else unique_values[0]
                print(f"⚠️  '{input_value}' not found in training data. Using most common: '{most_common}'")
                input_data[col] = most_common
                input_value = most_common
            
            # Encode categorical value to numeric
            # Create a mapping based on the training data
            unique_sorted = sorted(unique_values)
            value_to_index = {val: idx for idx, val in enumerate(unique_sorted)}
            numeric_value = value_to_index.get(input_value, 0)
            
            print(f"Encoding '{input_value}' as {numeric_value}")
            input_data[col] = numeric_value
        
        if not categorical_features:
            print("No categorical features in this model (probably using fallback model)")
        
        print("=====================================")
        
        # Check data types
        print("=== Final data for prediction ===")
        print("Data types:")
        print(input_data.dtypes)
        print("Data values:")
        print(input_data.head())
        print("==================================")
            
        # Make prediction
        try:
            # Double-check model type before prediction
            print(f"About to predict with model type: {type(ml_model)}")
            if not hasattr(ml_model, 'predict'):
                raise ValueError(f"Model object is not a valid ML model. Type: {type(ml_model)}, Content: {str(ml_model)[:100]}")
            
            predicted_price = ml_model.predict(input_data)[0]
            print(f"=== Prediction successful ===")
            print(f"Predicted price: {predicted_price}")
            print("=============================")
        except Exception as pred_error:
            print(f"Prediction error details:")
            print(f"- Model type: {type(ml_model)}")
            print(f"- Model has predict: {hasattr(ml_model, 'predict')}")
            print(f"- Input data shape: {input_data.shape}")
            print(f"- Input data types: {input_data.dtypes}")
            print(f"- Error: {str(pred_error)}")
            raise ValueError(f"Model prediction failed: {str(pred_error)}")
        
        # Generate a simple visualization comparing to similar vehicles
        plt.figure(figsize=(10, 6))
        
        # Filter similar vehicles for comparison
        similar_vehicles = df_data[
            (df_data['vehicle.type'] == vehicle_type) & 
            (df_data['vehicle.year'].between(year-2, year+2))
        ]['rate.daily'].dropna()
        
        if len(similar_vehicles) > 0:
            plt.hist(similar_vehicles, bins=20, alpha=0.7, color='lightblue', 
                    label=f'Similar {vehicle_type}s ({year-2}-{year+2})')
            plt.axvline(predicted_price, color='red', linestyle='--', linewidth=2, 
                       label=f'Predicted Price: ${predicted_price:.2f}')
            plt.xlabel('Daily Rental Rate ($)')
            plt.ylabel('Number of Vehicles')
            plt.title(f'Rental Price Comparison for {vehicle_type}s')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No similar vehicles found for comparison', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title(f'Predicted Price: ${predicted_price:.2f}')
        
        plt.tight_layout()
        
        # Save plot to base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plot_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'success': True,
            'predicted_price': f'{predicted_price:.2f}',
            'vehicle_info': f'{year} {make} {vehicle_model} ({vehicle_type})',
            'plot_image': plot_base64
        })
        
    except Exception as e:
        print(f"=== ERROR OCCURRED ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("=====================")
        return jsonify({
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}"
        }), 400

if __name__ == '__main__':
    app.run(debug=True)