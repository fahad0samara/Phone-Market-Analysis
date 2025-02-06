import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import re
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

class MobilePhoneML:
    def __init__(self):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Initialize database connection
        self.engine = create_engine('sqlite:///database/mobile_phones.db')
        
    def prepare_features(self, df):
        """Prepare features for ML models"""
        print("Preparing features...")
        
        # Create brand encoder
        self.brand_encoder = LabelEncoder()
        df['brand_encoded'] = self.brand_encoder.fit_transform(df['brand'])
        
        # Extract numeric features from name and corpus
        def extract_storage(text):
            match = re.search(r'(\d+)\s*GB', str(text))
            return float(match.group(1)) if match else 0
            
        def extract_ram(text):
            match = re.search(r'RAM(\d+)', str(text))
            return float(match.group(1)) if match else 0
            
        def extract_camera(text):
            match = re.search(r'(\d+)\s*MP', str(text))
            return float(match.group(1)) if match else 0
            
        def extract_battery(text):
            match = re.search(r'(\d+)\s*mAh', str(text))
            return float(match.group(1)) if match else 0
        
        # Extract features
        df['storage_gb'] = df['name'].apply(extract_storage)
        df['ram_gb'] = df['corpus'].apply(extract_ram)
        df['camera_mp'] = df['corpus'].apply(extract_camera)
        df['battery_mah'] = df['corpus'].apply(extract_battery)
        
        # Create feature matrix
        self.feature_columns = ['brand_encoded', 'storage_gb', 'ram_gb', 'camera_mp', 'battery_mah']
        X = df[self.feature_columns]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, self.feature_columns
    
    def train_price_predictor(self):
        """Train a model to predict phone prices"""
        print("\nTraining price prediction model...")
        
        # Get data from database
        df = pd.read_sql('SELECT * FROM mobile_phones', self.engine)
        
        # Prepare features
        X_scaled, feature_columns = self.prepare_features(df)
        y = df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.price_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.price_model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.price_model.predict(X_train)
        test_pred = self.price_model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nPrice Prediction Results:")
        print(f"Train RMSE: ₹{train_rmse:.2f}")
        print(f"Test RMSE: ₹{test_rmse:.2f}")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.price_model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Feature Importance for Price Prediction')
        plt.tight_layout()
        plt.savefig('visualizations/price_feature_importance.png')
        plt.close()
        
        # Save model
        joblib.dump(self.price_model, 'models/price_predictor.joblib')
        joblib.dump(self.scaler, 'models/feature_scaler.joblib')
        joblib.dump(self.brand_encoder, 'models/brand_encoder.joblib')
        
        return importance
    
    def train_rating_predictor(self):
        """Train a model to predict phone ratings"""
        print("\nTraining rating prediction model...")
        
        # Get data from database
        df = pd.read_sql('SELECT * FROM mobile_phones', self.engine)
        
        # Prepare features
        X_scaled, feature_columns = self.prepare_features(df)
        
        # Add price as a feature
        price_scaled = StandardScaler().fit_transform(df[['price']])
        X_scaled_with_price = np.hstack([X_scaled, price_scaled])
        self.feature_columns_with_price = feature_columns + ['price']
        
        y = df['ratings']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_with_price, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.rating_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.rating_model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.rating_model.predict(X_train)
        test_pred = self.rating_model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nRating Prediction Results:")
        print(f"Train MAE: {train_mae:.3f} stars")
        print(f"Test MAE: {test_mae:.3f} stars")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns_with_price,
            'importance': self.rating_model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Feature Importance for Rating Prediction')
        plt.tight_layout()
        plt.savefig('visualizations/rating_feature_importance.png')
        plt.close()
        
        # Save model
        joblib.dump(self.rating_model, 'models/rating_predictor.joblib')
        
        return importance
    
    def get_feature_importance(self):
        """Get feature importance from the price prediction model"""
        if not hasattr(self, 'price_model'):
            self.train_price_predictor()
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.price_model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
    
    def predict_price(self, brand, storage, ram, camera, battery):
        """Predict price for a new phone"""
        # Load models if not already loaded
        if not hasattr(self, 'price_model'):
            self.price_model = joblib.load('models/price_predictor.joblib')
            self.scaler = joblib.load('models/feature_scaler.joblib')
            self.brand_encoder = joblib.load('models/brand_encoder.joblib')
            self.feature_columns = ['brand_encoded', 'storage_gb', 'ram_gb', 'camera_mp', 'battery_mah']
        
        # Prepare features
        brand_encoded = self.brand_encoder.transform([brand])[0]
        features = pd.DataFrame([[brand_encoded, storage, ram, camera, battery]], 
                              columns=self.feature_columns)
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        predicted_price = self.price_model.predict(features_scaled)[0]
        return predicted_price
    
    def predict_rating(self, brand, storage, ram, camera, battery, price):
        """Predict rating for a new phone"""
        # Load model if not already loaded
        if not hasattr(self, 'rating_model'):
            self.rating_model = joblib.load('models/rating_predictor.joblib')
            if not hasattr(self, 'feature_columns'):
                self.feature_columns = ['brand_encoded', 'storage_gb', 'ram_gb', 'camera_mp', 'battery_mah']
                self.feature_columns_with_price = self.feature_columns + ['price']
        
        # Prepare features (including price)
        brand_encoded = self.brand_encoder.transform([brand])[0]
        features = pd.DataFrame([[brand_encoded, storage, ram, camera, battery]], 
                              columns=self.feature_columns)
        features_scaled = self.scaler.transform(features)
        
        # Scale price separately
        price_scaled = StandardScaler().fit_transform([[price]])
        features_scaled_with_price = np.hstack([features_scaled, price_scaled])
        
        # Make prediction
        predicted_rating = self.rating_model.predict(features_scaled_with_price)[0]
        return predicted_rating

def main():
    # Initialize ML system
    ml_system = MobilePhoneML()
    
    # Train models
    print("Training machine learning models...")
    price_importance = ml_system.train_price_predictor()
    rating_importance = ml_system.train_rating_predictor()
    
    # Save feature importance results
    price_importance.to_csv('exports/price_feature_importance.csv', index=False)
    rating_importance.to_csv('exports/rating_feature_importance.csv', index=False)
    
    # Example predictions
    print("\nExample Predictions:")
    example_phone = {
        'brand': 'SAMSUNG',
        'storage': 128,
        'ram': 8,
        'camera': 48,
        'battery': 5000
    }
    
    predicted_price = ml_system.predict_price(
        example_phone['brand'],
        example_phone['storage'],
        example_phone['ram'],
        example_phone['camera'],
        example_phone['battery']
    )
    
    predicted_rating = ml_system.predict_rating(
        example_phone['brand'],
        example_phone['storage'],
        example_phone['ram'],
        example_phone['camera'],
        example_phone['battery'],
        predicted_price
    )
    
    print(f"\nPredictions for {example_phone['brand']} phone:")
    print(f"Storage: {example_phone['storage']}GB")
    print(f"RAM: {example_phone['ram']}GB")
    print(f"Camera: {example_phone['camera']}MP")
    print(f"Battery: {example_phone['battery']}mAh")
    print(f"Predicted Price: ₹{predicted_price:.2f}")
    print(f"Predicted Rating: {predicted_rating:.2f} stars")

if __name__ == "__main__":
    main()
