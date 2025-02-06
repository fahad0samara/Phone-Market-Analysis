import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class MobilePhoneML:
    def __init__(self):
        # Load data
        self.df = pd.read_csv('mobile_dataset_cleaned.csv')
        
        # Extract brand from name
        self.df['brand'] = self.df['name'].apply(lambda x: x.split()[0])
        
        # Convert price and ratings to numeric
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df['ratings'] = pd.to_numeric(self.df['ratings'], errors='coerce')
        
        # Create brand encoder
        self.brand_encoder = LabelEncoder()
        self.df['brand_encoded'] = self.brand_encoder.fit_transform(self.df['brand'])
        
        # Create feature matrix
        self.feature_columns = ['brand_encoded', 'ratings']
        self.target_column = 'price'
        
        # Scale features
        self.scaler = StandardScaler()
        self.prepare_features()
    
    def prepare_features(self):
        """Prepare features for ML models"""
        print("Preparing features...")
        
        # Create feature matrix
        X = self.df[self.feature_columns]
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = self.df[self.target_column]
    
    def train_price_predictor(self):
        """Train a model to predict phone prices"""
        print("\nTraining price prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
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
        print(f"Train RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        
        # Plot feature importance
        self.plot_feature_importance()
    
    def predict_price(self, brand, rating):
        """Predict price for a given phone configuration"""
        # Encode brand
        brand_encoded = self.brand_encoder.transform([brand])[0]
        
        # Create feature vector
        features = np.array([[brand_encoded, rating]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        predicted_price = self.price_model.predict(features_scaled)[0]
        
        return predicted_price
    
    def plot_feature_importance(self):
        """Plot feature importance scores"""
        importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.price_model.feature_importances_
        })
        importance = importance.sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['Feature'], importance['Importance'])
        plt.title('Feature Importance for Price Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        plt.close()
        
        print("\nFeature importance plot saved to visualizations/feature_importance.png")

def main():
    # Initialize and train model
    ml_system = MobilePhoneML()
    ml_system.train_price_predictor()
    
    # Example predictions
    print("\nExample Predictions:")
    for brand in ['Samsung', 'Apple', 'Xiaomi']:
        for rating in [3.0, 4.0, 5.0]:
            price = ml_system.predict_price(brand, rating)
            print(f"{brand} phone with {rating} rating: ${price:,.2f}")

if __name__ == "__main__":
    main()
