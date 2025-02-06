from mobile_phone_ml import MobilePhoneML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

class PhonePredictor:
    def __init__(self):
        self.ml_system = MobilePhoneML()
        self.engine = create_engine('sqlite:///database/mobile_phones.db')
        
        # Load existing brands
        df = pd.read_sql('SELECT DISTINCT brand FROM mobile_phones', self.engine)
        self.available_brands = sorted(df['brand'].tolist())
        print("\nAvailable brands:", ", ".join(self.available_brands))
    
    def predict_single_phone(self, brand, storage, ram, camera, battery):
        """Predict price and rating for a single phone configuration"""
        # Predict price
        predicted_price = self.ml_system.predict_price(
            brand, storage, ram, camera, battery
        )
        
        # Predict rating using the predicted price
        predicted_rating = self.ml_system.predict_rating(
            brand, storage, ram, camera, battery, predicted_price
        )
        
        return predicted_price, predicted_rating
    
    def compare_brands(self, storage, ram, camera, battery):
        """Compare predictions across different brands"""
        results = []
        for brand in self.available_brands:
            price, rating = self.predict_single_phone(
                brand, storage, ram, camera, battery
            )
            results.append({
                'brand': brand,
                'price': price,
                'rating': rating,
                'value_score': rating / price * 10000  # Higher score = better value
            })
        
        return pd.DataFrame(results)
    
    def analyze_feature_impact(self, brand, base_storage=128, base_ram=6, 
                             base_camera=48, base_battery=5000):
        """Analyze how changing each feature impacts price and rating"""
        if brand not in self.available_brands:
            print(f"Error: Brand '{brand}' not found. Available brands:", ", ".join(self.available_brands))
            return None
            
        results = {
            'feature': [],
            'value': [],
            'price': [],
            'rating': []
        }
        
        # Test different storage options
        for storage in [32, 64, 128, 256, 512]:
            price, rating = self.predict_single_phone(
                brand, storage, base_ram, base_camera, base_battery
            )
            results['feature'].append('Storage (GB)')
            results['value'].append(storage)
            results['price'].append(price)
            results['rating'].append(rating)
        
        # Test different RAM options
        for ram in [2, 4, 6, 8, 12, 16]:
            price, rating = self.predict_single_phone(
                brand, base_storage, ram, base_camera, base_battery
            )
            results['feature'].append('RAM (GB)')
            results['value'].append(ram)
            results['price'].append(price)
            results['rating'].append(rating)
        
        # Test different camera options
        for camera in [12, 24, 48, 64, 108]:
            price, rating = self.predict_single_phone(
                brand, base_storage, base_ram, camera, base_battery
            )
            results['feature'].append('Camera (MP)')
            results['value'].append(camera)
            results['price'].append(price)
            results['rating'].append(rating)
        
        # Test different battery options
        for battery in [3000, 4000, 5000, 6000, 7000]:
            price, rating = self.predict_single_phone(
                brand, base_storage, base_ram, base_camera, battery
            )
            results['feature'].append('Battery (mAh)')
            results['value'].append(battery)
            results['price'].append(price)
            results['rating'].append(rating)
        
        return pd.DataFrame(results)
    
    def plot_feature_impact(self, impact_df, save_path=None):
        """Plot how features impact price and rating"""
        features = impact_df['feature'].unique()
        fig, axes = plt.subplots(len(features), 2, figsize=(15, 5*len(features)))
        
        for i, feature in enumerate(features):
            feature_data = impact_df[impact_df['feature'] == feature]
            
            # Price impact
            sns.lineplot(data=feature_data, x='value', y='price', ax=axes[i,0])
            axes[i,0].set_title(f'{feature} Impact on Price')
            axes[i,0].set_ylabel('Price (₹)')
            
            # Rating impact
            sns.lineplot(data=feature_data, x='value', y='rating', ax=axes[i,1])
            axes[i,1].set_title(f'{feature} Impact on Rating')
            axes[i,1].set_ylabel('Rating')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def find_best_value_configs(self, min_rating=4.0, max_price=30000):
        """Find the best value phone configurations"""
        results = []
        
        # Test different configurations
        for brand in self.available_brands:
            for storage in [64, 128, 256]:
                for ram in [4, 6, 8]:
                    price, rating = self.predict_single_phone(
                        brand, storage, ram, 48, 5000  # Standard camera and battery
                    )
                    
                    if rating >= min_rating and price <= max_price:
                        value_score = rating / price * 10000
                        results.append({
                            'brand': brand,
                            'storage': storage,
                            'ram': ram,
                            'price': price,
                            'rating': rating,
                            'value_score': value_score
                        })
        
        return pd.DataFrame(results).sort_values('value_score', ascending=False)

def main():
    predictor = PhonePredictor()
    
    while True:
        print("\nMobile Phone Predictor")
        print("=====================")
        print("1. Predict single phone")
        print("2. Compare brands")
        print("3. Analyze feature impact")
        print("4. Find best value configurations")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            brand = input("Enter brand (e.g., SAMSUNG): ").upper()
            if brand not in predictor.available_brands:
                print(f"Error: Brand '{brand}' not found. Available brands:", ", ".join(predictor.available_brands))
                continue
            storage = float(input("Enter storage in GB (e.g., 128): "))
            ram = float(input("Enter RAM in GB (e.g., 8): "))
            camera = float(input("Enter camera in MP (e.g., 48): "))
            battery = float(input("Enter battery in mAh (e.g., 5000): "))
            
            price, rating = predictor.predict_single_phone(
                brand, storage, ram, camera, battery
            )
            
            print(f"\nPredictions for {brand} phone:")
            print(f"Storage: {storage}GB")
            print(f"RAM: {ram}GB")
            print(f"Camera: {camera}MP")
            print(f"Battery: {battery}mAh")
            print(f"Predicted Price: ₹{price:.2f}")
            print(f"Predicted Rating: {rating:.2f} stars")
        
        elif choice == '2':
            storage = float(input("Enter storage in GB (e.g., 128): "))
            ram = float(input("Enter RAM in GB (e.g., 8): "))
            camera = float(input("Enter camera in MP (e.g., 48): "))
            battery = float(input("Enter battery in mAh (e.g., 5000): "))
            
            results = predictor.compare_brands(storage, ram, camera, battery)
            results = results.sort_values('value_score', ascending=False)
            
            print("\nBrand Comparison:")
            print(results.to_string(index=False))
            
            # Save results
            results.to_csv('exports/brand_comparison.csv', index=False)
            print("\nResults saved to exports/brand_comparison.csv")
        
        elif choice == '3':
            brand = input("Enter brand to analyze (e.g., SAMSUNG): ").upper()
            impact_df = predictor.analyze_feature_impact(brand)
            
            if impact_df is None:
                continue
            
            # Save results
            impact_df.to_csv('exports/feature_impact.csv', index=False)
            predictor.plot_feature_impact(impact_df, 'visualizations/feature_impact.png')
            
            print("\nFeature impact analysis saved to:")
            print("- exports/feature_impact.csv")
            print("- visualizations/feature_impact.png")
        
        elif choice == '4':
            min_rating = float(input("Enter minimum rating (e.g., 4.0): "))
            max_price = float(input("Enter maximum price (e.g., 30000): "))
            
            results = predictor.find_best_value_configs(min_rating, max_price)
            print("\nTop 10 Best Value Configurations:")
            print(results.head(10).to_string(index=False))
            
            # Save results
            results.to_csv('exports/best_value_configs.csv', index=False)
            print("\nFull results saved to exports/best_value_configs.csv")
        
        elif choice == '5':
            print("\nThank you for using Mobile Phone Predictor!")
            break
        
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()
