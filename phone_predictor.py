from mobile_phone_ml import MobilePhoneML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

class PhonePredictor:
    def __init__(self):
        # Load data from CSV
        self.df = pd.read_csv('mobile_dataset_cleaned.csv')
        
        # Extract brand from name
        self.df['brand'] = self.df['name'].apply(lambda x: x.split()[0])
        
        # Convert price and ratings to numeric
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df['ratings'] = pd.to_numeric(self.df['ratings'], errors='coerce')
        
        # Get available brands
        self.available_brands = sorted(self.df['brand'].unique())
        print("\nAvailable brands:", ", ".join(self.available_brands))
    
    def predict_price(self, rating):
        """Predict price based on rating"""
        # Simple linear regression
        slope = self.df['price'].cov(self.df['ratings']) / self.df['ratings'].var()
        intercept = self.df['price'].mean() - slope * self.df['ratings'].mean()
        
        predicted_price = slope * rating + intercept
        return max(0, predicted_price)  # Ensure non-negative price
    
    def predict_rating(self, price):
        """Predict rating based on price"""
        # Simple linear regression
        slope = self.df['ratings'].cov(self.df['price']) / self.df['price'].var()
        intercept = self.df['ratings'].mean() - slope * self.df['price'].mean()
        
        predicted_rating = slope * price + intercept
        return min(max(1, predicted_rating), 5)  # Ensure rating is between 1 and 5
    
    def compare_brands(self, min_price=None, max_price=None, min_rating=None):
        """Compare brands based on price and rating ranges"""
        filtered_df = self.df.copy()
        
        if min_price is not None:
            filtered_df = filtered_df[filtered_df['price'] >= min_price]
        if max_price is not None:
            filtered_df = filtered_df[filtered_df['price'] <= max_price]
        if min_rating is not None:
            filtered_df = filtered_df[filtered_df['ratings'] >= min_rating]
        
        brand_stats = filtered_df.groupby('brand').agg({
            'price': ['mean', 'min', 'max'],
            'ratings': ['mean', 'min', 'max'],
            'name': 'count'
        }).round(2)
        
        brand_stats.columns = [
            'avg_price', 'min_price', 'max_price',
            'avg_rating', 'min_rating', 'max_rating',
            'model_count'
        ]
        
        # Calculate value score (higher rating and lower price is better)
        brand_stats['value_score'] = (brand_stats['avg_rating'] / brand_stats['avg_price']) * 10000
        
        return brand_stats.sort_values('value_score', ascending=False)
    
    def get_recommendations(self, budget=None, min_rating=None, brands=None):
        """Get phone recommendations based on criteria"""
        filtered_df = self.df.copy()
        
        if budget is not None:
            filtered_df = filtered_df[filtered_df['price'] <= budget]
        if min_rating is not None:
            filtered_df = filtered_df[filtered_df['ratings'] >= min_rating]
        if brands is not None and len(brands) > 0:
            filtered_df = filtered_df[filtered_df['brand'].isin(brands)]
        
        # Calculate value score
        filtered_df['value_score'] = (filtered_df['ratings'] / filtered_df['price']) * 10000
        
        # Get top 10 recommendations
        recommendations = filtered_df.nlargest(10, 'value_score')
        return recommendations[['name', 'brand', 'price', 'ratings', 'value_score']]

def main():
    predictor = PhonePredictor()
    
    while True:
        print("\nPhone Predictor Menu:")
        print("1. Predict Price")
        print("2. Predict Rating")
        print("3. Compare Brands")
        print("4. Get Recommendations")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            try:
                rating = float(input("Enter rating (1-5): "))
                predicted_price = predictor.predict_price(rating)
                print(f"\nPredicted Price: ${predicted_price:,.2f}")
            except ValueError as e:
                print("Error:", str(e))
        
        elif choice == '2':
            try:
                price = float(input("Enter price ($): "))
                predicted_rating = predictor.predict_rating(price)
                print(f"\nPredicted Rating: {predicted_rating:.1f}/5.0")
            except ValueError as e:
                print("Error:", str(e))
        
        elif choice == '3':
            try:
                min_price = float(input("Enter minimum price (or press Enter for none): ") or 0)
                max_price = float(input("Enter maximum price (or press Enter for none): ") or float('inf'))
                min_rating = float(input("Enter minimum rating (or press Enter for none): ") or 0)
                
                results = predictor.compare_brands(min_price, max_price, min_rating)
                print("\nBrand Comparison Results:")
                print(results)
            except ValueError as e:
                print("Error:", str(e))
        
        elif choice == '4':
            try:
                budget = float(input("Enter your budget (or press Enter for none): ") or None)
                min_rating = float(input("Enter minimum rating (or press Enter for none): ") or None)
                
                print("\nAvailable brands:", ", ".join(predictor.available_brands))
                brands_input = input("Enter brands to filter (comma-separated, or press Enter for all): ")
                brands = [b.strip() for b in brands_input.split(',')] if brands_input.strip() else None
                
                recommendations = predictor.get_recommendations(budget, min_rating, brands)
                print("\nTop Recommendations:")
                print(recommendations)
            except ValueError as e:
                print("Error:", str(e))
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
