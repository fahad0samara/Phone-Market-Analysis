import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class PhoneRecommender:
    def __init__(self):
        self.df = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()
    
    def load_data(self, df):
        """Load and preprocess the phone data"""
        self.df = df.copy()
        
        # Create feature matrix
        features = ['price', 'ratings']
        self.feature_matrix = self.df[features].copy()
        
        # Scale features
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
    
    def get_recommendations(self, preferences, n=10):
        """Get phone recommendations based on user preferences"""
        if self.df is None:
            raise ValueError("Model not trained. Call load_data first.")
        
        filtered_df = self.df.copy()
        
        # Apply filters
        if preferences.get('min_price') is not None:
            filtered_df = filtered_df[filtered_df['price'] >= preferences['min_price']]
        if preferences.get('max_price') is not None:
            filtered_df = filtered_df[filtered_df['price'] <= preferences['max_price']]
        if preferences.get('min_rating') is not None:
            filtered_df = filtered_df[filtered_df['ratings'] >= preferences['min_rating']]
        if preferences.get('brands') is not None and len(preferences['brands']) > 0:
            filtered_df = filtered_df[filtered_df['brand'].isin(preferences['brands'])]
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        # Calculate value score
        filtered_df['price_norm'] = (filtered_df['price'] - filtered_df['price'].mean()) / filtered_df['price'].std()
        filtered_df['rating_norm'] = (filtered_df['ratings'] - filtered_df['ratings'].mean()) / filtered_df['ratings'].std()
        
        # Calculate weighted score
        filtered_df['value_score'] = (
            -0.5 * filtered_df['price_norm'] +  # Lower price is better
            0.5 * filtered_df['rating_norm']    # Higher rating is better
        )
        
        # Get top recommendations
        recommendations = filtered_df.nlargest(n, 'value_score')
        
        return recommendations
    
    def get_price_segments(self):
        """Get price segment information"""
        if self.df is None:
            raise ValueError("Model not trained. Call load_data first.")
        
        # Create price segments
        self.df['price_segment'] = pd.qcut(
            self.df['price'],
            q=4,
            labels=['Budget', 'Mid-Range', 'Premium', 'Ultra Premium']
        )
        
        # Get segment statistics
        segments = self.df.groupby('price_segment').agg({
            'price': ['min', 'max', 'mean', 'count'],
            'ratings': 'mean'
        }).round(2)
        
        segments.columns = ['Min Price', 'Max Price', 'Avg Price', 'Count', 'Avg Rating']
        return segments
    
    def analyze_brand(self, brand):
        """Get brand analysis"""
        if self.df is None:
            raise ValueError("Model not trained. Call load_data first.")
        
        brand_df = self.df[self.df['brand'] == brand]
        
        if len(brand_df) == 0:
            return None
        
        analysis = {
            'model_count': len(brand_df),
            'avg_price': brand_df['price'].mean(),
            'avg_rating': brand_df['ratings'].mean(),
            'price_range': (brand_df['price'].min(), brand_df['price'].max())
        }
        
        return analysis
