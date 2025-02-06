import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO
import json
from datetime import datetime
from recommendation_engine import PhoneRecommender

# Initialize recommender
recommender = PhoneRecommender()

# Set page configuration
st.set_page_config(
    page_title="Mobile Phone Market Analyzer",
    page_icon="📱",
    layout="wide"
)

@st.cache_data
def load_phone_data():
    """Load and preprocess phone data"""
    df = pd.read_csv('mobile_dataset_cleaned.csv')
    
    # Extract brand from name
    df['brand'] = df['name'].apply(lambda x: x.split()[0])
    
    # Convert price and ratings to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
    
    return df

@st.cache_resource
def get_predictor():
    """Get or create price predictor"""
    return PhonePredictor()

class PhonePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.df = load_phone_data()
        self.train_model()
    
    def train_model(self):
        """Train the price prediction model"""
        # Prepare features
        features = ['ratings']
        X = self.df[features]
        y = self.df['price']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
    
    def predict_price(self, rating):
        """Predict phone price based on features"""
        features = [[rating]]
        features_scaled = self.scaler.transform(features)
        predicted_price = self.model.predict(features_scaled)[0]
        return predicted_price
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        features = ['Rating']
        importance = self.model.feature_importances_
        return pd.DataFrame({'Feature': features, 'Importance': importance})

# Initialize predictor
predictor = get_predictor()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Price Predictor", "Brand Comparison", "Phone Recommendations"]
)

if page == "Dashboard":
    # Main Header
    st.title("📱 Mobile Phone Market Analysis")
    
    df = load_phone_data()
    
    # Market Overview
    st.markdown("### 📊 Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Distribution
        st.subheader("Price Distribution")
        fig = px.histogram(df, x='price', nbins=30, title='Price Distribution')
        st.plotly_chart(fig)
        
    with col2:
        # Rating Distribution
        st.subheader("Rating Distribution")
        fig = px.histogram(df, x='ratings', nbins=20, title='Rating Distribution')
        st.plotly_chart(fig)
    
    # Brand Analysis
    st.markdown("### 🏢 Brand Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Brands by Average Price
        top_brands_price = df.groupby('brand').agg({
            'price': ['mean', 'count']
        }).sort_values(('price', 'mean'), ascending=False).head(10)
        
        fig = px.bar(
            x=top_brands_price.index,
            y=top_brands_price[('price', 'mean')],
            title='Top Brands by Average Price'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Top Brands by Average Rating
        top_brands_rating = df.groupby('brand').agg({
            'ratings': ['mean', 'count']
        }).sort_values(('ratings', 'mean'), ascending=False).head(10)
        
        fig = px.bar(
            x=top_brands_rating.index,
            y=top_brands_rating[('ratings', 'mean')],
            title='Top Brands by Average Rating'
        )
        st.plotly_chart(fig)

elif page == "Price Predictor":
    st.header("💰 Price Predictor")
    
    # User input
    rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    
    # Make prediction
    predicted_price = predictor.predict_price(rating)
    
    # Display prediction
    st.markdown(f"### Predicted Price: ${predicted_price:,.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = predictor.get_feature_importance()
    fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
    st.plotly_chart(fig)

elif page == "Brand Comparison":
    st.header("🔄 Brand Comparison")
    
    df = load_phone_data()
    
    # Select brands to compare
    all_brands = sorted(df['brand'].unique())
    selected_brands = st.multiselect(
        "Select brands to compare",
        all_brands,
        default=all_brands[:3]
    )
    
    if selected_brands:
        # Filter data
        brand_data = df[df['brand'].isin(selected_brands)]
        
        # Price comparison
        st.subheader("Price Comparison")
        fig = px.box(brand_data, x='brand', y='price', title='Price Distribution by Brand')
        st.plotly_chart(fig)
        
        # Rating comparison
        st.subheader("Rating Comparison")
        fig = px.box(brand_data, x='brand', y='ratings', title='Rating Distribution by Brand')
        st.plotly_chart(fig)

elif page == "Phone Recommendations":
    st.title("📱 Smart Phone Recommendations")
    
    df = load_phone_data()
    
    # Filters
    st.sidebar.header("Filters")
    
    # Price range
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=float(df['price'].min()),
        max_value=float(df['price'].max()),
        value=(float(df['price'].min()), float(df['price'].max()))
    )
    
    # Rating filter
    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=float(df['ratings'].min()),
        max_value=float(df['ratings'].max()),
        value=4.0
    )
    
    # Brand selection
    all_brands = sorted(df['brand'].unique())
    selected_brands = st.sidebar.multiselect(
        "Select Brands",
        all_brands,
        default=[]
    )
    
    # Get recommendations
    preferences = {
        'min_price': price_range[0],
        'max_price': price_range[1],
        'min_rating': min_rating,
        'brands': selected_brands
    }
    
    recommender.load_data(df)
    recommendations = recommender.get_recommendations(preferences)
    
    if len(recommendations) > 0:
        st.markdown("### 📱 Recommended Phones")
        
        # Display recommendations
        for _, phone in recommendations.iterrows():
            st.markdown(f"""
            **{phone['name']}**
            - Price: ${phone['price']:,.2f}
            - Rating: {phone['ratings']:.1f}/5.0
            """)
        
        # Visual comparison
        st.markdown("### 📊 Visual Comparison")
        
        fig = px.scatter(
            recommendations,
            x='price',
            y='ratings',
            text='name',
            title='Price vs Rating',
            labels={'price': 'Price ($)', 'ratings': 'Rating'}
        )
        st.plotly_chart(fig)
    else:
        st.warning("No phones found matching your criteria. Try adjusting the filters.")

# Competitive Analysis
st.markdown("### 🔄 Competitive Analysis")

# Load data
df = load_phone_data()

# Calculate market position scores
df['price_normalized'] = (df['price'] - df['price'].mean()) / df['price'].std()
df['rating_normalized'] = (df['ratings'] - df['ratings'].mean()) / df['ratings'].std()
df['market_position'] = df['rating_normalized'] - df['price_normalized']

# Scale market position to positive values for bubble size
df['bubble_size'] = (df['market_position'] - df['market_position'].min() + 1) * 20

st.markdown("#### 📊 Value Proposition Analysis")
col1, col2 = st.columns(2)

with col1:
    # Value Proposition Map
    fig = px.scatter(
        df,
        x='price',
        y='ratings',
        color='brand',
        size='bubble_size',
        title='Value Proposition Map',
        labels={
            'price': 'Price ($)',
            'ratings': 'Rating',
            'market_position': 'Market Position'
        },
        hover_data={
            'name': True,
            'bubble_size': False,
            'market_position': ':.2f'
        }
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Price ($)",
        yaxis_title="Rating",
        hovermode='closest'
    )
    
    fig.update_traces(
        hovertemplate="<br>".join([
            "Brand: %{customdata[0]}",
            "Model: %{customdata[1]}",
            "Price: $%{x:,.0f}",
            "Rating: %{y:.2f}⭐",
            "Market Position: %{customdata[2]:.2f}"
        ])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ##### 📝 Interpretation
    - **Bubble Size**: Larger bubbles indicate better value proposition
    - **Position**: Top-left is ideal (high rating, lower price)
    - **Color**: Different colors represent different brands
    """)

with col2:
    # Top Value Propositions
    top_value = df.nlargest(10, 'market_position')[
        ['brand', 'name', 'price', 'ratings', 'market_position']
    ]
    
    st.markdown("#### 🏆 Top 10 Value Propositions")
    st.dataframe(
        top_value.style.format({
            'price': '$ {:,.0f}',
            'ratings': '{:.2f}⭐',
            'market_position': '{:.2f}'
        }).background_gradient(subset=['market_position'], cmap='RdYlGn')
    )
    
    st.markdown("""
    ##### 📝 Value Score Interpretation
    - **Higher Score**: Better value for money (high rating, reasonable price)
    - **Lower Score**: Lower value for money (lower rating, higher price)
    - **Color Scale**: Green indicates better value proposition
    """)

# Brand Positioning Analysis
st.markdown("### 🎯 Brand Positioning Analysis")

# Calculate brand positioning metrics
brand_positioning = df.groupby('brand').agg({
    'price': ['mean', 'std'],
    'ratings': ['mean', 'std'],
    'market_position': 'mean',
    'name': 'count'
}).round(2)

brand_positioning.columns = [
    'Avg Price', 'Price Std', 'Avg Rating', 'Rating Std',
    'Market Position', 'Model Count'
]

# Filter top 15 brands by model count
top_brands = brand_positioning.nlargest(15, 'Model Count')

col1, col2 = st.columns(2)

with col1:
    # Brand Position Map
    brand_data = top_brands.reset_index()
    
    # Create hover text
    brand_data['hover_text'] = brand_data.apply(
        lambda row: f"Brand: {row['brand']}<br>" +
                   f"Average Price: ${row['Avg Price']:,.0f}<br>" +
                   f"Average Rating: {row['Avg Rating']:.2f}⭐<br>" +
                   f"Models: {row['Model Count']}<br>" +
                   f"Market Position: {row['Market Position']:.2f}",
        axis=1
    )
    
    fig = px.scatter(
        brand_data,
        x='Avg Price',
        y='Avg Rating',
        size='Model Count',
        color='Market Position',
        text='brand',
        title='Brand Position Map',
        labels={
            'Avg Price': 'Average Price ($)',
            'Avg Rating': 'Average Rating'
        },
        hover_name='brand',
        hover_data={
            'hover_text': True,
            'Avg Price': False,
            'Avg Rating': False,
            'Model Count': False,
            'Market Position': False,
            'brand': False
        },
        size_max=50
    )
    
    fig.update_traces(
        textposition='top center',
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Average Price ($)",
        yaxis_title="Average Rating",
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ##### 📝 Interpretation
    - **Size**: Larger bubbles indicate more models
    - **Color**: Red to green indicates market position (value for money)
    - **Position**: Top-left is ideal (high rating, lower price)
    """)

with col2:
    # Brand Consistency Analysis
    st.markdown("#### 📊 Brand Consistency Analysis")
    st.dataframe(
        top_brands.style.format({
            'Avg Price': '$ {:,.0f}',
            'Price Std': '$ {:,.0f}',
            'Avg Rating': '{:.2f}⭐',
            'Rating Std': '{:.2f}',
            'Market Position': '{:.2f}',
            'Model Count': '{:,.0f}'
        }).background_gradient(subset=['Market Position'], cmap='RdYlGn')
    )
    
    st.markdown("""
    ##### 📝 Metrics Interpretation
    - **Price Std**: Lower values indicate consistent pricing
    - **Rating Std**: Lower values indicate consistent quality
    - **Market Position**: Higher values indicate better value proposition
    - **Model Count**: Number of models in the market
    """)

# Price Segment Competition
st.subheader("Price Segment Competition")

# Create price segments
df['price_bracket'] = pd.qcut(
    df['price'],
    q=4,
    labels=['Budget', 'Mid-Range', 'Premium', 'Ultra Premium']
)

# Calculate segment metrics
segment_competition = df.groupby(['price_bracket', 'brand']).size().unstack(fill_value=0)
segment_share = segment_competition.div(segment_competition.sum(axis=1), axis=0) * 100

# Plot segment competition
fig = px.bar(
    segment_share.reset_index().melt(id_vars=['price_bracket']),
    x='price_bracket',
    y='value',
    color='brand',
    title='Brand Competition by Price Segment',
    labels={
        'price_bracket': 'Price Segment',
        'value': 'Market Share (%)',
        'brand': 'Brand'
    }
)

fig.update_layout(
    barmode='stack',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.02
    ),
    margin=dict(r=150)
)

st.plotly_chart(fig, use_container_width=True)

# Market Concentration
st.subheader("Market Concentration")

col1, col2 = st.columns(2)

with col1:
    # Brand Concentration by Segment
    segment_brands = df.groupby('price_bracket')['brand'].nunique()
    segment_models = df.groupby('price_bracket').size()
    
    concentration_data = pd.DataFrame({
        'Unique Brands': segment_brands,
        'Total Models': segment_models,
        'Models per Brand': (segment_models / segment_brands).round(2)
    })
    
    st.write("Segment Concentration Metrics")
    st.dataframe(
        concentration_data.style.format({
            'Unique Brands': '{:,.0f}',
            'Total Models': '{:,.0f}',
            'Models per Brand': '{:.2f}'
        }).background_gradient(cmap='YlOrRd')
    )

with col2:
    # Market Concentration Ratio
    top_brand_share = (df['brand'].value_counts() / len(df) * 100).head(5)
    
    fig = px.bar(
        x=top_brand_share.index,
        y=top_brand_share.values,
        title='Top 5 Brands Market Concentration',
        labels={'x': 'Brand', 'y': 'Market Share (%)'}
    )
    
    fig.update_traces(
        text=top_brand_share.values.round(1),
        texttemplate='%{text}%',
        textposition='outside',
        hovertemplate="Brand: %{x}<br>Share: %{y:.1f}%<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and Machine Learning</p>
        <p> 2025 Mobile Phone Analyzer Pro</p>
    </div>
""", unsafe_allow_html=True)
