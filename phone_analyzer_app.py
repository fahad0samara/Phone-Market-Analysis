import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mobile_phone_ml import MobilePhoneML
from phone_predictor import PhonePredictor
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import base64
from io import BytesIO
import json
from datetime import datetime
from recommendation_engine import PhoneRecommender

# Initialize recommender
recommender = PhoneRecommender()

# Set page configuration
st.set_page_config(
    page_title="Mobile Phone Analyzer",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def get_predictor():
    return PhonePredictor()

predictor = get_predictor()

# Database connection
@st.cache_resource
def get_engine():
    return create_engine('sqlite:///database/mobile_phones.db')

engine = get_engine()

# Load database data
@st.cache_data
def load_phone_data():
    """Load and preprocess phone data"""
    df = pd.read_sql('SELECT * FROM mobile_phones', engine)
    
    # Convert storage and RAM to numeric values
    df['storage'] = pd.to_numeric(df['storage'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['ram'] = pd.to_numeric(df['ram'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
    
    # Drop rows with NaN values in essential columns
    df = df.dropna(subset=['price', 'ratings'])
    
    return df

# Helper functions
def create_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def plot_correlation_matrix(df, features):
    """Create a correlation matrix plot"""
    # Ensure all feature names exist in the dataframe
    valid_features = [col for col in features if col in df.columns]
    if not valid_features:
        st.error("No valid features found for correlation analysis")
        return None
        
    corr = df[valid_features].corr()
    fig = px.imshow(corr,
                    labels=dict(color="Correlation"),
                    title="Feature Correlation Matrix")
    return fig


# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Price Predictor", "Brand Comparison", "Feature Analysis", 
     "Best Value Finder", "Market Trends", "Performance Analysis", "Custom Analysis", "Phone Recommendations"]
)

if page == "Dashboard":
    # Main Header
    st.title("📱 Mobile Phone Market Analysis")
    
    df = load_phone_data()

    # Market Overview in a clean card-like container
    st.markdown("### 📊 Market Overview")
    with st.container():
        st.markdown("""
        <style>
        .metric-row {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        
        # First Row of Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_models = len(df)
            total_brands = df['brand'].nunique()
            st.metric(
                "Market Size",
                f"{total_models:,} Models",
                f"{total_brands} Active Brands"
            )
        
        with col2:
            avg_price = df['price'].mean()
            price_std = df['price'].std()
            st.metric(
                "Price Metrics",
                f"₹{avg_price:,.0f} Avg",
                f"₹{price_std:,.0f} Std Dev",
                delta_color="off"
            )
        
        with col3:
            avg_rating = df['ratings'].mean()
            rating_std = df['ratings'].std()
            st.metric(
                "Rating Metrics",
                f"{avg_rating:.2f}⭐ Avg",
                f"±{rating_std:.2f} Std Dev",
                delta_color="off"
            )
        
        # Second Row of Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            models_per_brand = len(df) / df['brand'].nunique()
            st.metric(
                "Brand Diversity",
                f"{models_per_brand:.1f}",
                "Models per Brand",
                delta_color="off"
            )
        
        with col2:
            price_range = f"₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}"
            median_price = df['price'].median()
            st.metric(
                "Price Range",
                price_range,
                f"Median: ₹{median_price:,.0f}",
                delta_color="off"
            )
        
        with col3:
            top_rated = df.nlargest(1, 'ratings')
            st.metric(
                "Top Rating",
                f"{top_rated['ratings'].iloc[0]:.2f}⭐",
                f"{top_rated['brand'].iloc[0]} {top_rated['name'].iloc[0]}"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Market Share Analysis
    st.markdown("### 📈 Market Share Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall market share metrics
        brand_share = df['brand'].value_counts()
        total_models = len(df)
        
        # Calculate market concentration metrics
        hhi_index = ((brand_share / total_models * 100) ** 2).sum() / 10000
        top_2_concentration = (brand_share.head(2).sum() / total_models * 100).round(1)
        top_5_concentration = (brand_share.head(5).sum() / total_models * 100).round(1)
        
        st.markdown("#### 📊 Market Concentration")
        st.markdown(f"- **HHI Index**: {hhi_index:.2f} ({'High' if hhi_index > 0.25 else 'Moderate' if hhi_index > 0.15 else 'Low'} concentration)")
        st.markdown(f"- **CR2**: {top_2_concentration}% (Top 2 brands)")
        st.markdown(f"- **CR5**: {top_5_concentration}% (Top 5 brands)")
        
        # Add interpretation
        st.markdown("""
        #### 📝 Interpretation
        - **HHI** < 0.15: Low concentration
        - **HHI** 0.15-0.25: Moderate concentration
        - **HHI** > 0.25: High concentration
        """)
    
    with col2:
        # Brand Market Share Visualization
        fig = px.pie(
            values=brand_share.head(10),
            names=brand_share.head(10).index,
            title='Top 10 Brands Market Share',
            hole=0.4
        )
        fig.update_layout(
            height=400,
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
    
    # Add a divider
    st.markdown("---")
    

        
    # Price Segment Analysis
    st.markdown("### 💰 Price Segment Analysis")
    
    # Create price segments
    df['price_segment'] = pd.qcut(
        df['price'].fillna(df['price'].median()), 
        q=4,
        labels=['Budget', 'Mid-Range', 'Premium', 'Ultra Premium']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate brand distribution in each segment
        segment_dist = pd.crosstab(
            df['brand'], 
            df['price_segment'], 
            normalize='index'
        ) * 100
        
        # Create stacked bar chart
        fig = go.Figure()
        
        for segment in segment_dist.columns:
            fig.add_trace(go.Bar(
                name=segment,
                x=segment_dist.index,
                y=segment_dist[segment],
                hovertemplate=(
                    "Brand: %{x}<br>" +
                    "Segment: " + segment + "<br>" +
                    "Percentage: %{y:.1f}%<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title='Brand Distribution Across Price Segments',
            yaxis_title='Percentage of Models',
            barmode='stack',
            height=500,
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
    
    with col2:
        # Segment-wise market share
        segment_market_share = pd.crosstab(
            df['price_segment'],
            df['brand']
        ).apply(lambda x: x / x.sum() * 100, axis=1)
        
        # Get top 3 brands in each segment
        top_brands_by_segment = {}
        for segment in segment_market_share.index:
            top_3 = segment_market_share.loc[segment].nlargest(3)
            top_brands_by_segment[segment] = [
                f"{brand}: {share:.1f}%" 
                for brand, share in top_3.items()
            ]
        
        st.markdown("#### 🏆 Market Leaders by Segment")
        for segment, brands in top_brands_by_segment.items():
            st.markdown(f"**{segment}**")
            for brand in brands:
                st.markdown(f"- {brand}")
            st.markdown("")  # Add space between segments
    



    # Market Insights
    st.subheader("📈 Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand Market Share
        brand_share = df['brand'].value_counts().head(10)
        total_phones = len(df)
        brand_share_pct = (brand_share / total_phones * 100).round(1)
        
        fig = px.pie(
            values=brand_share_pct,
            names=brand_share.index,
            title='Top 10 Brands Market Share (%)',
            hole=0.4
        )
        fig.update_traces(
            textposition='inside',
            textinfo='label+percent',
            hovertemplate="Brand: %{label}<br>Share: %{percent}<br>Models: %{value:.1f}%<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price Segments Distribution
        df['price_segment'] = pd.qcut(
            df['price'],
            q=4,
            labels=['Budget', 'Mid-Range', 'Premium', 'Ultra Premium']
        )
        segment_dist = df['price_segment'].value_counts()
        
        fig = px.pie(
            values=segment_dist,
            names=segment_dist.index,
            title='Market Distribution by Price Segment',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_traces(
            textposition='inside',
            textinfo='label+percent',
            hovertemplate="Segment: %{label}<br>Share: %{percent}<br>Models: %{value}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Brand Analysis
    st.subheader("Brand Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Average Price by Top Brands
        top_brands_price = df.groupby('brand')['price'].agg(['mean', 'count'])
        top_brands_price = top_brands_price.sort_values('mean', ascending=False).head(10)
        
        fig = px.bar(
            x=top_brands_price.index,
            y=top_brands_price['mean'],
            title='Average Price by Top Brands',
            labels={'x': 'Brand', 'y': 'Average Price (₹)'},
            text=top_brands_price['count']
        )
        fig.update_traces(
            texttemplate='%{text} models',
            textposition='outside',
            hovertemplate="Brand: %{x}<br>Avg Price: ₹%{y:,.0f}<br>Models: %{text}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average Rating by Top Brands
        top_brands_rating = df.groupby('brand').agg({
            'ratings': ['mean', 'count']
        }).sort_values(('ratings', 'mean'), ascending=False).head(10)
        
        fig = px.bar(
            x=top_brands_rating.index,
            y=top_brands_rating[('ratings', 'mean')],
            title='Average Rating by Top Brands',
            labels={'x': 'Brand', 'y': 'Average Rating'},
            text=top_brands_rating[('ratings', 'count')]
        )
        fig.update_traces(
            texttemplate='%{text} models',
            textposition='outside',
            hovertemplate="Brand: %{x}<br>Avg Rating: %{y:.2f}<br>Models: %{text}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Price Range Analysis
    st.subheader("Price Range Analysis")
    
    # Create detailed price ranges
    price_ranges = [0, 10000, 20000, 30000, 50000, float('inf')]
    price_labels = ['Budget', 'Lower Mid', 'Upper Mid', 'Premium', 'Ultra Premium']
    df['price_range'] = pd.cut(df['price'], bins=price_ranges, labels=price_labels)
    
    # Calculate statistics by price range
    price_stats = df.groupby('price_range').agg({
        'price': ['count', 'mean'],
        'ratings': 'mean',
        'ram': 'mean',
        'storage': 'mean'
    }).round(2)
    
    # Format the stats
    price_stats.columns = ['Count', 'Avg Price', 'Avg Rating', 'Avg RAM', 'Avg Storage']
    st.dataframe(
        price_stats.style.format({
            'Count': '{:,.0f}',
            'Avg Price': '₹{:,.0f}',
            'Avg Rating': '{:.2f}⭐',
            'Avg RAM': '{:.1f} GB',
            'Avg Storage': '{:.0f} GB'
        }).background_gradient(subset=['Avg Rating'], cmap='RdYlGn')
    )
    

    
     
    

    
    # Create price ranges
    df['price_range'] = pd.cut(
        df['price'],
        bins=[0, 10000, 20000, 30000, 50000, float('inf')],
        labels=['Budget', 'Lower Mid', 'Upper Mid', 'Premium', 'Ultra Premium']
    )
    
    price_range_stats = df.groupby('price_range').agg({
        'price': ['count', 'mean'],
        'ratings': 'mean'
    }).round(2)
    
    price_range_stats.columns = ['Count', 'Avg Price', 'Avg Rating']
    
    # Format the stats
    st.dataframe(
        price_range_stats.style.format({
            'Count': '{:,.0f}',
            'Avg Price': '₹{:,.0f}',
            'Avg Rating': '{:.2f}'
        }).background_gradient(subset=['Avg Rating'], cmap='RdYlGn')
    )
    
    # Top Rated Phones
    st.subheader("Top Rated Phones")
    
    top_rated = df.nlargest(10, 'ratings')[
        ['brand', 'name', 'price', 'ratings', 'ram', 'storage']
    ]
    
    st.dataframe(
        top_rated.style.format({
            'price': '₹{:,.0f}',
            'ratings': '{:.2f}',
            'ram': '{:.0f} GB',
            'storage': '{:.0f} GB'
        }).background_gradient(subset=['ratings'], cmap='RdYlGn')
    )
    
    # Best Value Phones
    st.subheader("Best Value Phones")
    
    df['value_score'] = df['ratings'] / df['price'] * 10000
    best_value = df.nlargest(10, 'value_score')[
        ['brand', 'name', 'price', 'ratings', 'value_score']
    ]
    
    st.dataframe(
        best_value.style.format({
            'price': '₹{:,.0f}',
            'ratings': '{:.2f}',
            'value_score': '{:.4f}'
        }).background_gradient(subset=['value_score'], cmap='RdYlGn')
    )

 
    # Rating Analysis
    st.subheader("Rating Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating Distribution by Brand
        fig = px.box(
            df,
            x='brand',
            y='ratings',
            title='Rating Distribution by Brand',
            labels={'brand': 'Brand', 'ratings': 'Rating'},
            points="outliers"
        )
        
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average Rating vs Price Range
        df['price_bracket'] = pd.qcut(
            df['price'],
            q=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        price_rating = df.groupby('price_bracket')['ratings'].mean().reset_index()
        
        fig = px.bar(
            price_rating,
            x='price_bracket',
            y='ratings',
            title='Average Rating by Price Range',
            labels={'price_bracket': 'Price Range', 'ratings': 'Average Rating'}
        )
        
        fig.update_traces(
            texttemplate='%{y:.2f}',
            textposition='outside',
            hovertemplate="Price Range: %{x}<br>Avg Rating: %{y:.2f}<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
   
    st.header("🔄 Brand Comparison")
    
    df = load_phone_data()
    
    # Create tabs for different comparison types
    tab1, tab2 = st.tabs(["Specification-based", "Market Analysis"])
    
    with tab1:
        st.subheader("Specification-based")
        
        # Input fields for specifications
        col1, col2 = st.columns(2)
        with col1:
            storage = st.number_input("Storage (GB)", min_value=0, value=128)
            ram = st.number_input("RAM (GB)", min_value=0, value=8)
        with col2:
            camera = st.number_input("Camera (MP)", min_value=0, value=48)
            battery = st.number_input("Battery (mAh)", min_value=0, value=5000)
        
        # Filter data based on specifications
        df_filtered = df.copy()
        df_filtered['match_score'] = 0
        
        # Calculate match score based on specifications
        if 'storage' in df.columns:
            df_filtered.loc[df_filtered['storage'] == storage, 'match_score'] += 1
        if 'ram' in df.columns:
            df_filtered.loc[df_filtered['ram'] == ram, 'match_score'] += 1
            
        # Get top matches
        df_matches = df_filtered.sort_values('match_score', ascending=False).head(10)
        
        if not df_matches.empty:
            st.write("Top Matching Models:")
            st.dataframe(df_matches[['brand', 'name', 'price', 'ratings', 'storage', 'ram']])
    
    with tab2:
        st.subheader("Market Analysis")
        
        # Price Distribution Analysis
        st.subheader("Price Distribution by Brand")
        
        # Add price range filter
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Minimum Price (₹)", 
                                      value=int(df['price'].min()),
                                      step=1000)
        with col2:
            max_price = st.number_input("Maximum Price (₹)", 
                                      value=int(df['price'].max()),
                                      step=1000)
        
        # Filter data based on price range
        df_filtered = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
        
        # Chart type selector
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Violin Plot", "Box Plot", "Histogram"]
        )
        
        if chart_type == "Violin Plot":
            fig = go.Figure()
            for brand in df_filtered['brand'].unique():
                brand_data = df_filtered[df_filtered['brand'] == brand]
                
                fig.add_trace(go.Violin(
                    y=brand_data['price'].dropna(),
                    name=brand,
                    box_visible=True,
                    meanline_visible=True,
                    points="outliers",
                    hovertemplate=(
                        "Brand: " + brand + "<br>" +
                        "Price: ₹%{y:,.0f}<br>" +
                        "<extra></extra>"
                    )
                ))
            
        elif chart_type == "Box Plot":
            fig = go.Figure()
            for brand in df_filtered['brand'].unique():
                brand_data = df_filtered[df_filtered['brand'] == brand]
                
                fig.add_trace(go.Box(
                    y=brand_data['price'].dropna(),
                    name=brand,
                    boxpoints='outliers',
                    hovertemplate=(
                        "Brand: " + brand + "<br>" +
                        "Price: ₹%{y:,.0f}<br>" +
                        "<extra></extra>"
                    )
                ))
                
        else:  # Histogram
            fig = go.Figure()
            for brand in df_filtered['brand'].unique():
                brand_data = df_filtered[df_filtered['brand'] == brand]
                
                fig.add_trace(go.Histogram(
                    y=brand_data['price'].dropna(),
                    name=brand,
                    opacity=0.7,
                    hovertemplate=(
                        "Brand: " + brand + "<br>" +
                        "Price Range: ₹%{y:,.0f}<br>" +
                        "Count: %{x}<br>" +
                        "<extra></extra>"
                    )
                ))
            
            fig.update_layout(barmode='overlay')
        
        fig.update_layout(
            title=f'Price Distribution by Brand ({chart_type})',
            yaxis_title='Price (₹)',
            showlegend=True,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics for the selected price range
        st.subheader(f"Summary Statistics (₹{min_price:,} - ₹{max_price:,})")
        
        stats = df_filtered.groupby('brand')['price'].agg([
            ('Count', 'count'),
            ('Median', 'median'),
            ('Mean', 'mean')
        ]).round(2)
        
        stats = stats.sort_values('Count', ascending=False)
        
        st.dataframe(
            stats.style.format({
                'Count': '{:,.0f}',
                'Median': '₹{:,.0f}',
                'Mean': '₹{:,.0f}'
            })
        )
        
        # Price segment analysis
        st.subheader("Price Segment Analysis")
        
        # Create price segments
        df['price_segment'] = pd.qcut(
            df['price'].fillna(df['price'].median()), 
            q=4,
            labels=['Budget', 'Mid-Range', 'Premium', 'Ultra Premium']
        )
        
        # Calculate brand distribution in each segment
        segment_dist = pd.crosstab(
            df['brand'], 
            df['price_segment'], 
            normalize='index'
        ) * 100
        
        # Create stacked bar chart
        fig = go.Figure()
        
        for segment in segment_dist.columns:
            fig.add_trace(go.Bar(
                name=segment,
                x=segment_dist.index,
                y=segment_dist[segment],
                hovertemplate=(
                    "Brand: %{x}<br>" +
                    "Segment: " + segment + "<br>" +
                    "Percentage: %{y:.1f}%<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title='Brand Distribution Across Price Segments',
            yaxis_title='Percentage of Models',
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
        
        # Price statistics table
        st.subheader("Price Statistics by Brand")
        
        price_stats = df.groupby('brand')['price'].agg([
            ('Count', 'count'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std Dev', 'std')
        ]).round(2)
        
        # Sort by count
        price_stats = price_stats.sort_values('Count', ascending=False)
        
        # Display with formatting
        st.dataframe(
            price_stats.style.format({
                'Count': '{:,.0f}',
                'Min': '₹{:,.0f}',
                'Max': '₹{:,.0f}',
                'Mean': '₹{:,.0f}',
                'Median': '₹{:,.0f}',
                'Std Dev': '₹{:,.0f}'
            })
        )
        
        # Create scatter plot for phones with valid price and ratings
        st.subheader("Price vs Rating Analysis")
        df_scatter = df[df['price'].notna() & df['ratings'].notna()].copy()
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each brand
        for brand in df_scatter['brand'].unique():
            brand_data = df_scatter[df_scatter['brand'] == brand]
            
            fig.add_trace(go.Scatter(
                x=brand_data['price'],
                y=brand_data['ratings'],
                name=brand,
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.7
                ),
                hovertemplate=(
                    "Brand: " + brand + "<br>" +
                    "Price: ₹%{x:,.0f}<br>" +
                    "Rating: %{y:.2f}<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Update layout
        fig.update_layout(
            title='Price vs Rating by Brand',
            xaxis_title='Price (₹)',
            yaxis_title='Rating',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)  # Add right margin for legend
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Value score analysis
        st.subheader("Value Score Analysis")
        
        df_scatter['value_score'] = df_scatter['ratings'] / df_scatter['price'] * 10000
        
        # Calculate and display top value phones
        st.write("Top Value Phones by Brand:")
        
        top_value = (df_scatter.sort_values('value_score', ascending=False)
                    .groupby('brand')
                    .first()
                    .sort_values('value_score', ascending=False))
        
        st.dataframe(
            top_value[['name', 'price', 'ratings', 'value_score']]
            .style
            .format({
                'price': '₹{:,.0f}',
                'ratings': '{:.2f}',
                'value_score': '{:.2f}'
            })
            .background_gradient(subset=['value_score'], cmap='RdYlGn')
        )

elif page == "Feature Analysis":
    st.header("📊 Feature Analysis")
    
    tab1, tab2 = st.tabs(["Single Brand Analysis", "Cross-Brand Analysis"])
    
    with tab1:
        brand = st.selectbox("Select Brand for Analysis", predictor.available_brands)
        
        if st.button("Analyze Features"):
            impact_df = predictor.analyze_feature_impact(brand)
            
            if impact_df is not None:
                features = impact_df['feature'].unique()
                
                for feature in features:
                    st.subheader(f"{feature} Impact Analysis")
                    feature_data = impact_df[impact_df['feature'] == feature]
                    
                    # Create interactive 3D plot
                    fig = go.Figure(data=[go.Scatter3d(
                        x=feature_data['value'],
                        y=feature_data['price'],
                        z=feature_data['rating'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=feature_data['value'],
                            colorscale='Viridis',
                        ),
                        text=[f"{feature}: {v}<br>Price: ₹{p:,.2f}<br>Rating: {r:.2f}"
                              for v, p, r in zip(feature_data['value'],
                                               feature_data['price'],
                                               feature_data['rating'])],
                        hoverinfo='text'
                    )])
                    
                    fig.update_layout(
                        title=f'{feature} Impact on Price and Rating',
                        scene=dict(
                            xaxis_title=feature,
                            yaxis_title='Price (₹)',
                            zaxis_title='Rating'
                        )
                    )
                    
                    st.plotly_chart(fig)
    
    with tab2:
        df = load_phone_data()
        
        # Correlation analysis
        st.subheader("Feature Correlation Analysis")
        features = ['price', 'ratings', 'storage', 'ram']
        fig = plot_correlation_matrix(df, features)
        st.plotly_chart(fig)
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        importance_df = predictor.ml_system.get_feature_importance()
        fig = px.bar(importance_df, x='feature', y='importance',
                    title='Feature Importance in Price Prediction')
        st.plotly_chart(fig)

elif page == "Best Value Finder":
    st.header("🎯 Best Value Finder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_rating = st.slider("Minimum Rating", 3.0, 5.0, 4.0, 0.1)
        max_price = st.slider("Maximum Price (₹)", 10000, 50000, 30000, 1000)
    
    with col2:
        selected_brands = st.multiselect("Select Brands (optional)",
                                       predictor.available_brands)
    
    if st.button("Find Best Value Phones"):
        results = predictor.find_best_value_configs(min_rating, max_price)
        
        if selected_brands:
            results = results[results['brand'].isin(selected_brands)]
        
        # Interactive scatter plot
        st.subheader("Price-Rating-Value Analysis")
        fig = px.scatter_3d(results,
                           x='price', y='rating', z='value_score',
                           color='brand',
                           size='value_score',
                           hover_data=['storage', 'ram'],
                           title='Price vs Rating vs Value Score')
        st.plotly_chart(fig)
        
        # Top configurations
        st.subheader("Top Configurations")
        st.dataframe(results.head(10).style.background_gradient(subset=['value_score']))
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Full Results (CSV)",
                results.to_csv(index=False),
                "best_value_phones.csv",
                "text/csv"
            )
        with col2:
            report = f"""
            # Mobile Phone Analysis Report
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ## Specifications
            - Minimum Rating: {min_rating}
            - Maximum Price: ₹{max_price:,}
            - Number of Configurations: {len(results)}
            - Top Brand: {results.iloc[0]['brand']}
            - Best Value Score: {results['value_score'].max():.2f}
            """
            st.download_button(
                "Download Analysis Report",
                report,
                "value_analysis_report.md",
                "text/markdown"
            )

elif page == "Market Trends":
    st.header("📈 Market Trends")
    
    df = load_phone_data()
    
    # Price trends
    st.subheader("Price Trends by Segment")
    
    # Create price segments only for non-NaN prices
    df_segments = df[df['price'].notna()].copy()
    price_segments = pd.qcut(df_segments['price'], q=4, labels=['Budget', 'Mid-Range', 'Premium', 'Ultra Premium'])
    segment_stats = df_segments.groupby(price_segments)['ratings'].agg(['mean', 'count'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=segment_stats.index,
        y=segment_stats['count'],
        name='Number of Phones',
        yaxis='y'
    ))
    fig.add_trace(go.Scatter(
        x=segment_stats.index,
        y=segment_stats['mean'],
        name='Average Rating',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Price Segments Analysis',
        yaxis=dict(title='Number of Phones'),
        yaxis2=dict(title='Average Rating', overlaying='y', side='right')
    )
    
    st.plotly_chart(fig)
    
    # Feature trends
    st.subheader("Feature Trends")
    features = ['storage', 'ram']
    feature_labels = {
        'storage': 'Storage (GB)',
        'ram': 'RAM (GB)'
    }
    
    for feature in features:
        # Only plot non-NaN values
        df_feature = df[df[feature].notna()].copy()
        fig = px.box(df_feature, x='brand', y=feature,
                     title=f'{feature_labels[feature]} Distribution by Brand')
        st.plotly_chart(fig)
    
    # Value analysis
    st.subheader("Value Analysis")
    
    # Create scatter plot for phones with valid price and ratings
    df_scatter = df[df['price'].notna() & df['ratings'].notna()].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each brand
    for brand in df_scatter['brand'].unique():
        mask = df_scatter['brand'] == brand
        brand_data = df_scatter[mask]
        
        fig.add_trace(go.Scatter(
            x=brand_data['price'],
            y=brand_data['ratings'],
            name=brand,
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.7
            ),
            hovertemplate=(
                f"Brand: %{text}<br>" +
                "Price: ₹%{x:,.0f}<br>" +
                "Rating: %{y:.2f}<br>" +
                "<extra></extra>"
            ),
            text=[brand] * len(brand_data)  # For hover template
        ))
    
    # Add trend line for all data
    z = np.polyfit(df_scatter['price'], df_scatter['ratings'], 1)
    p = np.poly1d(z)
    x_trend = [df_scatter['price'].min(), df_scatter['price'].max()]
    y_trend = p(x_trend)
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode='lines',
        name='Overall Trend',
        line=dict(
            color='black',
            dash='dash'
        ),
        hovertemplate="Price: ₹%{x:,.0f}<br>Predicted Rating: %{y:.2f}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title='Price vs Rating by Brand',
        xaxis_title='Price (₹)',
        yaxis_title='Rating',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        margin=dict(r=150)  # Add right margin for legend
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add trend analysis text
    slope = z[0]
    correlation = df_scatter['price'].corr(df_scatter['ratings'])
    
    st.markdown(f"""
    **Trend Analysis:**
    - Price-Rating Correlation: {correlation:.3f}
    - Slope: {slope:.6f} (rating points per ₹)
    - Direction: {'Positive' if slope > 0 else 'Negative'} relationship between price and rating
    """)
    
    # Add a trend analysis section
    st.subheader("Price Range Analysis")
    
    # Create price ranges for non-NaN values
    df_ranges = df[df['price'].notna()].copy()
    df_ranges['price_range'] = pd.qcut(df_ranges['price'], 
                                     q=5, 
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Calculate statistics for each price range
    price_range_stats = df_ranges.groupby('price_range').agg({
        'price': ['mean', 'count'],
        'ratings': ['mean', 'std'],
        'ram': 'mean',
        'storage': 'mean'
    }).round(2)
    
    # Format the stats for display
    st.write("Statistics by Price Range:")
    st.dataframe(price_range_stats.style.background_gradient(subset=['mean'], cmap='RdYlGn'))
    
    # Add correlation heatmap
    st.subheader("Feature Correlation Analysis")
    
    # Select numeric columns and ensure they exist
    numeric_cols = ['price', 'ratings', 'ram', 'storage']
    valid_cols = [col for col in numeric_cols if col in df.columns]
    
    # Drop NaN values for correlation analysis
    df_corr = df[valid_cols].dropna()
    correlation = df_corr.corr()
    
    # Create correlation heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation,
        x=correlation.columns,
        y=correlation.columns,
        text=correlation.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        width=700,
        height=700
    )
    
    st.plotly_chart(fig)

elif page == "Performance Analysis":
    st.header("⚡ Performance Analysis")
    
    df = load_phone_data()
    
    # Performance Overview
    st.subheader("Phone Performance Overview")
    
    # Performance score calculation
    df['performance_score'] = (
        df['ram'].fillna(df['ram'].mean()) / df['ram'].mean() * 0.4 +  # 40% weight to RAM
        df['storage'].fillna(df['storage'].mean()) / df['storage'].mean() * 0.3 +  # 30% weight to storage
        df['ratings'].fillna(df['ratings'].mean()) / df['ratings'].max() * 0.3  # 30% weight to ratings
    ) * 100
    
    # Top performing phones
    st.write("Top Performing Phones")
    top_phones = df.nlargest(10, 'performance_score')[
        ['brand', 'name', 'ram', 'storage', 'ratings', 'performance_score']
    ]
    
    st.dataframe(
        top_phones.style.format({
            'ram': '{:.0f} GB',
            'storage': '{:.0f} GB',
            'ratings': '{:.2f}',
            'performance_score': '{:.1f}'
        }).background_gradient(subset=['performance_score'], cmap='RdYlGn')
    )
    
    # Performance distribution
    st.subheader("Performance Score Distribution")
    
    fig = px.histogram(
        df,
        x='performance_score',
        nbins=30,
        title='Distribution of Performance Scores',
        labels={'performance_score': 'Performance Score', 'count': 'Number of Phones'},
        color_discrete_sequence=['#2ecc71']
    )
    
    fig.update_layout(
        xaxis_title="Performance Score",
        yaxis_title="Number of Phones",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # RAM vs Storage Analysis
    st.subheader("RAM vs Storage Analysis")
    
    fig = px.scatter(
        df,
        x='ram',
        y='storage',
        color='ratings',
        size='performance_score',
        hover_data=['brand', 'name', 'price'],
        title='RAM vs Storage Comparison',
        labels={
            'ram': 'RAM (GB)',
            'storage': 'Storage (GB)',
            'ratings': 'Rating'
        },
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Brand Performance Comparison
    st.subheader("Brand Performance Comparison")
    
    brand_perf = df.groupby('brand').agg({
        'performance_score': ['mean', 'max', 'count'],
        'ram': 'mean',
        'storage': 'mean',
        'ratings': 'mean'
    }).round(2)
    
    # Flatten column names
    brand_perf.columns = [
        'Avg Score', 'Max Score', 'Models',
        'Avg RAM', 'Avg Storage', 'Avg Rating'
    ]
    
    # Sort by average score
    brand_perf = brand_perf.sort_values('Avg Score', ascending=False)
    
    st.dataframe(
        brand_perf.style.format({
            'Avg Score': '{:.1f}',
            'Max Score': '{:.1f}',
            'Models': '{:.0f}',
            'Avg RAM': '{:.1f} GB',
            'Avg Storage': '{:.0f} GB',
            'Avg Rating': '{:.2f}'
        }).background_gradient(subset=['Avg Score'], cmap='RdYlGn')
    )
    
    # Performance-Price Ratio
    st.subheader("Performance-Price Ratio")
    
    df['value_ratio'] = df['performance_score'] / df['price']
    
    fig = px.scatter(
        df,
        x='price',
        y='performance_score',
        color='brand',
        size='ratings',
        hover_data=['name', 'ram', 'storage'],
        title='Performance Score vs Price',
        labels={
            'price': 'Price (₹)',
            'performance_score': 'Performance Score',
            'brand': 'Brand'
        }
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Best value phones
    st.write("Best Value Phones (Performance/Price)")
    best_value = df.nlargest(10, 'value_ratio')[
        ['brand', 'name', 'price', 'performance_score', 'value_ratio']
    ]
    
    st.dataframe(
        best_value.style.format({
            'price': '₹{:,.0f}',
            'performance_score': '{:.1f}',
            'value_ratio': '{:.4f}'
        }).background_gradient(subset=['value_ratio'], cmap='RdYlGn')
    )

elif page == "Custom Analysis":
    st.header("🔍 Custom Analysis")
    
    df = load_phone_data()
    
    # Ensure we have clean data
    df_clean = df.dropna(subset=['price', 'ratings']).copy()
    
    # Feature selection
    st.subheader("Select Features")
    
    features = st.multiselect(
        "Choose features to analyze",
        ['price', 'ratings', 'storage', 'ram'],
        ['price', 'ratings']
    )
    
    if len(features) < 2:
        st.warning("Please select at least two features to analyze.")
    else:
        # Visualization type
        viz_type = st.selectbox(
            "Choose visualization type",
            ["Scatter Plot", "Box Plot", "Violin Plot", "Bar Chart"]
        )
        
        if viz_type == "Scatter Plot":
            x_feature = st.selectbox("Select X-axis feature", features)
            y_feature = st.selectbox("Select Y-axis feature", 
                                   [f for f in features if f != x_feature])
            
            # Create hover text
            df_clean['hover_text'] = df_clean.apply(
                lambda row: f"Brand: {row['brand']}<br>" +
                          f"Model: {row['name']}<br>" +
                          f"{x_feature}: {row[x_feature]:,.0f}<br>" +
                          f"{y_feature}: {row[y_feature]:.2f}",
                axis=1
            )
            
            fig = px.scatter(
                df_clean,
                x=x_feature,
                y=y_feature,
                color='brand',
                title=f'{y_feature} vs {x_feature}',
                hover_name='brand',
                hover_data={
                    'hover_text': True,
                    x_feature: False,
                    y_feature: False,
                    'brand': False
                }
            )
            
            fig.update_traces(
                hovertemplate="%{customdata[0]}<extra></extra>"
            )
            
            fig.update_layout(
                height=600,
                xaxis_title=x_feature.title(),
                yaxis_title=y_feature.title(),
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add correlation information
            correlation = df_clean[x_feature].corr(df_clean[y_feature])
            st.markdown(f"**Correlation between {x_feature} and {y_feature}:** {correlation:.2f}")
            
            # Add interpretation
            st.markdown("""
            ##### 📝 Interpretation
            - **Correlation**: Values closer to 1 indicate strong positive correlation
            - **Distribution**: Look for patterns and clusters
            - **Outliers**: Points far from the main cluster
            """)
            
        elif viz_type == "Box Plot":
            feature = st.selectbox("Select feature to analyze", features)
            
            fig = go.Figure()
            
            for brand in df_clean['brand'].unique():
                fig.add_trace(go.Box(
                    y=df_clean[df_clean['brand'] == brand][feature],
                    name=brand,
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title=f'{feature} Distribution by Brand',
                yaxis_title=feature,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Violin Plot":
            feature = st.selectbox("Select feature to analyze", features)
            
            fig = go.Figure()
            
            for brand in df_clean['brand'].unique():
                fig.add_trace(go.Violin(
                    y=df_clean[df_clean['brand'] == brand][feature],
                    name=brand,
                    box_visible=True,
                    meanline_visible=True
                ))
            
            fig.update_layout(
                title=f'{feature} Distribution by Brand',
                yaxis_title=feature,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Bar Chart":
            feature = st.selectbox("Select feature to analyze", features)
            agg_func = st.selectbox(
                "Select aggregation function",
                ["mean", "median", "count", "sum"]
            )
            
            if agg_func == "mean":
                data = df_clean.groupby('brand')[feature].mean()
            elif agg_func == "median":
                data = df_clean.groupby('brand')[feature].median()
            elif agg_func == "count":
                data = df_clean.groupby('brand')[feature].count()
            else:  # sum
                data = df_clean.groupby('brand')[feature].sum()
            
            fig = go.Figure(data=go.Bar(
                x=data.index,
                y=data.values,
                text=data.values.round(2),
                textposition='auto',
            ))
            
            fig.update_layout(
                title=f'{agg_func.title()} {feature} by Brand',
                xaxis_title='Brand',
                yaxis_title=f'{agg_func.title()} {feature}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        st.subheader("Summary Statistics")
        
        stats = df_clean[features].describe()
        st.dataframe(stats.style.format("{:.2f}"))
        
        # Add correlation analysis
        if len(features) > 1:
            st.subheader("Correlation Analysis")
            
            corr = df_clean[features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr,
                x=corr.columns,
                y=corr.columns,
                text=corr.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                colorscale='RdBu'
            ))
            
            fig.update_layout(
                title='Feature Correlation Heatmap',
                width=600,
                height=600
            )
            
            st.plotly_chart(fig)

elif page == "Phone Recommendations":
    st.title("📱 Smart Phone Recommendations")
    
    # Load data
    df = load_phone_data()
    
    # User Preferences Input
    st.markdown("### 🎯 Your Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Range
        st.markdown("#### 💰 Budget")
        min_price = st.number_input("Minimum Price (₹)", 
                                  min_value=0, 
                                  max_value=int(df['price'].max()),
                                  value=10000)
        max_price = st.number_input("Maximum Price (₹)", 
                                  min_value=min_price, 
                                  max_value=int(df['price'].max()),
                                  value=min_price + 10000)
        
        # Brand Preference
        st.markdown("#### 🏢 Brand Preference")
        preferred_brands = st.multiselect(
            "Select preferred brands (optional)",
            options=sorted(df['brand'].unique()),
            default=None
        )
    
    with col2:
        # Features
        st.markdown("#### ⚙️ Important Features")
        min_ram = st.slider("Minimum RAM (GB)", 
                          min_value=1, 
                          max_value=16, 
                          value=4)
        
        min_storage = st.slider("Minimum Storage (GB)", 
                              min_value=16, 
                              max_value=512, 
                              value=64)
        
        min_rating = st.slider("Minimum Rating", 
                             min_value=1.0, 
                             max_value=5.0, 
                             value=4.0,
                             step=0.1)
    
    # Filter phones based on preferences
    filtered_df = df[
        (df['price'] >= min_price) &
        (df['price'] <= max_price) &
        (df['ram'] >= min_ram) &
        (df['storage'] >= min_storage) &
        (df['ratings'] >= min_rating)
    ]
    
    # Apply brand filter if specified
    if preferred_brands:
        filtered_df = filtered_df[filtered_df['brand'].isin(preferred_brands)]
    
    # Calculate value score
    filtered_df['price_norm'] = (filtered_df['price'] - filtered_df['price'].mean()) / filtered_df['price'].std()
    filtered_df['rating_norm'] = (filtered_df['ratings'] - filtered_df['ratings'].mean()) / filtered_df['ratings'].std()
    filtered_df['ram_norm'] = filtered_df['ram'] / filtered_df['ram'].max()
    filtered_df['storage_norm'] = filtered_df['storage'] / filtered_df['storage'].max()
    
    # Calculate weighted score
    filtered_df['value_score'] = (
        -0.3 * filtered_df['price_norm'] +  # Lower price is better
        0.3 * filtered_df['rating_norm'] +
        0.2 * filtered_df['ram_norm'] +
        0.2 * filtered_df['storage_norm']
    )
    
    # Get top recommendations
    recommendations = filtered_df.nlargest(10, 'value_score')
    
    # Display Results
    st.markdown("### 🎁 Recommended Phones")
    
    if len(recommendations) > 0:
        for idx, row in recommendations.iterrows():
            with st.expander(f"{row['brand']} {row['name']} - ₹{row['price']:,.0f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Brand:** {row['brand']}")
                    st.markdown(f"**Price:** ₹{row['price']:,.0f}")
                    st.markdown(f"**Rating:** {row['ratings']}⭐")
                
                with col2:
                    st.markdown(f"**RAM:** {row['ram']} GB")
                    st.markdown(f"**Storage:** {row['storage']} GB")
                    st.markdown(f"**Value Score:** {row['value_score']:.2f}")
        
        # Add Visualization
        st.markdown("### 📊 Visual Comparison")
        
        fig = px.scatter(
            recommendations,
            x='price',
            y='ratings',
            size='ram',
            color='brand',
            hover_name='name',
            title='Recommended Phones Comparison',
            labels={
                'price': 'Price (₹)',
                'ratings': 'Rating',
                'ram': 'RAM (GB)'
            }
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        #### 📝 How to Read the Chart
        - **Size**: Larger bubbles indicate more RAM
        - **Position**: Top-left is better (high rating, lower price)
        - **Color**: Different colors represent different brands
        """)
    else:
        st.warning("No phones found matching your criteria. Try adjusting your preferences.")
    
    # Add Visualization
    if len(recommendations) > 0:
        st.markdown("### 📊 Visual Comparison")
        
        fig = px.scatter(
            recommendations,
            x='price',
            y='ratings',
            size='ram',
            color='brand',
            hover_name='name',
            title='Recommended Phones Comparison',
            labels={
                'price': 'Price (₹)',
                'ratings': 'Rating',
                'ram': 'RAM (GB)'
            }
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        #### 📝 How to Read the Chart
        - **Size**: Larger bubbles indicate more RAM
        - **Position**: Top-left is better (high rating, lower price)
        - **Color**: Different colors represent different brands
        """)

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
            'price': 'Price (₹)',
            'ratings': 'Rating',
            'market_position': 'Market Position'
        },
        hover_data={
            'name': True,
            'ram': True,
            'storage': True,
            'bubble_size': False,
            'market_position': ':.2f'
        }
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Price (₹)",
        yaxis_title="Rating",
        hovermode='closest'
    )
    
    fig.update_traces(
        hovertemplate="<br>".join([
            "Brand: %{customdata[0]}",
            "Model: %{customdata[1]}",
            "Price: ₹%{x:,.0f}",
            "Rating: %{y:.2f}⭐",
            "RAM: %{customdata[2]} GB",
            "Storage: %{customdata[3]} GB",
            "Market Position: %{customdata[4]:.2f}"
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
            'price': '₹{:,.0f}',
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
                   f"Average Price: ₹{row['Avg Price']:,.0f}<br>" +
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
            'Avg Price': 'Average Price (₹)',
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
        xaxis_title="Average Price (₹)",
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
            'Avg Price': '₹{:,.0f}',
            'Price Std': '₹{:,.0f}',
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
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and Machine Learning</p>
        <p> 2025 Mobile Phone Analyzer Pro</p>
    </div>
""", unsafe_allow_html=True)
