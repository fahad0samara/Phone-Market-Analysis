import pandas as pd
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MobilePhoneAnalyzer:
    def __init__(self):
        # Create necessary directories
        os.makedirs('database', exist_ok=True)
        os.makedirs('exports', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Initialize database
        self.Base = declarative_base()
        self.engine = create_engine('sqlite:///database/mobile_phones.db')
        self.Session = sessionmaker(bind=self.engine)
        
    def setup_database(self):
        """Set up the database schema and import data"""
        class MobilePhone(self.Base):
            __tablename__ = 'mobile_phones'
            
            id = Column(Integer, primary_key=True)
            name = Column(String(255), nullable=False)
            brand = Column(String(50), index=True)
            price = Column(Float, index=True)
            ratings = Column(Float, index=True)
            storage = Column(Float)
            ram = Column(Float)
            price_segment = Column(String(50), index=True)
            corpus = Column(Text)
            img_url = Column(String(500))
        
        # Create tables
        self.Base.metadata.create_all(self.engine)
        return MobilePhone
    
    def clean_and_import_data(self, file_path):
        """Clean and import data from CSV"""
        print("Reading and cleaning data...")
        df = pd.read_csv(file_path)
        
        # Clean data
        df = df.dropna(subset=['name', 'price', 'ratings'])
        
        # Extract brand from name
        df['brand'] = df['name'].str.split().str[0].str.upper()
        
        # Extract storage and RAM
        def extract_storage(text):
            import re
            match = re.search(r'(\d+)\s*GB', str(text))
            return float(match.group(1)) if match else None
            
        def extract_ram(text):
            import re
            match = re.search(r'RAM(\d+)', str(text))
            return float(match.group(1)) if match else None
        
        df['storage'] = df['name'].apply(extract_storage)
        df['ram'] = df['corpus'].apply(extract_ram)
        
        # Create price segments
        def get_price_segment(price):
            if price < 10000: return 'Budget'
            elif price < 20000: return 'Mid-Range'
            elif price < 30000: return 'High-End'
            else: return 'Premium'
            
        df['price_segment'] = df['price'].apply(get_price_segment)
        
        # Save cleaned data
        df.to_csv('mobile_dataset_cleaned.csv', index=False)
        return df
    
    def import_to_database(self, df, MobilePhone):
        """Import cleaned data to database"""
        print("Importing data to database...")
        session = self.Session()
        
        # Clear existing data
        session.query(MobilePhone).delete()
        
        # Import data
        mobile_phones = []
        for _, row in df.iterrows():
            phone = MobilePhone(
                name=row['name'],
                brand=row['brand'],
                price=row['price'],
                ratings=row['ratings'],
                storage=row['storage'],
                ram=row['ram'],
                price_segment=row['price_segment'],
                corpus=row['corpus'],
                img_url=row['imgURL']
            )
            mobile_phones.append(phone)
        
        session.bulk_save_objects(mobile_phones)
        session.commit()
        session.close()
    
    def analyze_data(self):
        """Perform data analysis and generate reports"""
        print("Analyzing data...")
        queries = {
            'basic_stats': """
                SELECT 
                    COUNT(*) as total_phones,
                    AVG(price) as avg_price,
                    AVG(ratings) as avg_rating,
                    AVG(storage) as avg_storage,
                    AVG(ram) as avg_ram
                FROM mobile_phones
            """,
            'brand_analysis': """
                SELECT brand,
                       COUNT(*) as model_count,
                       AVG(price) as avg_price,
                       AVG(ratings) as avg_rating
                FROM mobile_phones
                GROUP BY brand
                HAVING model_count > 10
                ORDER BY model_count DESC
            """,
            'segment_analysis': """
                SELECT price_segment,
                       COUNT(*) as phone_count,
                       AVG(price) as avg_price,
                       AVG(ratings) as avg_rating,
                       AVG(storage) as avg_storage,
                       AVG(ram) as avg_ram
                FROM mobile_phones
                GROUP BY price_segment
                ORDER BY avg_price DESC
            """,
            'top_phones': """
                SELECT name, brand, price, ratings, storage, ram
                FROM mobile_phones
                ORDER BY ratings DESC
                LIMIT 20
            """,
            'best_value': """
                SELECT name, brand, price, ratings, storage, ram,
                       (ratings / price * 10000) as value_score
                FROM mobile_phones
                ORDER BY value_score DESC
                LIMIT 20
            """
        }
        
        results = {}
        for name, query in queries.items():
            results[name] = pd.read_sql(query, self.engine)
            
        # Export results
        for name, df in results.items():
            df.to_csv(f'exports/{name}.csv', index=False)
            
        return results
    
    def generate_report(self, results):
        """Generate summary report"""
        print("Generating report...")
        with open('exports/analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("Mobile Phone Market Analysis Summary\n")
            f.write("===================================\n\n")
            
            # Basic stats
            stats = results['basic_stats'].iloc[0]
            f.write("1. Overall Statistics:\n")
            f.write(f"Total Phones: {stats['total_phones']}\n")
            f.write(f"Average Price: ₹{stats['avg_price']:.2f}\n")
            f.write(f"Average Rating: {stats['avg_rating']:.2f}\n")
            f.write(f"Average Storage: {stats['avg_storage']:.1f} GB\n")
            f.write(f"Average RAM: {stats['avg_ram']:.1f} GB\n\n")
            
            # Brand analysis
            f.write("2. Top Brands:\n")
            for _, row in results['brand_analysis'].head().iterrows():
                f.write(f"{row['brand']}: {row['model_count']} models\n")
                f.write(f"  Average Price: ₹{row['avg_price']:.2f}\n")
                f.write(f"  Average Rating: {row['avg_rating']:.2f}\n\n")
            
            # Segment analysis
            f.write("3. Price Segments:\n")
            for _, row in results['segment_analysis'].iterrows():
                f.write(f"{row['price_segment']}: {row['phone_count']} phones\n")
                f.write(f"  Average Price: ₹{row['avg_price']:.2f}\n")
                f.write(f"  Average Rating: {row['avg_rating']:.2f}\n\n")
            
            # Top phones
            f.write("4. Top Rated Phones:\n")
            for _, row in results['top_phones'].head().iterrows():
                f.write(f"{row['name']}: {row['ratings']} stars, ₹{row['price']}\n")
            
            f.write("\n5. Best Value Phones:\n")
            for _, row in results['best_value'].head().iterrows():
                f.write(f"{row['name']}: Rating: {row['ratings']}, ")
                f.write(f"Price: ₹{row['price']}, ")
                f.write(f"Value Score: {row['value_score']:.2f}\n")

def main():
    # Initialize analyzer
    analyzer = MobilePhoneAnalyzer()
    
    # Setup database
    MobilePhone = analyzer.setup_database()
    
    # Clean and import data
    df = analyzer.clean_and_import_data('mobile_recommendation_system_dataset.csv')
    analyzer.import_to_database(df, MobilePhone)
    
    # Analyze data
    results = analyzer.analyze_data()
    
    # Generate report
    analyzer.generate_report(results)
    
    print("\nAnalysis complete! Check the following directories:")
    print("1. exports/ - Contains CSV files and analysis summary")
    print("2. database/ - Contains the SQLite database")
    print("3. mobile_dataset_cleaned.csv - Cleaned dataset")

if __name__ == "__main__":
    main()
