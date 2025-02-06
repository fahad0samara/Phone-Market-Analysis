from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from database_setup import MobilePhone, Base

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine('sqlite:///database/mobile_phones.db')
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_all_phones(self):
        """Get all mobile phones"""
        return self.session.query(MobilePhone).all()

    def get_phones_by_brand(self, brand):
        """Get phones by brand name"""
        return self.session.query(MobilePhone).filter(MobilePhone.brand == brand.upper()).all()

    def get_phones_by_price_range(self, min_price, max_price):
        """Get phones within a price range"""
        return self.session.query(MobilePhone).filter(
            MobilePhone.price.between(min_price, max_price)
        ).all()

    def get_phones_by_rating(self, min_rating):
        """Get phones with rating >= min_rating"""
        return self.session.query(MobilePhone).filter(
            MobilePhone.ratings >= min_rating
        ).all()

    def get_phones_by_segment(self, segment):
        """Get phones by price segment"""
        return self.session.query(MobilePhone).filter(
            MobilePhone.price_segment == segment
        ).all()

    def get_brand_statistics(self):
        """Get statistics for each brand"""
        return self.session.query(
            MobilePhone.brand,
            func.count(MobilePhone.id).label('count'),
            func.avg(MobilePhone.price).label('avg_price'),
            func.avg(MobilePhone.ratings).label('avg_rating')
        ).group_by(MobilePhone.brand).all()

    def get_top_rated_phones(self, limit=10):
        """Get top rated phones"""
        return self.session.query(MobilePhone).order_by(
            MobilePhone.ratings.desc()
        ).limit(limit).all()

    def get_best_value_phones(self, limit=10):
        """Get phones with best rating/price ratio"""
        return self.session.query(MobilePhone).order_by(
            (MobilePhone.ratings / MobilePhone.price).desc()
        ).limit(limit).all()

    def search_phones(self, query):
        """Search phones by name"""
        search = f"%{query}%"
        return self.session.query(MobilePhone).filter(
            MobilePhone.name.like(search)
        ).all()

    def get_segment_statistics(self):
        """Get statistics for each price segment"""
        return self.session.query(
            MobilePhone.price_segment,
            func.count(MobilePhone.id).label('count'),
            func.avg(MobilePhone.price).label('avg_price'),
            func.avg(MobilePhone.ratings).label('avg_rating')
        ).group_by(MobilePhone.price_segment).all()

    def close(self):
        """Close the database session"""
        self.session.close()

# Example usage
if __name__ == "__main__":
    db = DatabaseManager()
    
    print("\nTop 5 Rated Phones:")
    top_phones = db.get_top_rated_phones(5)
    for phone in top_phones:
        print(f"{phone.name}: {phone.ratings} stars, ₹{phone.price}")
    
    print("\nBrand Statistics:")
    brand_stats = db.get_brand_statistics()
    for brand, count, avg_price, avg_rating in brand_stats:
        print(f"{brand}: {count} models, Avg Price: ₹{avg_price:.2f}, Avg Rating: {avg_rating:.2f}")
    
    db.close()
