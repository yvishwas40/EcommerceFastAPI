from sqlalchemy import Column, DateTime, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base
from datetime import datetime


class ReviewModel(Base):
    __tablename__ = "review"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    comment = Column(String(255), nullable=False)
    rating = Column(Integer, nullable=False)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="reviews")

    product_id = Column(Integer, ForeignKey("product.id"), nullable=False)
    product = relationship("ProductModel", back_populates="reviews_user")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Review by {self.name} - Rating: {self.rating}>"
