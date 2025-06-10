from fastapi import Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from models.productmodels import ProductModel
from models.reviewmodels import ReviewModel
from models.usermodels import User
from config.database import get_db
from config.token import get_currentUser
from dto.reviewschema import ReviewCreate


class ReviewService:
    def get_all(db: Session):
        return db.query(ReviewModel).all()

    def create_review(
        request: ReviewCreate,
        productId: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_currentUser),
    ):
        try:
            # Fetch the product to review
            product = db.query(ProductModel).filter(ProductModel.id == productId).first()
            if not product:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Produk tidak ditemukan.",
                )

            # Check if user has already reviewed this product
            existing_review = db.query(ReviewModel).filter(
                ReviewModel.user_id == current_user.id,
                ReviewModel.product_id == productId
            ).first()
            if existing_review:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Anda sudah memberikan ulasan untuk produk ini.",
                )

            # Create new review
            review_new = ReviewModel(
                name=current_user.name,
                user_id=current_user.id,
                rating=request.rating,
                comment=request.comment,
                product_id=productId
            )

            db.add(review_new)
            db.commit()

            # Recalculate the average rating for the product
            total_rating, total_reviews = db.query(
                func.sum(ReviewModel.rating), func.count(ReviewModel.id)
            ).filter(
                ReviewModel.product_id == productId
            ).first()

            if total_reviews == 0 or total_rating is None:
                product.rating = 0
            else:
                product.rating = int(total_rating / total_reviews)

            db.commit()

            return review_new  # Optionally return the newly created review

        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Terjadi kesalahan saat menambahkan ulasan.",
            )
