from fastapi import Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm.session import Session
from config.database import get_db
# from models.productmodels import ProductModel, ReviewModel
from models.productmodels import ProductModel, ReviewModel
from models.usermodels import User  # ✅ Add this line
from dto.productschema import ProductSchema
from config.hashing import Hashing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


class ProductService:
    @staticmethod
    def get_all_product(db: Session):
        return db.query(ProductModel).order_by(ProductModel.rating.desc()).limit(10).all()

    @staticmethod
    def recommend_products(db: Session) -> dict:
        products = ProductService.get_all_product(db)

        if not products:
            return {"recommended_products": [], "accuracy": 0.0}

        df = pd.DataFrame([{
            "id": product.id,
            "name": product.name,
            "description": product.description,
            "image": product.image,
            "countInStock": product.countInStock,
            "price": product.price,
            "rating": float(product.rating),
        } for product in products])

        # Normalize rating and price
        scaler = MinMaxScaler()
        features = scaler.fit_transform(df[['rating', 'price']])

        # Nearest Neighbors with n_neighbors=10 to match requested neighbors
        nn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute', n_jobs=-1)
        nn_model.fit(features)

        # Find 10 nearest neighbors for the first product (excluding itself)
        similar_indices = nn_model.kneighbors(features[[0]], n_neighbors=10, return_distance=False)[0]
        similar_indices = similar_indices[1:]  # Exclude the product itself

        recommended_products = df.iloc[similar_indices].to_dict(orient='records')

        # Sort recommendations by rating descending, then price ascending
        recommended_products.sort(key=lambda x: (-x['rating'], x['price']))
        top_recommended_products = recommended_products[:10]

        actual_top_rated_products = df.nlargest(10, 'rating')
        matching_products = [p for p in top_recommended_products if p['id'] in actual_top_rated_products['id'].tolist()]
        accuracy = len(matching_products) / 10.0 if top_recommended_products else 0.0

        result = {
            "recommended_products": top_recommended_products,
            "accuracy": accuracy
        }
        return result

    @staticmethod
    def create_product(request: ProductSchema, db: Session):
        new_product = ProductModel(
            name=request.name,
            image=request.image,
            category=request.category,
            description=request.description,
            price=request.price,
            countInStock=request.countInStock,
            rating=request.rating,
        )

        db.add(new_product)
        db.commit()
        db.refresh(new_product)

        return new_product

    @staticmethod
    def show_product(productid: int, db: Session) -> dict:
        show_p = db.query(ProductModel).filter(ProductModel.id == productid).first()
        if not show_p:
            raise HTTPException(status_code=404, detail="Product not found")

        review_id = db.query(ReviewModel).filter(ReviewModel.product_id == show_p.id).all()

        reviews_with_sentiment = []
        for review in review_id:
            sentiment_scores = sia.polarity_scores(review.comment)
            sentiment_score = sentiment_scores['compound']

            if sentiment_score >= 0.05:
                sentiment_label = 'POSITIVE'
            elif sentiment_score <= -0.05:
                sentiment_label = 'NEGATIVE'
            else:
                sentiment_label = 'NEUTRAL'

            review_info = {
                "id": review.id,
                "rating": review.rating,
                "comment": review.comment,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score
            }
            reviews_with_sentiment.append(review_info)

        response = {
            "id": show_p.id,
            "category": show_p.category,
            "price": show_p.price,
            "rating": show_p.rating,
            "image": show_p.image,
            "name": show_p.name,
            "description": show_p.description,
            "countInStock": show_p.countInStock,
            "reviews": reviews_with_sentiment,
        }

        return response

    @staticmethod
    def update_product(productid: int, request: ProductSchema, db: Session):
        product = db.query(ProductModel).filter(ProductModel.id == productid).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        product.name = request.name
        product.image = request.image
        product.category = request.category
        product.description = request.description
        product.price = request.price
        product.countInStock = request.countInStock
        product.rating = request.rating
        db.commit()
        db.refresh(product)

        return product

    @staticmethod
    def delete_product(productid: int, db: Session):
        del_product = db.query(ProductModel).filter(ProductModel.id == productid).first()
        if not del_product:
            raise HTTPException(status_code=404, detail="Product not found")

        db.delete(del_product)
        db.commit()

        return {"message": "Product deleted successfully"}
    @staticmethod
    def recommend_products_personalized(db: Session, current_user: User) -> dict:
        # Step 1: Get user's past review history
        reviews = db.query(ReviewModel).filter(ReviewModel.user_id == current_user.id).all()
        if not reviews:
            # No history, fallback to global recommendations
            return ProductService.recommend_products_global(db)

        # Step 2: Create personalized vector from rated products
        product_ids = [r.product_id for r in reviews]
        rated_products = db.query(ProductModel).filter(ProductModel.id.in_(product_ids)).all()
        df = pd.DataFrame([{
            "rating": float(p.rating),
            "price": float(p.price)
        } for p in rated_products])
        if df.empty:
            return ProductService.recommend_products_global(db)

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        user_pref_vector = scaler.fit_transform(df).mean(axis=0).reshape(1, -1)

        # Step 3: Compare with all products
        all_products = db.query(ProductModel).all()
        df_all = pd.DataFrame([{
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "image": p.image,
            "countInStock": p.countInStock,
            "price": float(p.price),
            "rating": float(p.rating),
        } for p in all_products])

        if df_all.empty:
            return {"recommended_products": [], "accuracy": 0.0}

        features_all = scaler.fit_transform(df_all[['rating', 'price']])
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(user_pref_vector, features_all).flatten()

        df_all['similarity'] = similarities
        recommended = df_all.sort_values(by='similarity', ascending=False).head(10)
        return {
            "recommended_products": recommended.drop(columns='similarity').to_dict(orient="records"),
            "personalized": True
        }
