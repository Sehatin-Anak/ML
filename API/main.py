import os
import uvicorn
import traceback
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import re
import ast
import gdown

from zipfile import ZipFile
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from food_model import FoodModel
from typing import List

folder_path = os.path.join(os.getcwd(), "data")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Download Data
url = 'https://drive.google.com/uc?id=1uUlLxPPFTKUVDvMHUYAuvezVmd0UFvKq'
output = './data/models.zip'
gdown.download(url=url, output=output)

extract_folder = './data'
with ZipFile(output, 'r') as modelFolder:
    modelFolder.extractall(extract_folder)


# Initialize Model
df_recipes = pd.read_csv('data/recipes_reduce.csv')
df_reviews = pd.read_csv('data/reviews_reduce.csv')
df_reviews = df_reviews.astype({'AuthorId': 'string'})

Reviews = tf.data.Dataset.from_tensor_slices(
    dict(df_reviews[['AuthorId', 'Name', 'Rating']]))
Recipes = tf.data.Dataset.from_tensor_slices(dict(df_recipes[['Name']]))

Reviews = Reviews.map(lambda x: {
    "Name": x["Name"],
    "AuthorId": x["AuthorId"],
    "Rating": float(x["Rating"])
})

Recipes = Recipes.map(lambda x: x["Name"])

food_titles = Recipes.batch(1_000)
user_ids = Reviews.batch(1_000).map(lambda x: x["AuthorId"])

unique_food_titles = np.unique(np.concatenate(list(food_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

model = FoodModel(rating_weight=1.0,
                  retrieval_weight=1.0,
                  unique_user_ids=unique_user_ids,
                  unique_food_titles=unique_food_titles,
                  Recipes=Recipes)

dummy_input = {
    "AuthorId": [str(unique_user_ids[0])],
    "Name": [str(unique_food_titles[0])],
    "Rating": [0.0]
}

for key in dummy_input:
    dummy_input[key] = tf.constant(
        dummy_input[key], dtype=tf.string if key != "Rating" else tf.float32)

model(dummy_input)

model.load_weights('data/food_recommendation_model.h5')


def recommend_food_for_random_user(model, top_n=100):
    random_user_id = df_reviews['AuthorId'].sample(1).values[0]
    k = top_n

    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.zip(
            (Recipes.batch(100), Recipes.batch(100).map(model.food_model)))
    )

    _, titles = index(tf.constant([str(random_user_id)]), k=k)

    recommended_titles = [title.decode("utf-8")for title in titles[0].numpy()]
    return recommended_titles, random_user_id


# Function to get image URL for a given recipe name


def get_image_url(recipe_df, recipe_name, image_column='Images'):
    image_urls = recipe_df[recipe_df['Name']
                           == recipe_name][image_column].values
    if len(image_urls) > 0:
        match = re.search(r'https://.*?\.(jpg|JPG|jpeg)', image_urls[0])
        if match:
            return match.group()
    return None


app = FastAPI()


class Recommendation(BaseModel):
    Name: str
    Images: str
    Ingredient: List[str]
    AggregatedRating: float
    ReviewCount: float
    Category: str
    Description: str
    Calories: float
    Fat: float
    SaturatedFat: float
    Cholesterol: float
    Sodium: float
    Carbohydrate: float
    Fiber: float
    Sugar: float
    Protein: float
    Instructions: List[str]


# Route to generate and save JSON file


@app.get("/generate_json/{age_group}")
def generate_json(age_group: int):
    try:
        # Get recommendation results
        recommendations, random_user_id = recommend_food_for_random_user(
            model, top_n=100)

        # Create a Recommendation object from the recommendation results
        recommendation_objects = []
        for title in recommendations:
            ingredient = df_recipes[df_recipes['Name']
                                    == title]['Ingredients'].values[0]
            aggregated_rating = df_recipes[df_recipes['Name']
                                           == title]['AggregatedRating'].values[0]
            review_count = df_recipes[df_recipes['Name']
                                      == title]['ReviewCount'].values[0]
            calories = df_recipes[df_recipes['Name']
                                  == title]['Calories'].values[0]
            fat = df_recipes[df_recipes['Name'] == title]['Fat'].values[0]
            saturated_fat = df_recipes[df_recipes['Name']
                                       == title]['SaturatedFat'].values[0]
            cholesterol = df_recipes[df_recipes['Name']
                                     == title]['Cholesterol'].values[0]
            sodium = df_recipes[df_recipes['Name']
                                == title]['Sodium'].values[0]
            carbohydrate = df_recipes[df_recipes['Name']
                                      == title]['Carbohydrate'].values[0]
            fiber = df_recipes[df_recipes['Name']
                               == title]['Fiber'].values[0]
            sugar = df_recipes[df_recipes['Name'] == title]['Sugar'].values[0]
            protein = df_recipes[df_recipes['Name']
                                 == title]['Protein'].values[0]
            instructions = df_recipes[df_recipes['Name']
                                      == title]['Instructions'].values[0]
            description = df_recipes[df_recipes['Name']
                                     == title]['Description'].values[0]
            category = df_recipes[df_recipes['Name']
                                  == title]['RecipeCategory'].values[0]

            ingredient = ast.literal_eval(ingredient)
            instructions = ast.literal_eval(instructions)

            recom_object = Recommendation(
                Name=title,
                Images=get_image_url(df_recipes, title),
                Ingredient=ingredient,
                AggregatedRating=float(aggregated_rating),
                ReviewCount=float(review_count),
                Description=description,
                Category=category,
                Calories=float(calories),
                Fat=float(fat),
                SaturatedFat=float(saturated_fat),
                Cholesterol=float(cholesterol),
                Sodium=float(sodium),
                Carbohydrate=float(carbohydrate),
                Fiber=float(fiber),
                Sugar=float(sugar),
                Protein=float(protein),
                Instructions=instructions
            )

            # age group 1 to 3 years
            if age_group == 1:
                if (recom_object.Calories <= 450 and
                        recom_object.Protein <= 7 and
                        recom_object.Carbohydrate <= 72):
                    recommendation_objects.append(recom_object)
            # age group 4 to 6 years
            elif age_group == 2:
                if (recom_object.Calories <= 467 and
                        recom_object.Protein <= 9 and
                        recom_object.Carbohydrate <= 74):
                    recommendation_objects.append(recom_object)
            # age group 7 to 9 years
            elif age_group == 3:
                if (recom_object.Calories <= 550 and
                        recom_object.Protein <= 14 and
                        recom_object.Carbohydrate <= 84):
                    recommendation_objects.append(recom_object)
            else:
                print("not available in age groups")

        # Converts Recommendation objects to JSON form
        json_encoder = jsonable_encoder({
            "random_user_id": random_user_id,
            "recommendations": recommendation_objects,
        })

        return JSONResponse(content=json_encoder)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)


# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
