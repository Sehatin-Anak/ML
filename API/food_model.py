import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from typing import Dict, Text

class FoodModel(tfrs.models.Model):

    def __init__(self, rating_weight, retrieval_weight, unique_user_ids, unique_food_titles, Recipes):

        super().__init__()

        embedding_dimension = 64

        self.food_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_food_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_food_titles) + 1, embedding_dimension)
        ])
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=Recipes.batch(128).map(self.food_model)
            )
        )
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_model(features["AuthorId"])
        food_embeddings = self.food_model(features["Name"])

        return (
            user_embeddings,
            food_embeddings,

            self.rating_model(
                tf.concat([user_embeddings, food_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = features.pop("Rating")

        user_embeddings, food_embeddings, rating_predictions = self(features)

        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, food_embeddings)
        
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)
