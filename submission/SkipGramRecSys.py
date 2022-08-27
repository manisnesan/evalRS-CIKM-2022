import numpy as np
import pandas as pd
from reclist.abstractions import RecModel

from gensim.models import Word2Vec
import random
from tqdm import tqdm

class SkipGramRecSys(RecModel):
    def __init__(self, items: pd.DataFrame, top_k: int = 100, **kwargs):
        super(SkipGramRecSys, self).__init__()
        self.items = items
        self.top_k = top_k
        self.mappings = None  # user -> sampled tracked
        self.word2vec_model = None
        return

    def train(self, train_df: pd.DataFrame):
        print(train_df.head(1))

        # tracks are built in order to build sequences
        df = train_df[["user_id", "track_id", "timestamp"]].sort_values("timestamp")

        # group by user to create sequence of tracks
        user_tracks = df.groupby("user_id", sort=False)["track_id"].agg(list)

        # build sequences of tracks
        sentences = user_tracks.values.tolist()
    
        #From
        # train word2vec
        self.word2vec_model = Word2Vec(
            sentences, min_count=3, vector_size=100, window=40, epochs=5, sg=1
        )

        user_tracks_df = pd.DataFrame(user_tracks)

        # we sample 40 songs for each user. This will be used at runtime to build
        # a user vector
        user_tracks_df["track_id_sampled"] = user_tracks_df["track_id"].apply(
            lambda x: random.choices(x, k=40)
        )

        # this dictionary maps users to the songs:
        # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}
        self.mappings = user_tracks_df.T.to_dict()

        print("Training completed!")
        return
    
    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        """

        This function takes as input all the users that we want to predict the top-k items for, and
        returns all the predicted songs.

        While in this example is just a random generator, the same logic in your implementation
        would allow for batch predictions of all the target data points.

        """
        k = self.top_k
        num_users = len(user_ids)
        user_ids = user_ids.copy()

        pbar = tqdm(total=len(user_ids), position=0)

        predictions = []

        for user in user_ids["user_id"]:
            # for each user we get the sample tracks
            user_tracks = self.mappings[user]["track_id_sampled"]

            # average to get user embedding
            get_user_embedding = np.mean(
                [self.word2vec_model.wv[t] for t in user_tracks], axis=0
            )

            max_items_to_return = len(self.mappings[user].get("track_id")) + self.top_k
            # track predictions for the user
            user_predictions = [
                k[0]
                for k in self.word2vec_model.wv.most_similar(
                    positive=[get_user_embedding], topn=max_items_to_return
                )
            ]

            # filter songs that user already listened
            user_predictions = list(
                filter(
                    lambda x: x not in self.mappings[user].get("track_id"),
                    user_predictions,
                )
            )[0:self.top_k]

            predictions.append(user_predictions)
            pbar.update(1)
        pbar.close()

        users = user_ids["user_id"].values.reshape(-1, 1)
        pred = np.concatenate([users, np.array(predictions)], axis=1)
        return pd.DataFrame(
            pred, columns=["user_id", *[str(i) for i in range(k)]]
        ).set_index("user_id")
