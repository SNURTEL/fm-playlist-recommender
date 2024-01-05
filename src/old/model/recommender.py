import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank


class Recommender:
    def __init__(self,
                 users,
                 items,
                 user_features=None,
                 item_features=None,
                 item_feature_names=None):

        if (item_features is None and item_feature_names is not None) or (item_features is not None and item_feature_names is None):
            raise ValueError("Either both or none of item_features and item_feature_names need to be provided")

        self._in_users = users
        self._in_items = items
        self._in_user_features = user_features
        self._in_item_features = item_features
        self._in_item_feature_names = item_feature_names

        self._dataset = Dataset()
        self._dataset.fit(
            users=users,
            items=items
        )

        self._dataset.fit_partial(
            items=items,
            item_features=item_features
        )

        self._interactions = None
        self._item_features = None

        self._model = None

    def fit(self,
            interactions,
            loss,
            learning_rate,
            no_components,
            epochs,
            num_threads=1,
            items_alpha=0,
            users_alpha=0):
        if self._item_features is None:
            self._build_item_features(self._in_item_feature_names)

        if self._interactions is None:
            raise ValueError("Call .set_interactions first")

        self._model = LightFM(
            loss=loss,
            learning_rate=learning_rate,
            item_alpha=items_alpha,
            user_alpha=users_alpha,
            no_components=no_components,
        )

        self._model.fit(
            interactions=self._interactions,
            item_features=self._item_features,
            epochs=epochs,
            num_threads=num_threads,
            verbose=True)

    def set_interactions(self, interactions):
        self._interactions = interactions

    def _build_item_features(self, feature_names):
        if self._in_items is None:
            raise ValueError("Items not set")

        self._item_features = self._dataset.build_item_features(
            ((i, feature_names) for i in self._in_items),
            normalize=False)

    def recommend_single(self, user: int, number: int, num_threads=1):
        indices = self._predict_single_indices(user, number, num_threads=num_threads)
        return self._in_items.loc[indices]

    def _predict_single_indices(self, user: int, number: int, num_threads=1):
        predicted_scores = self._model.predict(user, np.arange(self._interactions.shape[1]),
                                               item_features=self._item_features,
                                               num_threads=num_threads)
        return np.argpartition(predicted_scores, -number)[-number:]

    def recommend_multiple(self, users, number, num_threads=1):
        indices = self._predict_multiple_indices(users, number, num_threads)

        return self._in_items.loc[indices]

    def _predict_multiple_indices(self, users, number, num_threads=1):
        if len(users) == 1:
            return frozenset(self._predict_single_indices(users[0], number, num_threads=num_threads))

        predictions = [frozenset(self._predict_single_indices(u, number, num_threads=num_threads)) for u in users]
        intersection = predictions[0].intersection(*predictions[1:])

        if len(intersection) == 0:
            print("No matching items. Note that the model does not yet support cold start for users - please train it "
                  "on the entire dataset.")

        return intersection

    def measure_clustering(self, indices):
        coords = np.take(self._model.get_item_representations()[1], list(indices), axis=0)
        center = np.sum(coords, axis=0) / coords.shape[0]
        return np.average(np.apply_along_axis(lambda x: np.linalg.norm(center - x, ord=2), 1, coords))

    def evaluate_submodel(self, test_set, train_set=None):
        train_auc_h = auc_score(self._model, train_set,
                                item_features=self._item_features,
                                num_threads=12).mean()
        test_auc_h = auc_score(self._model, test_set,
                               train_interactions=train_set,
                               item_features=self._item_features,
                               num_threads=12).mean()

        train_precision_h = precision_at_k(self._model, train_set, k=10,
                                           item_features=self._item_features,
                                           num_threads=12).mean()
        test_precision_h = precision_at_k(self._model, test_set, k=10,
                                          train_interactions=train_set,
                                          item_features=self._item_features,
                                          num_threads=12).mean()

        train_recall_h = recall_at_k(self._model, train_set, k=10,
                                     item_features=self._item_features,
                                     num_threads=12).mean()
        test_recall_h = recall_at_k(self._model, test_set, k=10,
                                    train_interactions=train_set,
                                    item_features=self._item_features,
                                    num_threads=12).mean()

        train_reciprocal_rank_h = reciprocal_rank(self._model, train_set,
                                                  item_features=self._item_features,
                                                  num_threads=12).mean()
        test_reciprocal_rank_h = reciprocal_rank(self._model, test_set,
                                                 train_interactions=train_set,
                                                 item_features=self._item_features,
                                                 num_threads=12).mean()

        print('AUC: train %.6f, test %.6f.' % (train_auc_h, test_auc_h))
        print('Precision: train %.6f, test %.6f.' % (train_precision_h, test_precision_h))
        print('Recall: train %.6f, test %.6f.' % (train_recall_h, test_recall_h))
        print('Reciprocal rank: train %.6f, test %.6f.' % (train_reciprocal_rank_h, test_reciprocal_rank_h))


