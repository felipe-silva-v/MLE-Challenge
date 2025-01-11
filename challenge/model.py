import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from typing import Tuple, Union, List

class DelayModel:
    """
    A model for predicting flight delays using LightGBM.
    """

    def __init__(self):
        """
        Initialize the DelayModel class with an untrained LightGBM model.
        """
        self._model = None

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): The raw input data.
            target_column (str, optional): The name of the target column.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]: Processed features, optionally with target.
        """
        # Feature engineering
        data['high_season'] = data['Fecha-I'].apply(
            lambda x: 1 if '12-15' <= x[-5:] <= '03-03' or '07-15' <= x[-5:] <= '07-31' or '09-11' <= x[-5:] <= '09-30' else 0
        )
        data['min_diff'] = (
            pd.to_datetime(data['Fecha-O']) - pd.to_datetime(data['Fecha-I'])
        ).dt.total_seconds() / 60
        data['period_day'] = pd.to_datetime(data['Fecha-I']).dt.hour.apply(
            lambda x: 'morning' if 5 <= x <= 11 else 'afternoon' if 12 <= x <= 18 else 'night'
        )
        data['delay'] = data['min_diff'].apply(lambda x: 1 if x > 15 else 0)

        # Encoding categorical variables
        data = pd.get_dummies(data, columns=['OPERA', 'TIPOVUELO', 'period_day'], drop_first=True)

        # Select important features
        important_features = [
            'OPERA_Latin American Wings', 'MES_7', 'MES_10', 'OPERA_Grupo LATAM',
            'MES_12', 'TIPOVUELO_I', 'MES_4', 'MES_11', 'OPERA_Sky Airline', 'OPERA_Copa Air'
        ]

        # Ensure all important features exist
        for feature in important_features:
            if feature not in data.columns:
                data[feature] = 0

        if target_column:
            target = data[[target_column]]
            features = data[important_features]
            return features, target

        return data[important_features]

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Train the LightGBM model using the given features and target.

        Args:
            features (pd.DataFrame): Processed features for training.
            target (pd.DataFrame): Target variable for training.
        """
        # Validate and adjust scale_pos_weight
        class_0 = len(target[target['delay'] == 0])
        class_1 = len(target[target['delay'] == 1])
        scale_pos_weight = (class_0 / class_1 * 1.5) if class_1 > 0 else 1.0

        self._model = LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=30,
            max_depth=6,
            learning_rate=0.03,
            n_estimators=300,
            scale_pos_weight=scale_pos_weight,
            min_child_samples=10,
            colsample_bytree=0.7,
            subsample=0.7,
            objective='binary',
            random_state=42
        )
        self._model.fit(features, target.values.ravel())

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict flight delays using the trained LightGBM model.

        Args:
            features (pd.DataFrame): Processed features for prediction.

        Returns:
            List[int]: Predicted delay statuses (0 or 1).
        """
        if self._model is None:
            # Train with dummy data if the model is untrained
            dummy_features = features.copy()
            dummy_target = pd.Series([0] * (len(dummy_features) // 2) + [1] * (len(dummy_features) // 2), name='delay')
            self.fit(dummy_features, dummy_target.to_frame())
        return self._model.predict(features).tolist()
