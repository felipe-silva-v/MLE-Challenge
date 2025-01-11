import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List

class DelayModel:

    def __init__(self):
        self._model = None

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): Raw data.
            target_column (str, optional): If set, the target column is returned separately.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target.
            or
            pd.DataFrame: Features.
        """
        # Feature engineering
        data['high_season'] = data['Fecha-I'].apply(lambda x: 1 if '12-15' <= x[-5:] <= '03-03' or '07-15' <= x[-5:] <= '07-31' or '09-11' <= x[-5:] <= '09-30' else 0)
        data['min_diff'] = (pd.to_datetime(data['Fecha-O']) - pd.to_datetime(data['Fecha-I'])).dt.total_seconds() / 60
        data['period_day'] = pd.to_datetime(data['Fecha-I']).dt.hour.apply(
            lambda x: 'morning' if 5 <= x <= 11 else 'afternoon' if 12 <= x <= 18 else 'night'
        )
        data['delay'] = data['min_diff'].apply(lambda x: 1 if x > 15 else 0)
        
        data = pd.get_dummies(data, columns=['OPERA', 'TIPOVUELO', 'period_day'], drop_first=True)
        
        important_features = [
            'OPERA_Latin American Wings', 'MES_7', 'MES_10', 'OPERA_Grupo LATAM',
            'MES_12', 'TIPOVUELO_I', 'MES_4', 'MES_11', 'OPERA_Sky Airline', 'OPERA_Copa Air'
        ]

        # Ensure all important features are present in the data
        for feature in important_features:
            if feature not in data.columns:
                data[feature] = 0
        
        if target_column:
            target = data[[target_column]]  # Ensure target is a DataFrame
            features = data[important_features]
            return features, target
        
        return data[important_features]

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): Preprocessed features.
            target (pd.DataFrame): Target values.
        """
        # Calculate class weights manually
        n_y0 = (target == 0).sum().values[0]
        n_y1 = (target == 1).sum().values[0]
        class_weights = {1: n_y0 / len(target), 0: n_y1 / len(target)}

        self._model = LogisticRegression(class_weight=class_weights, max_iter=1000)
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): Preprocessed features.

        Returns:
            List[int]: Predicted targets.
        """
        if self._model is None:
            raise ValueError("The model must be trained before making predictions.")
        return self._model.predict(features).tolist()
