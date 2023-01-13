import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple
import warnings

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


MODELS = (DUMMY:=0, LINEAR:=1, KNEIGHBORS:=2, RANDOM_FOREST:=3, XGBOOST:=4, MLP:=5)


class NordpoolForecaster:
    def __init__(self, path_to_csv: str) -> None:
        self.dataset: pd.DataFrame = pd.read_csv(path_to_csv, index_col=[0])
        self.dataset.index = pd.to_datetime(self.dataset.index, dayfirst=True)
        self.reset()


    def reset(self) -> None:
        self.current_dataset: pd.DataFrame = self.dataset.copy()
        self.period = 1
        self.train_features = []
        self.train_targets = []
        self.test_features = []
        self.test_targets = []
        self.models = [None] * 6
        self.pipelines = [None] * len(self.models)


    def add_seasonal_decomposition_for_column(self, column_name: str, period: int = None) -> None:
        if period is None:
            period = self.period
        decomp = seasonal_decompose(self.current_dataset[column_name], model="addative", period=period, two_sided=False)
        self.current_dataset[f"{column_name}_TREND"] = decomp.trend
        self.current_dataset[f"{column_name}_SEASONAL"] = decomp.seasonal
        self.current_dataset[f"{column_name}_RESIDUAL"] = decomp.resid


    def add_history_for_columns(self, column_names: Tuple[str, ...], period: int, step: int) -> None:
        for name in column_names:
            for i in range(1, period + 1, step):
                self.current_dataset[f"{name}_{i}H_AGO"] = self.current_dataset[name].shift(i)


    def remove_trailling_and_leading_nans(self) -> None:
        for column in self.current_dataset.columns:
            column_series = self.current_dataset[column]
            trailling = column_series.first_valid_index()
            leading = column_series.last_valid_index()
            self.current_dataset = self.current_dataset[trailling: leading]

    
    def create_test_train_datasets(self, target_column: str, target_period: int, test_size: float) -> None:
        def create_target_arrays(target_column: str, target_period: int):
            targets = self.current_dataset[target_column].values
            if target_period < 2:
                return targets
            return np.array([targets[i:i + target_period] for i in range(len(self.current_dataset) - target_period)])

        features = self.current_dataset.drop(columns=[target_column]).values
        targets = create_target_arrays(target_column, target_period)
        train_size = len(self.current_dataset) - int(len(self.current_dataset) * test_size)

        self.train_features = features[:train_size]
        self.test_features = features[train_size:] if target_period < 2 else features[train_size:-target_period]
        self.train_targets = targets[:train_size]
        self.test_targets = targets[train_size:]

        summary = pd.DataFrame({
            "Type": ["Train", "Test"],
            "Features": [self.train_features.shape, self.test_features.shape],
            "Targets": [self.train_targets.shape, self.test_targets.shape]
        })

        print(summary)
        print("\n")


    def init_models(self) -> None:
        self.models = {
            DUMMY: DummyRegressor(strategy="mean"),

            LINEAR: LinearRegression(n_jobs=-1),

            KNEIGHBORS: KNeighborsRegressor(n_jobs=-1, n_neighbors=5, weights="distance"),

            XGBOOST: XGBRegressor(random_state=1, n_jobs=-1, n_estimators=300,
                max_depth=2, min_child_weight=1100, learning_rate=.1),

            RANDOM_FOREST: RandomForestRegressor(random_state=1, n_jobs=-1,
                n_estimators=600, max_samples=2500, min_impurity_decrease=.5),

            MLP: MLPRegressor(random_state=1, hidden_layer_sizes=(100, 20),
                max_iter=1000, activation="identity"),
        }

        self.scores = (
            mean_absolute_percentage_error,
            mean_squared_error,
            r2_score,
        )


    def train(self):
        for i, model in enumerate(MODELS):
            print("#" * 80)
            print("#" * 80)
            print(f"Model ({i + 1}/{len(MODELS)}): {str(self.models[model])}")
            print('\n')

            pipeline = make_pipeline(
                StandardScaler(),
                self.models[model]
            )

            pipeline.fit(self.train_features, self.train_targets)
            train_predictions = self.predict(pipeline, self.train_features)
            test_predictions = self.predict(pipeline,  self.test_features)

            print("Training")
            self.evaluate_model(self.train_targets, train_predictions)

            print("Testing")
            self.evaluate_model(self.test_targets, test_predictions)
            
            self.pipelines[model] = pipeline


    def evaluate_model(self, y_true, y_pred):
        for score in self.scores:
            print(f'{score.__name__:30s}: {score(y_true, y_pred):15.5f}')
        print('\n')
    

    def predict(self, pipeline, features: np.ndarray) -> np.ndarray:
        return pipeline.predict(features)


    def run(self) -> None:
        self.period = 24
        target_column = "Electricity_Price"
        factor_columns = [
            "Consumption", "Production", "Exchange", "Gasoline_Price",
        ]
        self.current_dataset = self.current_dataset.drop(columns="Wind_Power")
        self.add_history_for_columns(column_names=[target_column], period=self.period, step=1)
        self.add_history_for_columns(column_names=factor_columns, period=self.period, step=1)
        self.current_dataset = self.current_dataset.drop(columns=factor_columns)
        self.remove_trailling_and_leading_nans()
        self.create_test_train_datasets(target_column, target_period=self.period, test_size=.1)
        self.init_models()
        self.train()


if __name__ == "__main__":
    forecaster = NordpoolForecaster("data/nordpool.csv")
    forecaster.run()
    

