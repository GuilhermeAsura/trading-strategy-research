import xgboost as xgb
from sklearn.metrics import classification_report
import pandas as pd

class XGBoostTrainer:
    def __init__(self, df, features, target='label'):
        self.df = df
        self.features = features
        self.target = target
        self.model = None
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None

    def prepare_data(self, test_size=0.2):
        # time series splitting: no shuffling to avoid look-ahead bias
        x = self.df[self.features]
        y = self.df[self.target]
        
        # xgboost multi-class requires labels starting from 0
        # converting [-1, 0, 1] to [0, 1, 2]
        y_adjusted = y + 1
        
        split_idx = int(len(x) * (1 - test_size))
        
        self.x_train = x.iloc[:split_idx]
        self.y_train = y_adjusted.iloc[:split_idx]
        self.x_test = x.iloc[split_idx:]
        self.y_test = y_adjusted.iloc[split_idx:]

    def train(self):
        # objective multi:softmax for triple barrier (3 classes)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def evaluate(self):
        # generate predictions
        preds = self.model.predict(self.x_test)
        
        # map back to original labels for readability
        actual = self.y_test - 1
        predicted = preds - 1
        
        # printing metrics
        report = classification_report(actual, predicted)
        print("model evaluation report:\n", report)
        return report