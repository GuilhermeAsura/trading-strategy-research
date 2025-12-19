import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
    
class LinearRegressionTrainer:
    '''
    trainer to find the optimized linear combination of procurement parameters
    '''
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.model = LinearRegression()
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None

    def prepare_data(self, test_size=0.2):
        # for general cost analysis, random shuffle is usually acceptable
        x = self.df[self.features]
        y = self.df[self.target]
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=42
        )

    def train(self):
        # train the linear model to find the weights (beta coefficients)
        self.model.fit(self.x_train, self.y_train)
        
        # display the optimized linear combination coefficients
        print("\n- - - optimized linear combination - - -")
        print(f"base cost (intercept): {self.model.intercept_:.4f}")
        
        coefficients = pd.DataFrame({
            'parameter': self.features,
            'weight': self.model.coef_
        }).sort_values(by='weight', ascending=False)
        
        print(coefficients)
        return self.model

    def evaluate(self):
        # evaluate regression performance using mae and r-squared
        preds = self.model.predict(self.x_test)
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        
        print(f"\ncost model evaluation:")
        print(f"mean absolute error (mae): {mae:.4f}")
        print(f"r-squared score: {r2:.4f}")
        
        return {"mae": mae, "r2": r2}

    def get_formula(self):
        '''
        returns the mathematical formula string for easy automation deployment
        '''
        terms = [f"({coef:.4f} * {feat})" for coef, feat in zip(self.model.coef_, self.features)]
        formula = f"cost = {self.model.intercept_:.4f} + " + " + ".join(terms)
        return formula

class PortfolioOptimizer:
    '''
    finds optimal weights to linearize the cumulative equity curve
    '''
    def __init__(self, returns_matrix, total_budget=25):
        self.returns = returns_matrix
        self.total_budget = total_budget
        self.weights = None

    def objective_function(self, weights):
        # weighted returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        cumulative_equity = portfolio_returns.cumsum()
        
        y = cumulative_equity.values
        if not np.all(np.isfinite(y)) or len(y) < 2:
            return 1e12
        
        x = np.arange(len(y))
        
        try:
            # linear fit
            slope, intercept = np.polyfit(x, y, 1)
            linear_trend = slope * x + intercept
            
            # mse of residuals
            mse = np.mean((y - linear_trend)**2)
            
            # slope penalty: we want steep upward curves
            # if slope is low, penalty increases
            slope_penalty = 1 / (slope + 1e-6) if slope > 0 else 1e12
            
            # total risk: combine variance with negative slope check
            score = mse * slope_penalty
            
            if y[-1] <= y[0]:
                score += 1e12
                
            return score
        except:
            return 1e12

    def optimize(self):
        num_assets = self.returns.shape[1]
        if num_assets == 0: return {}

        # set a max weight per asset (e.g., 20% of total budget)
        max_weight_per_asset = self.total_budget * 0.20 
        
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - self.total_budget})
        
        # update bounds: (min, max) per asset
        bnds = tuple((0, max_weight_per_asset) for _ in range(num_assets))
        
        # initial guess: back to equal distribution for stability
        init_guess = np.full(num_assets, self.total_budget / num_assets)

        result = minimize(
            self.objective_function, 
            init_guess, 
            method='SLSQP', 
            bounds=bnds, 
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        
        self.weights = result.x
        return dict(zip(self.returns.columns, self.weights))