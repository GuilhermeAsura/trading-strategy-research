'''
script for generating technical features and applying labeling methods.
'''
import pandas as pd
import numpy as np

def generate_technical_features(df):
    '''
    Calculate log returns for statistical properties
    Log returns are better than simple returns for ml models
    '''
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # distance from 200 period moving average (z-score)
    # this helps identify if the asset is "cheap" or "expensive" relative to its trend
    window = 200
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    df['z_score_200'] = (df['close'] - rolling_mean) / rolling_std
    
    # distance between fast and medium moving averages (9 vs 20)
    # positive means short term momentum is strong
    df['ma_spread'] = (df['9_period'] / df['20_period']) - 1
    
    # stochastic oscillator momentum
    # change in k and d values over 3 days
    df['k_velocity'] = df['k'].diff(3)
    
    # volatility (std dev of returns) for risk adjustment
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    
    return df.dropna()

def apply_triple_barrier_labeling(df, profit_pct=0.05, loss_pct=0.02, days=7):
    '''
    Simplified triple barrier method
    Labels: 1 (profit), -1 (loss), 0 (vertical barrier/time out)
    '''
    df = df.copy() # avoid modifying original df
    labels = []
    prices = df['close'].values
    
    for i in range(len(prices)):
        # define thresholds
        upper_barrier = prices[i] * (1 + profit_pct)
        lower_barrier = prices[i] * (1 - loss_pct)
        
        # look forward in the time window
        future_prices = prices[i+1 : i+1+days]
        
        label = 0 # default is time out
        for p in future_prices:
            if p >= upper_barrier:
                label = 1
                break
            elif p <= lower_barrier:
                label = -1
                break
        labels.append(label)
        
    df.loc[:, 'label'] = labels
    return df