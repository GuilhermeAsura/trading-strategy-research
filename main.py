import os
import yaml
import pandas as pd
from src.data_loader import load_raw_data
from src.features import generate_technical_features, apply_triple_barrier_labeling
from src.models import XGBoostTrainer

def run_pipeline():
    # load configuration
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(f"--- execution: {config['project_name']} ---")

    # data ingestion & features
    df = load_raw_data(config['data']['raw_path'])
    df_features = generate_technical_features(df)

    # labeling (triple barrier)
    df_labeled = apply_triple_barrier_labeling(
        df_features,
        profit_pct=config['labeling']['profit_margin'],
        loss_pct=config['labeling']['stop_loss'],
        days=config['labeling']['barrier_days']
    )

    # model training
    print("starting model training phase...")
    # list the features we want to use for the model
    feature_cols = ['z_score_200', 'ma_spread', 'k', 'd', 'k_velocity', 'volatility_20']
    
    trainer = XGBoostTrainer(df_labeled, features=feature_cols)
    trainer.prepare_data(test_size=0.2)
    trainer.train()
    
    # evaluation
    trainer.evaluate()

    # persistence
    os.makedirs(os.path.dirname(config['data']['processed_path']), exist_ok=True)
    df_labeled.to_csv(config['data']['processed_path'])
    print(f"pipeline complete. data saved at {config['data']['processed_path']}")

if __name__ == "__main__":
    run_pipeline()