"""
IPL Match Winner Prediction - Model Training Script
Trains a Machine Learning model using historical IPL match data.
Compares Logistic Regression and Random Forest Classifier,
evaluates performance, and saves the trained Scikit-learn Pipeline.
"""

import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'ipl_matches.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'ipl_prediction_model.pkl')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

MATCHES_URL = 'https://raw.githubusercontent.com/shaadclt/IPL-Win-Probability-Predictor/main/matches.csv'
DELIVERIES_URL = 'https://raw.githubusercontent.com/shaadclt/IPL-Win-Probability-Predictor/main/deliveries.csv'

CORE_TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

def prepare_dataset():
    """
    Ensures data/ipl_matches.csv is available.
    If not, downloads matches.csv and deliveries.csv,
    performs data cleaning and feature engineering, and creates ipl_matches.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"[INFO] Found existing processed dataset at: {PROCESSED_DATA_PATH}")
        return pd.read_csv(PROCESSED_DATA_PATH)

    matches_path = os.path.join(DATA_DIR, 'matches.csv')
    deliveries_path = os.path.join(DATA_DIR, 'deliveries.csv')

    if not os.path.exists(matches_path):
        print(f"[INFO] Downloading matches.csv from {MATCHES_URL}...")
        req = urllib.request.Request(MATCHES_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(matches_path, 'wb') as f:
            f.write(resp.read())
        print("[INFO] matches.csv downloaded successfully.")

    if not os.path.exists(deliveries_path):
        print(f"[INFO] Downloading deliveries.csv from {DELIVERIES_URL}...")
        req = urllib.request.Request(DELIVERIES_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(deliveries_path, 'wb') as f:
            f.write(resp.read())
        print("[INFO] deliveries.csv downloaded successfully.")

    print("[INFO] Loading and preprocessing raw match and delivery records...")
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # Calculate first innings total to establish target
    total_score_df = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
    total_score_df = total_score_df[total_score_df['inning'] == 1]
    total_score_df['target'] = total_score_df['total_runs'] + 1

    match_df = matches.merge(total_score_df[['match_id', 'target']], left_on='id', right_on='match_id')

    # Harmonize team names across IPL seasons
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }
    match_df['team1'] = match_df['team1'].replace(team_mapping)
    match_df['team2'] = match_df['team2'].replace(team_mapping)
    deliveries['batting_team'] = deliveries['batting_team'].replace(team_mapping)
    deliveries['bowling_team'] = deliveries['bowling_team'].replace(team_mapping)

    # Filter for active core franchises and non-rain affected matches
    match_df = match_df[match_df['team1'].isin(CORE_TEAMS) & match_df['team2'].isin(CORE_TEAMS)]
    match_df = match_df[match_df['dl_applied'] == 0]

    # Impute missing city names from venue string
    match_df['city'] = match_df['city'].fillna(match_df['venue'].apply(lambda x: str(x).split()[0]))
    match_df['city'] = match_df['city'].replace({'Bangalore': 'Bengaluru'})

    # Merge ball-by-ball second innings data
    delivery_df = match_df[['match_id', 'city', 'winner', 'target']].merge(deliveries, on='match_id')
    delivery_df = delivery_df[delivery_df['inning'] == 2]

    # Compute running match situation metrics
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs'].cumsum()
    delivery_df['runs_left'] = delivery_df['target'] - delivery_df['current_score']
    delivery_df['balls_left'] = 120 - ((delivery_df['over'] - 1) * 6 + delivery_df['ball'])

    # Track wickets remaining
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna('0')
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == '0' else 1)
    wickets_lost = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
    delivery_df['wickets_left'] = 10 - wickets_lost

    # Compute run rates
    delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
    delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']

    # Target variable: 1 if batting team won, 0 if bowling team won
    delivery_df['result'] = (delivery_df['batting_team'] == delivery_df['winner']).astype(int)

    final_df = delivery_df[[
        'batting_team', 'bowling_team', 'city', 'runs_left',
        'balls_left', 'wickets_left', 'target', 'crr', 'rrr', 'result'
    ]]
    final_df = final_df.dropna()
    final_df = final_df[final_df['balls_left'] > 0]
    final_df = final_df[final_df['runs_left'] >= 0]

    print(f"[INFO] Saving cleaned dataset ({final_df.shape[0]} samples) to {PROCESSED_DATA_PATH}...")
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    return final_df

def train_and_evaluate():
    """
    Trains the ML model pipeline and saves the model artifact.
    """
    df = prepare_dataset()
    print(f"[INFO] Dataset loaded with shape: {df.shape}")

    X = df.drop('result', axis=1)
    y = df['result']

    # Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # Categorical Preprocessing
    categorical_features = ['batting_team', 'bowling_team', 'city']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # 1. Baseline Model: Logistic Regression
    print("\n" + "="*50)
    print("EXPERIMENT 1: Logistic Regression Classifier")
    print("="*50)
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    print(f"Logistic Regression Validation Accuracy: {lr_accuracy * 100:.2f}%\n")

    # 2. Target Model: Random Forest Classifier
    print("="*50)
    print("EXPERIMENT 2: Random Forest Classifier")
    print("="*50)
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    print("[INFO] Fitting Random Forest Pipeline...")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)

    print(f"\nRandom Forest Validation Accuracy: {rf_accuracy * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, rf_preds, target_names=['Bowling Team Wins (0)', 'Batting Team Wins (1)']))

    # Choose best model (Random Forest)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[INFO] Saving trained model to {MODEL_PATH}...")
    joblib.dump(rf_pipeline, MODEL_PATH)

    # Save metadata (unique teams & cities) for frontend & API synchronization
    unique_teams = sorted(list(set(df['batting_team'].unique()) | set(df['bowling_team'].unique())))
    unique_cities = sorted([str(c) for c in df['city'].unique()])

    metadata = {
        'model_name': 'RandomForestClassifier',
        'accuracy': float(rf_accuracy),
        'comparison': {
            'logistic_regression_accuracy': float(lr_accuracy),
            'random_forest_accuracy': float(rf_accuracy)
        },
        'teams': unique_teams,
        'cities': unique_cities,
        'features': list(X.columns)
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Model metadata saved to {METADATA_PATH}.")
    print(f"[SUCCESS] Training complete! Model is ready for deployment.")

if __name__ == '__main__':
    train_and_evaluate()

