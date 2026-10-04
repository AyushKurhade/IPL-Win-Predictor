"""
IPL Match Prediction - Flask Backend Application
Provides web routes and a REST API endpoint (/predict) for match outcome prediction.
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ipl_prediction_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

# Team Abbreviations for professional display
TEAM_ABBREVIATIONS = {
    'Chennai Super Kings': 'CSK',
    'Delhi Capitals': 'DC',
    'Kings XI Punjab': 'PBKS',
    'Kolkata Knight Riders': 'KKR',
    'Mumbai Indians': 'MI',
    'Rajasthan Royals': 'RR',
    'Royal Challengers Bangalore': 'RCB',
    'Sunrisers Hyderabad': 'SRH'
}

# Load model and metadata at startup
model = None
metadata = {}

def load_resources():
    global model, metadata
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read metadata: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("[INFO] Trained ML model loaded successfully into memory.")
        except Exception as e:
            print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
            model = None
    else:
        print(f"[WARNING] Model file not found at {MODEL_PATH}. Please run train_model.py first.")

load_resources()

# Fallback lists if metadata not found
TEAMS = metadata.get('teams', sorted(list(TEAM_ABBREVIATIONS.keys())))
CITIES = metadata.get('cities', [
    'Abu Dhabi', 'Ahmedabad', 'Bengaluru', 'Chandigarh', 'Chennai',
    'Delhi', 'Dharamsala', 'Dubai', 'Hyderabad', 'Indore', 'Jaipur',
    'Kolkata', 'Mohali', 'Mumbai', 'Pune', 'Raipur', 'Ranchi', 'Sharjah', 'Visakhapatnam'
])

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html', active_page='home')

@app.route('/prediction')
def prediction_page():
    """Main Prediction page route"""
    return render_template(
        'prediction.html',
        active_page='prediction',
        teams=TEAMS,
        cities=CITIES,
        team_abbr=TEAM_ABBREVIATIONS
    )

@app.route('/about')
def about_page():
    """About project page route"""
    return render_template(
        'about.html',
        active_page='about',
        metadata=metadata
    )

@app.route('/api/metadata')
def get_metadata():
    """Endpoint providing available teams and cities"""
    return jsonify({
        'teams': TEAMS,
        'cities': CITIES,
        'team_abbreviations': TEAM_ABBREVIATIONS
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction API endpoint.
    Accepts match situation JSON and returns winning team and probabilities.
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'ML model is not loaded. Please run "python train_model.py" to train and generate the model file.'
        }), 503

    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request. Please provide JSON payload.'}), 400

        # Extract parameters
        batting_team = data.get('batting_team', '').strip()
        bowling_team = data.get('bowling_team', '').strip()
        city = data.get('city', '').strip()
        target_raw = data.get('target')
        score_raw = data.get('current_score')
        overs_raw = data.get('overs')
        wickets_raw = data.get('wickets')

        # 1. Validation: Non-empty check
        if not batting_team or not bowling_team or not city:
            return jsonify({'success': False, 'error': 'Please select Batting Team, Bowling Team, and Host City.'}), 400

        if target_raw is None or score_raw is None or overs_raw is None or wickets_raw is None:
            return jsonify({'success': False, 'error': 'All numeric match fields are required.'}), 400

        # 2. Validation: Teams must be different
        if batting_team == bowling_team:
            return jsonify({'success': False, 'error': 'Batting and Bowling teams cannot be the same. Please choose different teams.'}), 400

        # Validate teams exist in our dataset
        if batting_team not in TEAMS or bowling_team not in TEAMS:
            return jsonify({'success': False, 'error': f'Selected team is not supported. Supported teams: {", ".join(TEAMS)}'}), 400

        # Type conversion
        try:
            target = int(target_raw)
            current_score = int(score_raw)
            overs = float(overs_raw)
            wickets = int(wickets_raw)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Target, Score, Overs, and Wickets must be valid numbers.'}), 400

        # 3. Numeric bounds validation
        if target <= 0:
            return jsonify({'success': False, 'error': 'Target score must be greater than 0.'}), 400

        if current_score < 0:
            return jsonify({'success': False, 'error': 'Current score cannot be negative.'}), 400

        if current_score >= target:
            return jsonify({'success': False, 'error': f'Current score ({current_score}) has already reached or exceeded the target ({target}). Batting team has already won.'}), 400

        if overs < 0.0 or overs > 20.0:
            return jsonify({'success': False, 'error': 'Overs completed must be between 0 and 20.'}), 400

        # Validate ball decimal portion (e.g. 12.0 to 12.5 are valid balls; 12.6 is 13.0)
        over_whole = int(overs)
        ball_part = round((overs - over_whole) * 10)
        if ball_part > 5:
            return jsonify({'success': False, 'error': f'Invalid overs format: "{overs}". A cricket over has only 6 balls (.0 to .5). Example: 12.3'}), 400

        if wickets < 0 or wickets > 10:
            return jsonify({'success': False, 'error': 'Wickets out must be between 0 and 10.'}), 400

        # Special boundary scenario: All out
        if wickets == 10:
            # Batting team is bowled out before reaching target -> Bowling team wins
            bowling_abbr = TEAM_ABBREVIATIONS.get(bowling_team, '')
            winner_str = f"{bowling_team} ({bowling_abbr})" if bowling_abbr else bowling_team
            return jsonify({
                'success': True,
                'winner': winner_str,
                'likely_winner_raw': bowling_team,
                'batting_team': batting_team,
                'bowling_team': bowling_team,
                'batting_team_probability': 0.0,
                'bowling_team_probability': 1.0,
                'batting_win_percentage': 0,
                'bowling_win_percentage': 100,
                'note': 'Batting team is all out (10 wickets down).'
            })

        # Calculate balls bowled and balls remaining
        balls_bowled = over_whole * 6 + ball_part
        balls_left = 120 - balls_bowled
        if balls_left <= 0:
            # 20 overs finished and current_score < target -> Bowling team wins
            bowling_abbr = TEAM_ABBREVIATIONS.get(bowling_team, '')
            winner_str = f"{bowling_team} ({bowling_abbr})" if bowling_abbr else bowling_team
            return jsonify({
                'success': True,
                'winner': winner_str,
                'likely_winner_raw': bowling_team,
                'batting_team': batting_team,
                'bowling_team': bowling_team,
                'batting_team_probability': 0.0,
                'bowling_team_probability': 1.0,
                'batting_win_percentage': 0,
                'bowling_win_percentage': 100,
                'note': 'All 20 overs completed without reaching target.'
            })

        # Feature calculations
        runs_left = target - current_score
        wickets_left = 10 - wickets
        crr = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0.0
        rrr = (runs_left * 6) / balls_left

        # Build feature DataFrame matching pipeline schema
        features_df = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets_left': wickets_left,
            'target': target,
            'crr': crr,
            'rrr': rrr
        }])

        # Predict win probabilities using trained Scikit-learn pipeline
        probabilities = model.predict_proba(features_df)[0]
        bowling_prob = float(probabilities[0])
        batting_prob = float(probabilities[1])

        # Normalize to ensure exactly 100% total
        total = bowling_prob + batting_prob
        if total > 0:
            batting_prob = batting_prob / total
            bowling_prob = bowling_prob / total

        batting_pct = round(batting_prob * 100, 1)
        bowling_pct = round(100.0 - batting_pct, 1)

        # Determine winner
        if batting_prob >= bowling_prob:
            likely_team = batting_team
            win_prob_val = batting_prob
        else:
            likely_team = bowling_team
            win_prob_val = bowling_prob

        abbr = TEAM_ABBREVIATIONS.get(likely_team, '')
        winner_formatted = f"{likely_team} ({abbr})" if abbr else likely_team

        return jsonify({
            'success': True,
            'winner': winner_formatted,
            'likely_winner_raw': likely_team,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'batting_team_abbr': TEAM_ABBREVIATIONS.get(batting_team, ''),
            'bowling_team_abbr': TEAM_ABBREVIATIONS.get(bowling_team, ''),
            'batting_team_probability': round(batting_prob, 4),
            'bowling_team_probability': round(bowling_prob, 4),
            'batting_win_percentage': batting_pct,
            'bowling_win_percentage': bowling_pct,
            'winning_probability_pct': round(win_prob_val * 100, 1),
            'stats': {
                'runs_left': runs_left,
                'balls_left': balls_left,
                'wickets_left': wickets_left,
                'crr': round(crr, 2),
                'rrr': round(rrr, 2)
            }
        })

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)

