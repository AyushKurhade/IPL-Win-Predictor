# IPL Match Prediction 🏏

An end-to-end Machine Learning web application designed to predict the winning team and real-time winning probabilities during the second innings of an Indian Premier League (IPL) cricket match.

Built using **Python, Flask, Scikit-learn, HTML5, CSS3, JavaScript, and Bootstrap 5**.

---

## 📌 Project Overview

**IPL Match Prediction** is an engineering machine learning solution that estimates match outcomes based on historical IPL data and live second-innings chasing game states. 

Given match inputs such as batting team, bowling team, host city, target score, current score, overs completed, and wickets lost, the application calculates dynamic derived features:
- **Runs Left**: `Target - Current Score`
- **Balls Remaining**: `120 - Balls Bowled`
- **Wickets Remaining**: `10 - Wickets Lost`
- **Current Run Rate (CRR)**: `(Current Score × 6) / Balls Bowled`
- **Required Run Rate (RRR)**: `(Runs Left × 6) / Balls Remaining`

These features are processed through a Scikit-learn pipeline (Categorical `OneHotEncoder` + `RandomForestClassifier`), outputting calibrated win probabilities for both teams without page reloads.

---

## 🌟 Key Features

- **Real Machine Learning Pipeline**: Trained on official historical ball-by-ball IPL match records (over 71,800 game state instances).
- **Random Forest Classification**: High-performance ensemble classifier evaluated at **95.8% validation accuracy** (compared with Logistic Regression at 81.6%).
- **Interactive Single-Page AJAX UI**: Uses JavaScript `fetch()` API for asynchronous prediction and smooth probability bar animations.
- **Strict Input Validation**:
  - Batting and bowling teams cannot be identical.
  - Target score must be positive.
  - Current score cannot be negative or equal/greater than target in a chasing scenario.
  - Overs validation adhering to cricket over rules (0 to 20 overs, with ball fractions strictly `.0` to `.5`).
  - Wickets validation (0 to 10).
- **Responsive Modern Navy-Blue Design**: Custom styled interface built on Bootstrap 5 that adapts seamlessly to desktop, laptop, tablet, and mobile screens.
- **RESTful API (`POST /predict`)**: Fully decoupled endpoint ready for integration or testing via cURL, Postman, or external services.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+ Fetch API), Bootstrap 5 |
| **Backend** | Python 3, Flask 3, Flask-CORS |
| **Machine Learning** | Scikit-learn, Pandas, NumPy, Joblib |
| **Model Architecture** | `Pipeline` with `ColumnTransformer` (OneHotEncoder) & `RandomForestClassifier` |

---
## 📸 Application Screenshots

### 1. Landing Page
Modern, responsive hero section showcasing project highlights and model performance.

![Home Page](screenshots/ui.jpg)

---

### 2. Match Prediction Inputs
Form interface with input validation for teams, venue, chase target, scores, overs, and wickets.

| Empty Input State | Filled Game State |
| :---: | :---: |
| ![Prediction Input Form Blank](screenshots/inputfeild.jpg) | ![Prediction Input Form Filled](screenshots/input.jpg) |

---

### 3. Prediction Result & Live Win Probabilities
Dynamic result card displaying likely winner, animated probability bars, and computed chase metrics.

![Prediction Output](screenshots/output.jpg)

---


---

## 📊 Dataset Details

The model uses the historical ball-by-ball IPL dataset:
- `matches.csv`: High-level match outcomes, venues, cities, and teams.
- `deliveries.csv`: Ball-by-ball deliveries across all IPL seasons.
- `data/ipl_matches.csv`: The cleaned, derived 2nd innings chasing dataset (71,859 records) bundled directly in the repository.

### Supported Teams (8 Core Franchises)
- Chennai Super Kings (CSK)
- Delhi Capitals (DC)
- Kings XI Punjab (PBKS)
- Kolkata Knight Riders (KKR)
- Mumbai Indians (MI)
- Rajasthan Royals (RR)
- Royal Challengers Bangalore (RCB)
- Sunrisers Hyderabad (SRH)

---

## 📁 Project Directory Structure

```
IPL Win Predictor/
├── app.py                     # Flask web server and /predict REST API
├── train_model.py             # Script to load data, clean, train ML model, evaluate & save
├── requirements.txt           # Python dependencies
├── README.md                  # Complete documentation
│
├── data/
│   ├── deliveries.csv         # Raw historical ball-by-ball deliveries
│   ├── matches.csv            # Raw historical match results
│   └── ipl_matches.csv        # Preprocessed chasing match-situation dataset
│
├── models/
│   ├── ipl_prediction_model.pkl # Trained Scikit-learn Pipeline
│   └── model_metadata.json      # Saved model metadata, teams, and city lists
│
├── templates/
│   ├── base.html              # Shared navbar, branding, and layout structure
│   ├── index.html             # Home page with hero section & navigation
│   ├── prediction.html        # Main prediction page with input form & result card
│   └── about.html             # Project details, technologies & pipeline explanation
│
└── static/
    ├── css/
    │   └── style.css          # Navy-blue themed CSS styling & responsive adjustments
    ├── js/
    │   └── prediction.js      # Frontend validation, AJAX submission, and animation logic
    └── images/
        └── cricket_ball.svg   # Vector cricket ball brand icon
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Workspace
Open the project directory in VS Code or your terminal:
```bash
cd "c:\Users\ASUS\Desktop\Project's\IPL Win Predictor"
```

### 2. Create and Activate a Virtual Environment

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Train the Machine Learning Model (Optional / Pre-trained)
The pre-trained model is already included in `models/ipl_prediction_model.pkl`. To re-train or view real evaluation statistics:
```bash
python train_model.py
```
**Sample Output:**
```
==================================================
EXPERIMENT 1: Logistic Regression Classifier
==================================================
Logistic Regression Validation Accuracy: 81.56%

==================================================
EXPERIMENT 2: Random Forest Classifier
==================================================
Random Forest Validation Accuracy: 95.82%

--- Classification Report ---
                       precision    recall  f1-score   support
Bowling Team Wins (0)       0.97      0.96      0.96      7814
Batting Team Wins (1)       0.95      0.96      0.95      6558

             accuracy                           0.96     14372
```

### 2. Start the Flask Application
```bash
python app.py
```

### 3. Open in Browser
Navigate to:
```
http://127.0.0.1:5000
```
- **Home**: `http://127.0.0.1:5000/`
- **Prediction**: `http://127.0.0.1:5000/prediction`
- **About**: `http://127.0.0.1:5000/about`

---

## 📖 How to Use the Prediction Page

1. Navigate to the **Prediction** page from the top navigation bar.
2. **Select Batting Team** (e.g., *Mumbai Indians*).
3. **Select Bowling Team** (e.g., *Chennai Super Kings*).
4. **Select Host City** (e.g., *Mumbai*).
5. Enter **Target Score** (e.g., `180`).
6. Enter **Current Score** (e.g., `120`).
7. Enter **Overs Completed** (e.g., `15.0` or `12.3`).
8. Enter **Wickets Out** (e.g., `4`).
9. Click **Predict Winner**.
10. The result card displays below the form with:
    - 🏆 Likely Winner (e.g., `Mumbai Indians (MI)`)
    - Winning Probability (e.g., `68.6%`)
    - Smooth animated percentage bars for both teams
    - Real-time chase stats (Runs Left, Balls Left, Wickets Left, CRR, RRR).

---

## 📡 REST API Documentation

### `POST /predict`

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "batting_team": "Mumbai Indians",
  "bowling_team": "Chennai Super Kings",
  "city": "Mumbai",
  "target": 180,
  "current_score": 120,
  "overs": 15.0,
  "wickets": 4
}
```

**Response (`200 OK`):**
```json
{
  "success": true,
  "winner": "Mumbai Indians (MI)",
  "likely_winner_raw": "Mumbai Indians",
  "batting_team": "Mumbai Indians",
  "bowling_team": "Chennai Super Kings",
  "batting_team_abbr": "MI",
  "bowling_team_abbr": "CSK",
  "batting_team_probability": 0.6861,
  "bowling_team_probability": 0.3139,
  "batting_win_percentage": 68.6,
  "bowling_win_percentage": 31.4,
  "winning_probability_pct": 68.6,
  "stats": {
    "runs_left": 60,
    "balls_left": 30,
    "wickets_left": 6,
    "crr": 8.0,
    "rrr": 12.0
  }
}
```

---

## 🛡️ Input Validation & Error Handling

- Returns `400 Bad Request` with human-readable error messages for:
  - Identical batting and bowling teams.
  - Non-existent teams or unsupported values.
  - Invalid overs format (e.g. `12.7` because a cricket over has only 6 balls).
  - Target score `≤ 0` or negative scores.
  - Current score reaching or exceeding target (chase already accomplished).
  - Wickets outside the `0 - 10` range.
- Returns `503 Service Unavailable` if the ML model file is not present on the server.

