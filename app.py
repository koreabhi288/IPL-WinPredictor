import streamlit as st
import pandas as pd
import joblib
import os
import sklearn
import numpy as np

# Title
st.title('🏏 IPL Win Predictor')

# Load Model
pipe = None
model_load_successful = True

try:
    if not os.path.exists('pipe.pkl'):
        raise FileNotFoundError('❌ The pipe.pkl file is not found in the current working directory.')

    # Show library version info for debugging
    st.caption(f"🔧 Using scikit-learn version: {sklearn.__version__}")
    st.caption(f"🔧 Using numpy version: {np.__version__}")

    pipe = joblib.load('pipe.pkl')

except Exception as e:
    model_load_successful = False
    st.error(f"❌ Failed to load model: {e}")
    st.info("📌 This might be due to a mismatch in scikit-learn or numpy version. Try using the same version that was used to train the model.")

# Proceed only if model is loaded
if model_load_successful:

    # Teams and cities
    teams = [
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
        'Rajasthan Royals', 'Delhi Capitals'
    ]

    cities = [
        'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
        'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
        'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
        'Sharjah', 'Mohali', 'Bengaluru'
    ]

    # Input: Teams
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('🏏 Select the Batting Team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('🎯 Select the Bowling Team', sorted(teams))

    if batting_team == bowling_team:
        st.warning("⚠️ Batting and Bowling team cannot be the same.")

    # Input: City
    selected_city = st.selectbox('📍 Select Match City', sorted(cities))

    # Input: Target
    target = st.number_input('🎯 Target Score', min_value=1)

    # Match progress inputs
    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Current Score', min_value=0)
    with col4:
        overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
    with col5:
        wickets_out = st.number_input('Wickets Out', min_value=0, max_value=10)

    # Prediction
    if st.button('📈 Predict Win Probability'):

        # Input validation
        if score >= target:
            st.warning("🏆 Score already reached or exceeded target.")
        elif overs == 0:
            st.warning("⚠️ Overs must be greater than 0 for prediction.")
        elif wickets_out >= 10:
            st.warning("❌ All wickets are already out.")
        elif batting_team == bowling_team:
            st.warning("⚠️ Teams must be different.")
        else:
            try:
                runs_left = target - score
                balls_left = 120 - int(overs * 6)
                wickets_left = 10 - wickets_out
                crr = score / overs if overs > 0 else 0
                rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

                input_df = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'city': [selected_city],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets_left': [wickets_left],
                    'total_runs_x': [target],
                    'crr': [crr],
                    'rrr': [rrr]
                })

                result = pipe.predict_proba(input_df)
                loss = result[0][0]
                win = result[0][1]

                st.success(f"✅ Prediction Complete!")
                st.subheader(f"🏏 {batting_team}: {round(win * 100)}% chance to win")
                st.subheader(f"🎯 {bowling_team}: {round(loss * 100)}% chance to win")

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
