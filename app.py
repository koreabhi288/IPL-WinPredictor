import streamlit as st
import pickle
import pandas as pd
import os

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Check if pickle file exists
if not os.path.exists('pipe.pkl'):
    st.error("The pipe.pkl file is not found. Please ensure it's uploaded to your repository.")
    st.stop()

# Load the model
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target', min_value=1, value=150)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.1, max_value=20.0, value=5.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, value=0)

if st.button('Predict Probability'):
    # Validation
    if batting_team == bowling_team:
        st.error("Batting and bowling teams cannot be the same!")
    elif score >= target:
        st.success(f"{batting_team} has already won!")
    elif wickets >= 10:
        st.error("All wickets are out!")
    elif overs >= 20:
        st.error("Innings is complete!")
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        
        # Calculate current run rate
        crr = score / overs if overs > 0 else 0
        
        # Calculate required run rate
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        # Create input dataframe
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
        
        try:
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]
            
            st.header(f"{batting_team} - {round(win*100)}%")
            st.header(f"{bowling_team} - {round(loss*100)}%")
            
            # Additional match info
            st.subheader("Match Situation:")
            st.write(f"Runs needed: {runs_left}")
            st.write(f"Balls remaining: {int(balls_left)}")
            st.write(f"Wickets remaining: {wickets_left}")
            st.write(f"Current Run Rate: {crr:.2f}")
            st.write(f"Required Run Rate: {rrr:.2f}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
