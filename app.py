import streamlit as st
import pickle
import pandas as pd
import os
import sys

# Compatibility fix for older scikit-learn models
def fix_sklearn_compatibility():
    """Fix compatibility issues with older scikit-learn versions"""
    try:
        # Import necessary modules
        import sklearn.compose._column_transformer as ct
        from sklearn.compose import ColumnTransformer
        
        # Add missing attributes for older models
        if not hasattr(ct, '_RemainderColsList'):
            ct._RemainderColsList = list
        if not hasattr(ColumnTransformer, '_RemainderColsList'):
            ColumnTransformer._RemainderColsList = list
            
        # Additional compatibility fixes
        if not hasattr(ct, '_RemainderCols'):
            ct._RemainderCols = object
            
    except ImportError as e:
        st.error(f"Could not import sklearn modules: {e}")
    except Exception as e:
        st.warning(f"Could not apply compatibility fix: {e}")

# Apply the compatibility fix before importing sklearn
fix_sklearn_compatibility()

# Team and city data
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

# Function to load model with multiple attempts
def load_model_safe(filepath):
    """Try multiple methods to load the model"""
    
    # Method 1: Try with pickle
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model, "pickle"
    except Exception as pickle_error:
        st.warning(f"Pickle loading failed: {str(pickle_error)}")
    
    # Method 2: Try with joblib
    try:
        import joblib
        model = joblib.load(filepath)
        return model, "joblib"
    except Exception as joblib_error:
        st.warning(f"Joblib loading failed: {str(joblib_error)}")
    
    # Method 3: Try with different pickle protocols
    for protocol in [2, 3, 4]:
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            return model, f"pickle_protocol_{protocol}"
        except Exception:
            continue
    
    # Method 4: Try loading with compatibility mode
    try:
        # Additional compatibility patches
        import sklearn.utils._bunch
        import sklearn.compose._column_transformer
        
        # Patch missing classes
        if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
            sklearn.compose._column_transformer._RemainderColsList = list
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model, "compatibility_mode"
    except Exception as comp_error:
        st.error(f"Compatibility mode failed: {str(comp_error)}")
    
    raise Exception("Could not load model with any method")

# Streamlit App
st.title('🏏 IPL Win Predictor')
st.markdown("---")

# Check if pickle file exists
if not os.path.exists('pipe.pkl'):
    st.error("❌ The pipe.pkl file is not found. Please ensure it's uploaded to your repository.")
    st.info("💡 Make sure your trained model file 'pipe.pkl' is in the same directory as this app.")
    st.stop()

# Load the model
try:
    with st.spinner("Loading model..."):
        pipe, load_method = load_model_safe('pipe.pkl')
    st.success(f"✅ Model loaded successfully using {load_method}")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.error("Please try the following:")
    st.error("1. Ensure your model was saved with a compatible scikit-learn version")
    st.error("2. Try recreating your model with the current environment")
    st.error("3. Check if the pickle file is corrupted")
    st.stop()

# User Interface
st.subheader("🏆 Match Details")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('🏏 Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('🎯 Select the bowling team', sorted(teams))

selected_city = st.selectbox('🌍 Select host city', sorted(cities))

st.subheader("📊 Match Situation")

col3, col4 = st.columns(2)
with col3:
    target = st.number_input('🎯 Target Score', min_value=1, max_value=300, value=150)
with col4:
    score = st.number_input('💯 Current Score', min_value=0, max_value=299, value=50)

col5, col6, col7 = st.columns(3)
with col5:
    overs = st.number_input('⏰ Overs Completed', min_value=0.1, max_value=20.0, value=5.0, step=0.1)
with col6:
    wickets = st.number_input('🏏 Wickets Lost', min_value=0, max_value=10, value=2)
with col7:
    st.metric("Balls Remaining", f"{int(120 - (overs * 6))}")

# Prediction Button
if st.button('🔮 Predict Match Probability', type="primary"):
    # Validation
    if batting_team == bowling_team:
        st.error("❌ Batting and bowling teams cannot be the same!")
    elif score >= target:
        st.success(f"🎉 {batting_team} has already won the match!")
        st.balloons()
    elif wickets >= 10:
        st.error("❌ All wickets are out! Match over!")
    elif overs >= 20:
        st.error("❌ Innings is complete!")
    else:
        # Calculate match parameters
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        
        # Calculate rates
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        # Validate calculations
        if balls_left <= 0:
            st.error("❌ No balls remaining!")
        elif wickets_left <= 0:
            st.error("❌ No wickets remaining!")
        else:
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
                # Make prediction
                with st.spinner("Calculating probabilities..."):
                    result = pipe.predict_proba(input_df)
                
                loss_prob = result[0][0]
                win_prob = result[0][1]
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Match Prediction")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric(
                        label=f"🏆 {batting_team}",
                        value=f"{round(win_prob*100)}%",
                        delta=f"{round(win_prob*100) - 50}%" if win_prob > 0.5 else None
                    )
                    
                with col_result2:
                    st.metric(
                        label=f"🛡️ {bowling_team}",
                        value=f"{round(loss_prob*100)}%",
                        delta=f"{round(loss_prob*100) - 50}%" if loss_prob > 0.5 else None
                    )
                
                # Progress bars
                st.subheader("📈 Probability Visualization")
                st.progress(win_prob, text=f"{batting_team}: {round(win_prob*100)}%")
                st.progress(loss_prob, text=f"{bowling_team}: {round(loss_prob*100)}%")
                
                # Match situation details
                st.subheader("📋 Match Situation Analysis")
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Runs Needed", runs_left)
                    st.metric("Required Run Rate", f"{rrr:.2f}")
                    
                with col_stats2:
                    st.metric("Balls Remaining", int(balls_left))
                    st.metric("Current Run Rate", f"{crr:.2f}")
                    
                with col_stats3:
                    st.metric("Wickets in Hand", wickets_left)
                    st.metric("Run Rate Difference", f"{rrr - crr:.2f}")
                
                # Insights
                st.subheader("🔍 Match Insights")
                
                if rrr > crr + 2:
                    st.warning("⚠️ Required run rate is significantly higher than current run rate")
                elif rrr < crr:
                    st.info("ℹ️ Team is ahead of the required run rate")
                else:
                    st.success("✅ Match is evenly poised")
                
                if wickets_left <= 3:
                    st.warning("⚠️ Few wickets remaining - batting team under pressure")
                elif wickets_left >= 7:
                    st.success("✅ Plenty of wickets in hand")
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.error("This might be due to:")
                st.error("- Model incompatibility with current data format")
                st.error("- Missing features in the input data")
                st.error("- Corrupted model file")

# Footer
st.markdown("---")
st.markdown("### 📱 About")
st.info("This IPL Win Predictor uses machine learning to predict match outcomes based on current match situation. The model considers factors like runs needed, balls remaining, wickets in hand, and current run rate.")

# Debug information (only show if there's an error)
if st.checkbox("🔧 Show Debug Information"):
    st.subheader("Debug Info")
    st.write("Python version:", sys.version)
    
    try:
        import sklearn
        st.write("Scikit-learn version:", sklearn.__version__)
    except:
        st.write("Scikit-learn: Not available")
    
    try:
        import pandas as pd
        st.write("Pandas version:", pd.__version__)
    except:
        st.write("Pandas: Not available")
    
    st.write("Model file exists:", os.path.exists('pipe.pkl'))
    if os.path.exists('pipe.pkl'):
        st.write("Model file size:", os.path.getsize('pipe.pkl'), "bytes")
