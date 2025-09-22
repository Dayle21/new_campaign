import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
import datetime as dt
import pickle
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Company Campaign Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .company-tagline {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .new-campaign-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .comparison-excellent {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    .comparison-good {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .comparison-needs-improvement {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .prediction-excellent {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    .prediction-good {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .prediction-poor {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .insight-highlight {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .success-banner {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .ai-analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .debug-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Data file path
DATA_FILE = "synthetic_marketing_campaigns.xlsx"
PROCESSED_DATA_FILE = "processed_campaign_data.csv"
MODEL_FILE = "campaign_models.pkl"

@st.cache_data
def load_company_data():
    """Load the company's historical campaign data"""
    try:
        st.write(f"Checking for {DATA_FILE}...")
        if os.path.exists(DATA_FILE):
            st.write(f"Found {DATA_FILE}, reading Excel...")
            df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
            st.write("Processing data...")
            df = process_campaign_data(df)
            df.to_csv(PROCESSED_DATA_FILE, index=False)
            st.sidebar.success("✅ Loaded and processed company data from Excel")
        elif os.path.exists(PROCESSED_DATA_FILE):
            st.write(f"Found {PROCESSED_DATA_FILE}, reading CSV...")
            df = pd.read_csv(PROCESSED_DATA_FILE)
            st.sidebar.success("✅ Loaded existing processed data (CSV)")
        else:
            st.sidebar.error(f"❌ Company data file not found. Please ensure '{DATA_FILE}' is in the same directory.")
            return None
        
        # Verify required columns
        required_columns = {'Campaign_Name', 'Channel', 'Audience', 'Cost', 'Revenue', 'Impressions', 'Clicks', 'Conversions', 'CPA', 'ROI'}
        if not all(col in df.columns for col in required_columns):
            missing = required_columns - set(df.columns)
            st.sidebar.error(f"❌ Missing required columns: {missing}")
            return None
        
        return df
    
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
        return None

def process_campaign_data(df):
    """Process and clean the campaign data"""
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        elif 'date' in col.lower():
            # Handle date columns
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col].fillna(pd.Timestamp(dt.datetime.now().date()), inplace=True)
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
    
    df = df.drop_duplicates()
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != 'ROI':
            upper_bound = df[col].quantile(0.99)
            lower_bound = df[col].quantile(0.01)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    if 'Revenue' in df.columns and 'Cost' in df.columns:
        if 'Profit' not in df.columns:
            df['Profit'] = df['Revenue'] - df['Cost']
        if 'Profit_Margin' not in df.columns:
            df['Profit_Margin'] = (df['Profit'] / df['Revenue']) * 100
        if 'ROAS' not in df.columns:
            df['ROAS'] = df['Revenue'] / df['Cost']
    
    if 'Clicks' in df.columns and 'Impressions' in df.columns and 'CTR' not in df.columns:
        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
    
    if 'Conversions' in df.columns and 'Clicks' in df.columns and 'CVR' not in df.columns:
        df['CVR'] = (df['Conversions'] / df['Clicks']) * 100
    
    if 'ROI' in df.columns:
        df['ROI_Category'] = pd.cut(df['ROI'], 
                                   bins=[-np.inf, 0, 1, 2, np.inf], 
                                   labels=['Loss', 'Break_Even', 'Profitable', 'High_Profit'])
    
    return df

@st.cache_data
def train_company_models(df):
    """Train models on company data with improved error handling"""
    try:
        # Validate required columns
        required_columns = ['ROI', 'Cost', 'Revenue', 'Channel', 'Audience']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Check for sufficient data
        if len(df) < 10:
            st.error("Insufficient data for model training. Need at least 10 campaigns.")
            return None
        
        # Handle missing values more robustly
        df_model = df.copy()
        
        # Fill missing values
        for col in df_model.columns:
            if df_model[col].dtype in ['float64', 'int64']:
                df_model[col].fillna(df_model[col].median(), inplace=True)
            else:
                mode_val = df_model[col].mode()[0] if not df_model[col].mode().empty else 'Unknown'
                df_model[col].fillna(mode_val, inplace=True)
        
        # Prepare features
        categorical_cols = df_model.select_dtypes(include=['object']).columns
        le_dict = {}
        df_encoded = df_model.copy()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in ['Campaign_Name', 'Start_Date', 'End_Date'] and col in df_model.columns:
                try:
                    le = LabelEncoder()
                    df_encoded[f'{col}_Encoded'] = le.fit_transform(df_model[col].astype(str))
                    le_dict[col] = le
                except Exception as e:
                    st.warning(f"Could not encode column {col}: {str(e)}")
                    continue
        
        # Define feature columns more carefully
        feature_cols = []
        
        # Add numerical features (excluding target variable)
        numerical_features = ['Cost', 'Revenue', 'Impressions', 'Clicks', 'Conversions', 
                            'CPA', 'CTR', 'CVR', 'Profit', 'Profit_Margin', 'ROAS']
        
        for col in numerical_features:
            if col in df_encoded.columns and col != 'ROI':
                if not np.isfinite(df_encoded[col]).all():
                    df_encoded[col] = df_encoded[col].replace([np.inf, -np.inf], np.nan)
                    df_encoded[col].fillna(df_encoded[col].median(), inplace=True)
                feature_cols.append(col)
        
        # Add encoded categorical features
        for col in df_encoded.columns:
            if col.endswith('_Encoded'):
                feature_cols.append(col)
        
        if not feature_cols:
            st.error("No valid features found for training")
            return None
        
        # Prepare training data
        X = df_encoded[feature_cols].copy()
        y = df_encoded['ROI'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        y = y.fillna(y.median())
        
        # Check for valid target values
        if not np.isfinite(y).all():
            y = y.replace([np.inf, -np.inf], np.nan)
            y = y.fillna(y.median())
        
        # Ensure we have valid data
        if X.shape[0] == 0 or len(feature_cols) == 0:
            st.error("No valid training data available")
            return None
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
        except Exception as e:
            st.warning(f"Could not stratify split, using simple split: {str(e)}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Scale features for Linear Regression
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            st.error(f"Error in feature scaling: {str(e)}")
            return None
        
        models_results = {}
        
        # Train Random Forest
        try:
            st.write("Training Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            
            if not np.isfinite(rf_pred).all():
                st.warning("Random Forest produced invalid predictions")
                rf_r2, rf_rmse = -1, float('inf')
            else:
                rf_r2 = r2_score(y_test, rf_pred)
                rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            
            models_results['Random Forest'] = {
                'model': rf_model, 
                'r2': rf_r2, 
                'rmse': rf_rmse,
                'predictions': rf_pred
            }
            st.success(f"Random Forest trained - R²: {rf_r2:.3f}, RMSE: {rf_rmse:.3f}")
            
        except Exception as e:
            st.error(f"Random Forest training failed: {str(e)}")
            models_results['Random Forest'] = {'model': None, 'r2': -1, 'rmse': float('inf')}
        
        # Train Linear Regression
        try:
            st.write("Training Linear Regression...")
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            
            if not np.isfinite(lr_pred).all():
                st.warning("Linear Regression produced invalid predictions")
                lr_r2, lr_rmse = -1, float('inf')
            else:
                lr_r2 = r2_score(y_test, lr_pred)
                lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
            
            models_results['Linear Regression'] = {
                'model': lr_model, 
                'r2': lr_r2, 
                'rmse': lr_rmse,
                'predictions': lr_pred,
                'uses_scaling': True
            }
            st.success(f"Linear Regression trained - R²: {lr_r2:.3f}, RMSE: {lr_rmse:.3f}")
            
        except Exception as e:
            st.error(f"Linear Regression training failed: {str(e)}")
            models_results['Linear Regression'] = {'model': None, 'r2': -1, 'rmse': float('inf')}
        
        # Train XGBoost
        try:
            st.write("Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='rmse',
                verbosity=0,
                tree_method='hist'
            )
            
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            xgb_pred = xgb_model.predict(X_test)
            
            if not np.isfinite(xgb_pred).all():
                st.warning("XGBoost produced invalid predictions")
                xgb_r2, xgb_rmse = -1, float('inf')
            else:
                xgb_r2 = r2_score(y_test, xgb_pred)
                xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            
            models_results['XGBoost'] = {
                'model': xgb_model, 
                'r2': xgb_r2, 
                'rmse': xgb_rmse,
                'predictions': xgb_pred
            }
            st.success(f"XGBoost trained - R²: {xgb_r2:.3f}, RMSE: {xgb_rmse:.3f}")
            
        except Exception as e:
            st.error(f"XGBoost training failed: {str(e)}")
            models_results['XGBoost'] = {'model': None, 'r2': -1, 'rmse': float('inf')}
        
        # Select best model
        valid_models = {name: results for name, results in models_results.items() 
                       if results['model'] is not None and results['r2'] > -1}
        
        if not valid_models:
            st.error("All models failed to train successfully")
            return None
        
        best_name = max(valid_models, key=lambda name: (valid_models[name]['r2'], -valid_models[name]['rmse']))
        best_results = valid_models[best_name]
        
        # Feature importance
        feature_importance = {}
        if best_name in ['Random Forest', 'XGBoost'] and best_results['model'] is not None:
            try:
                feature_importance = dict(zip(feature_cols, best_results['model'].feature_importances_))
            except Exception as e:
                st.warning(f"Could not extract feature importance: {str(e)}")
        
        # Compile results
        models = {
            'rf_model': models_results.get('Random Forest', {}).get('model'),
            'lr_model': models_results.get('Linear Regression', {}).get('model'),
            'xgb_model': models_results.get('XGBoost', {}).get('model'),
            'best_model': best_results['model'],
            'best_model_name': best_name,
            'best_r2': best_results['r2'],
            'best_rmse': best_results['rmse'],
            'rf_r2': models_results.get('Random Forest', {}).get('r2', -1),
            'lr_r2': models_results.get('Linear Regression', {}).get('r2', -1),
            'xgb_r2': models_results.get('XGBoost', {}).get('r2', -1),
            'rf_rmse': models_results.get('Random Forest', {}).get('rmse', float('inf')),
            'lr_rmse': models_results.get('Linear Regression', {}).get('rmse', float('inf')),
            'xgb_rmse': models_results.get('XGBoost', {}).get('rmse', float('inf')),
            'feature_cols': feature_cols,
            'le_dict': le_dict,
            'feature_importance': feature_importance,
            'scaler': scaler,
            'models_results': models_results
        }
        
        # Save models
        try:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(models, f)
            st.success(f"Models saved successfully. Best model: {best_name}")
        except Exception as e:
            st.warning(f"Could not save models: {str(e)}")
        
        return models
        
    except Exception as e:
        st.error(f"Critical error in model training: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def get_company_insights(df):
    """Generate business insights from company data"""
    insights = {}
    
    insights['total_campaigns'] = len(df)
    insights['total_investment'] = df['Cost'].sum() if 'Cost' in df.columns else 0
    insights['total_revenue'] = df['Revenue'].sum() if 'Revenue' in df.columns else 0
    insights['total_profit'] = insights['total_revenue'] - insights['total_investment']
    insights['avg_roi'] = df['ROI'].mean() if 'ROI' in df.columns else 0
    insights['success_rate'] = (len(df[df['ROI'] > 1]) / len(df)) * 100 if 'ROI' in df.columns else 0
    
    if 'Channel' in df.columns:
        channel_perf = df.groupby('Channel').agg({
            'ROI': ['mean', 'count'],
            'Cost': 'sum',
            'Revenue': 'sum' if 'Revenue' in df.columns else 'count'
        }).round(2)
        channel_perf.columns = ['_'.join(col).strip() for col in channel_perf.columns]
        
        best_channel = channel_perf['ROI_mean'].idxmax()
        insights['best_channel'] = best_channel
        insights['best_channel_roi'] = channel_perf.loc[best_channel, 'ROI_mean']
        insights['best_channel_spend'] = channel_perf.loc[best_channel, 'Cost_sum']
        
        worst_channel = channel_perf['ROI_mean'].idxmin()
        insights['worst_channel'] = worst_channel
        insights['worst_channel_roi'] = channel_perf.loc[worst_channel, 'ROI_mean']
    
    if 'Audience' in df.columns:
        audience_perf = df.groupby('Audience')['ROI'].agg(['mean', 'count']).round(2)
        best_audience = audience_perf['mean'].idxmax()
        insights['best_audience'] = best_audience
        insights['best_audience_roi'] = audience_perf.loc[best_audience, 'mean']
    
    if 'ROI_Category' in df.columns:
        insights['high_performers'] = len(df[df['ROI_Category'] == 'High_Profit'])
        insights['profitable_campaigns'] = len(df[df['ROI_Category'].isin(['Profitable', 'High_Profit'])])
    else:
        insights['high_performers'] = len(df[df['ROI'] > 2])
        insights['profitable_campaigns'] = len(df[df['ROI'] > 1])
    
    if 'Channel' in df.columns and 'Audience' in df.columns:
        combo_perf = df.groupby(['Channel', 'Audience'])['ROI'].mean().reset_index()
        best_combo = combo_perf.loc[combo_perf['ROI'].idxmax()]
        insights['best_combo'] = f"{best_combo['Channel']} + {best_combo['Audience']}"
        insights['best_combo_roi'] = best_combo['ROI']
    
    return insights

def predict_campaign_performance(models, df, channel, audience, budget, impressions, ctr, cvr, cpa):
    """Fixed prediction function with better error handling"""
    try:
        if models is None or models['best_model'] is None:
            st.error("No trained model available for predictions")
            return None
        
        # Calculate derived metrics
        clicks = max(1, (impressions * ctr) / 100)
        conversions = max(1, (clicks * cvr) / 100)
        revenue = conversions * cpa * 3
        profit = revenue - budget
        profit_margin = (profit / revenue) * 100 if revenue > 0 else 0
        roas = revenue / budget if budget > 0 else 0
        
        # Build feature vector
        feature_vector = {}
        
        # Handle categorical encodings
        for col, le in models['le_dict'].items():
            try:
                if col == 'Channel':
                    feature_vector[f'{col}_Encoded'] = le.transform([channel])[0] if channel in le.classes_ else 0
                elif col == 'Audience':
                    feature_vector[f'{col}_Encoded'] = le.transform([audience])[0] if audience in le.classes_ else 0
                else:
                    feature_vector[f'{col}_Encoded'] = 0
            except Exception as e:
                st.warning(f"Error encoding {col}: {str(e)}")
                feature_vector[f'{col}_Encoded'] = 0
        
        # Add numerical features
        feature_vector.update({
            'Cost': budget,
            'Revenue': revenue,
            'Impressions': impressions,
            'Clicks': clicks,
            'Conversions': conversions,
            'CPA': cpa,
            'CTR': ctr,
            'CVR': cvr,
            'Profit': profit,
            'Profit_Margin': profit_margin,
            'ROAS': roas
        })
        
        # Build prediction vector
        pred_vector = []
        for col in models['feature_cols']:
            value = feature_vector.get(col, 0)
            if not np.isfinite(value):
                value = 0
            pred_vector.append(value)
        
        pred_vector = np.array(pred_vector, dtype=float).reshape(1, -1)
        
        # Apply scaling if needed
        if models['best_model_name'] == 'Linear Regression':
            pred_vector = models['scaler'].transform(pred_vector)
        
        # Make prediction
        predicted_roi = models['best_model'].predict(pred_vector)[0]
        
        if not np.isfinite(predicted_roi):
            st.warning("Model produced invalid prediction, using conservative estimate")
            predicted_roi = 1.0
        
        # Generate recommendations
        recommendations = []
        
        # Channel recommendation
        if 'Channel' in df.columns:
            try:
                channel_perf = df.groupby('Channel')['ROI'].mean().sort_values(ascending=False)
                if len(channel_perf) > 0 and channel != channel_perf.index[0]:
                    top_channel = channel_perf.index[0]
                    top_channel_roi = channel_perf.iloc[0]
                    current_channel_roi = channel_perf.get(channel, df['ROI'].median())
                    channel_audience_perf = df[df['Channel'] == channel].groupby('Audience')['ROI'].mean().sort_values(ascending=False)
                    if len(channel_audience_perf) > 1:
                        top_audience = channel_audience_perf.index[0]
                        top_audience_roi = channel_audience_perf.iloc[0]
                        current_audience_roi = channel_audience_perf.get(audience, df['ROI'].median())
                        if top_audience != audience and top_audience_roi > current_audience_roi + 0.1:
                            recommendations.append(
                                f"🎯 **Audience Targeting**: For '{channel}', target '{top_audience}' (avg ROI: {top_audience_roi:.2f}) "
                                f"instead of '{audience}' (avg ROI: {current_audience_roi:.2f}) to maximize impact."
                            )
                    else:
                        recommendations.append(f"ℹ️ Insufficient audience data for '{channel}' to make a recommendation.")
            except Exception as e:
                st.warning(f"Could not generate audience recommendation: {str(e)}")
        
        # Budget recommendation
        try:
            similar_campaigns = df[(df['Channel'] == channel) & (df['Audience'] == audience)]
            if len(similar_campaigns) > 0:
                median_budget = similar_campaigns['Cost'].median()
                avg_roi_similar = similar_campaigns['ROI'].mean()
                if budget < median_budget * 0.8:
                    recommendations.append(
                        f"💰 **Budget Adjustment**: Your budget (${budget:,.0f}) is below the median for similar campaigns "
                        f"(${median_budget:,.0f}). Increasing budget could improve ROI (avg: {avg_roi_similar:.2f})."
                    )
                elif budget > median_budget * 1.2:
                    recommendations.append(
                        f"⚖️ **Budget Efficiency**: Your budget (${budget:,.0f}) is above the median for similar campaigns "
                        f"(${median_budget:,.0f}). Consider optimizing spend to maintain ROI (avg: {avg_roi_similar:.2f})."
                    )
        except Exception as e:
            st.warning(f"Could not generate budget recommendation: {str(e)}")
        
        # Performance metrics recommendations
        try:
            hist_ctr = df['CTR'].mean() if 'CTR' in df.columns else 2.0
            hist_cvr = df['CVR'].mean() if 'CVR' in df.columns else 5.0
            hist_cpa = df['CPA'].mean() if 'CPA' in df.columns else 75.0
            top_campaigns = df.nlargest(5, 'ROI')
            
            if ctr < hist_ctr * 0.9:
                top_ctr = top_campaigns['CTR'].mean() if 'CTR' in top_campaigns.columns else hist_ctr
                recommendations.append(
                    f"📈 **CTR Optimization**: Your expected CTR ({ctr:.1f}%) is below historical average ({hist_ctr:.1f}%). "
                    f"Top campaigns achieve {top_ctr:.1f}%. Improve ad creatives or targeting."
                )
            
            if cvr < hist_cvr * 0.9:
                top_cvr = top_campaigns['CVR'].mean() if 'CVR' in top_campaigns.columns else hist_cvr
                recommendations.append(
                    f"🔄 **CVR Optimization**: Your expected CVR ({cvr:.1f}%) is below historical average ({hist_cvr:.1f}%). "
                    f"Top campaigns achieve {top_cvr:.1f}%. Optimize landing pages or offers."
                )
            
            if cpa > hist_cpa * 1.1:
                top_cpa = top_campaigns['CPA'].mean() if 'CPA' in top_campaigns.columns else hist_cpa
                recommendations.append(
                    f"💸 **CPA Optimization**: Your target CPA (${cpa:.0f}) is above historical average (${hist_cpa:.0f}). "
                    f"Top campaigns achieve ${top_cpa:.0f}. Refine conversion funnel or targeting."
                )
        except Exception as e:
            st.warning(f"Could not generate performance recommendations: {str(e)}")
        
        # Feature importance-based recommendation
        try:
            if models['best_model_name'] in ['Random Forest', 'XGBoost'] and models['feature_importance']:
                top_feature = max(models['feature_importance'], key=models['feature_importance'].get)
                top_importance = models['feature_importance'][top_feature]
                recommendations.append(
                    f"🧠 **Key Driver**: '{top_feature}' is the most influential factor for ROI "
                    f"(importance: {top_importance:.2f}). Focus optimization efforts here."
                )
        except Exception as e:
            st.warning(f"Could not generate feature importance recommendation: {str(e)}")
        
        # ROI-based recommendation
        if predicted_roi < 1.5:
            recommendations.append(
                f"⚠️ **Low ROI Warning**: Predicted ROI ({predicted_roi:.2f}) is below optimal threshold (1.5). "
                f"Review channel, audience, or performance metrics to improve returns."
            )
        elif predicted_roi > 2.5:
            recommendations.append(
                f"🚀 **High Potential**: Predicted ROI ({predicted_roi:.2f}) is excellent! "
                f"Consider scaling budget or replicating this setup for other campaigns."
            )
        
        return {
            'predicted_roi': predicted_roi,
            'expected_revenue': revenue,
            'expected_profit': profit,
            'expected_clicks': clicks,
            'expected_conversions': conversions,
            'expected_roas': roas,
            'recommendations': recommendations
        }
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def generate_ai_analysis(predicted_data, actual_data):
    """Generate AI-powered analysis comparing predicted vs actual performance"""
    roi_diff = actual_data['roi'] - predicted_data['predicted_roi']
    roi_accuracy = 100 - (abs(roi_diff) / max(abs(predicted_data['predicted_roi']), 0.1) * 100)
    
    revenue_diff = actual_data['revenue'] - predicted_data['expected_revenue']
    revenue_accuracy = 100 - (abs(revenue_diff) / max(abs(predicted_data['expected_revenue']), 1) * 100)
    
    clicks_diff = actual_data['clicks'] - predicted_data['expected_clicks']
    conversions_diff = actual_data['conversions'] - predicted_data['expected_conversions']
    
    if roi_accuracy >= 85 and actual_data['roi'] >= predicted_data['predicted_roi']:
        performance_category = "excellent"
        performance_message = "🎯 Outstanding Performance! Exceeded expectations with high accuracy."
    elif roi_accuracy >= 70:
        performance_category = "good"
        performance_message = "👍 Good Performance! Results aligned well with predictions."
    else:
        performance_category = "needs_improvement"
        performance_message = "📊 Learning Opportunity! Significant variance detected for model improvement."
    
    insights = []
    if roi_diff > 0.5:
        insights.append(f"🚀 ROI exceeded predictions by {roi_diff:.2f} - excellent execution or market conditions!")
    elif roi_diff < -0.5:
        insights.append(f"⚠️ ROI fell short by {abs(roi_diff):.2f} - consider analyzing execution gaps or market factors.")
    else:
        insights.append(f"✅ ROI closely matched predictions (difference: {roi_diff:+.2f})")
    
    if revenue_diff > predicted_data['expected_revenue'] * 0.1:
        insights.append(f"💰 Revenue exceeded expectations by ${revenue_diff:,.0f} - strong market response!")
    elif revenue_diff < -predicted_data['expected_revenue'] * 0.1:
        insights.append(f"📉 Revenue below expectations by ${abs(revenue_diff):,.0f} - review targeting or messaging.")
    
    predicted_ctr = predicted_data['expected_clicks'] / actual_data['impressions'] * 100
    ctr_diff = actual_data['ctr'] - predicted_ctr
    
    if abs(ctr_diff) > 0.5:
        if ctr_diff > 0:
            insights.append(f"🎯 CTR outperformed by {ctr_diff:.1f}% - excellent creative or targeting!")
        else:
            insights.append(f"📊 CTR underperformed by {abs(ctr_diff):.1f}% - consider creative optimization.")
    
    if conversions_diff > 0:
        insights.append(f"🔄 Generated {conversions_diff:.0f} more conversions than expected - strong funnel performance!")
    elif conversions_diff < 0:
        insights.append(f"⚡ {abs(conversions_diff):.0f} fewer conversions - optimize landing page or offer.")
    
    recommendations = []
    if actual_data['roi'] > 2.0:
        recommendations.append("🚀 Scale Up: High ROI indicates opportunity for budget increase")
    elif actual_data['roi'] > 1.0:
        recommendations.append("📈 Optimize: Good foundation, test improvements in underperforming areas")
    else:
        recommendations.append("🔧 Restructure: Below break-even, fundamental changes needed")
    
    if roi_accuracy < 70:
        recommendations.append("🤖 Model Learning: Add this data to improve future prediction accuracy")
    
    if actual_data['ctr'] > predicted_ctr + 0.5:
        recommendations.append("🎯 Replicate Success: Apply successful creative/targeting to similar campaigns")
    
    return {
        'performance_category': performance_category,
        'performance_message': performance_message,
        'roi_accuracy': roi_accuracy,
        'revenue_accuracy': revenue_accuracy,
        'insights': insights,
        'recommendations': recommendations,
        'metrics': {
            'roi_diff': roi_diff,
            'revenue_diff': revenue_diff,
            'clicks_diff': clicks_diff,
            'conversions_diff': conversions_diff
        }
    }

def find_similar_campaigns(df, channel, audience, budget, top_n=3):
    """Find similar campaigns from historical data for comparison"""
    similar = df[(df['Channel'] == channel) & (df['Audience'] == audience)].copy()
    
    if len(similar) == 0:
        similar = df[df['Channel'] == channel].copy()
    
    if len(similar) == 0:
        similar = df[df['Audience'] == audience].copy()
    
    if len(similar) == 0:
        return None
    
    similar['budget_similarity'] = 1 / (1 + abs(similar['Cost'] - budget) / budget)
    similar = similar.sort_values(['budget_similarity', 'ROI'], ascending=[False, False])
    
    return similar.head(top_n)

def add_new_campaign(df, new_campaign_data):
    """Add new campaign data and retrain models"""
    try:
        new_df = pd.DataFrame([new_campaign_data])
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df = process_campaign_data(updated_df)
        updated_df.to_csv(PROCESSED_DATA_FILE, index=False)
        st.cache_data.clear()
        return updated_df
    except Exception as e:
        st.error(f"Error adding campaign: {str(e)}")
        return df

# Initialize session state
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'company_models' not in st.session_state:
    st.session_state.company_models = None
if 'company_insights' not in st.session_state:
    st.session_state.company_insights = None
if 'predicted_campaigns' not in st.session_state:
    st.session_state.predicted_campaigns = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Header
st.markdown('<h1 class="main-header">🎯 Company Campaign Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="company-tagline">Your AI-Powered Marketing Decision Engine</p>', unsafe_allow_html=True)

# Load company data and models
with st.spinner('🔄 Loading your company data and training AI models...'):
    if st.session_state.company_data is None:
        st.write("Attempting to load company data...")
        df = load_company_data()
        if df is not None:
            st.write("Data loaded successfully, assigning to session state...")
            st.session_state.company_data = df
            st.write("Training models...")
            models = train_company_models(df)
            if models is not None:
                st.session_state.company_models = models
                st.write("Models trained successfully, generating insights...")
                insights = get_company_insights(df)
                st.session_state.company_insights = insights
                st.write("Insights generated, data ready!")
            else:
                st.write("Model training failed!")
        else:
            st.write("Data loading failed!")

# Main application
if st.session_state.company_data is not None:
    df = st.session_state.company_data
    models = st.session_state.company_models
    insights = st.session_state.company_insights
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🔧 Debug Mode", help="Show detailed model training information")
    
    # Sidebar - System Status
    with st.sidebar:
        st.markdown("### 🎛️ System Status")
        st.markdown(f"""
        - **📊 Data:** {insights['total_campaigns']} campaigns loaded
        - **🤖 AI Model:** {models['best_model_name']} (R²: {models['best_r2']:.1%}, RMSE: {models['best_rmse']:.2f})
        - **💰 Total Investment:** ${insights['total_investment']:,.0f}
        - **📈 Success Rate:** {insights['success_rate']:.0f}%
        """)
        
        if debug_mode and models:
            st.markdown("### 🔍 Debug Information")
            st.markdown(f"""
            <div class="debug-card">
            <strong>Feature Columns:</strong><br>
            {', '.join(models['feature_cols'][:5])}{'...' if len(models['feature_cols']) > 5 else ''}<br><br>
            
            <strong>Model Performance:</strong><br>
            RF R²: {models['rf_r2']:.3f} | RMSE: {models['rf_rmse']:.3f}<br>
            LR R²: {models['lr_r2']:.3f} | RMSE: {models['lr_rmse']:.3f}<br>
            XGB R²: {models['xgb_r2']:.3f} | RMSE: {models['xgb_rmse']:.3f}<br><br>
            
            <strong>Data Quality:</strong><br>
            Shape: {df.shape}<br>
            Missing: {df.isnull().sum().sum()}<br>
            ROI Range: {df['ROI'].min():.2f} to {df['ROI'].max():.2f}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🔄 Data Management")
        if st.button("📊 Export Current Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"company_campaigns_{dt.datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    # Quick Insights Dashboard
    st.markdown("### 📊 Your Campaign Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{insights['total_campaigns']}</h3>
            <p>Total Campaigns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${insights['total_profit']:,.0f}</h3>
            <p>Total Profit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{insights['avg_roi']:.2f}</h3>
            <p>Average ROI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{insights['high_performers']}</h3>
            <p>High Performers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Smart Campaign Builder", 
        "📈 Performance Analytics", 
        "🎯 Campaign Intelligence",
        "➕ Campaign Data & Comparison"
    ])
    
    with tab1:
        st.markdown("## 🚀 Build Your Next Winning Campaign")
        st.markdown("Get AI predictions based on your company's historical performance data!")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("### Campaign Configuration")
            
            campaign_name = st.text_input("Campaign Name", value=f"Campaign_{dt.datetime.now().strftime('%Y%m%d')}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                channels = df['Channel'].unique().tolist()
                selected_channel = st.selectbox("Channel", options=channels)
                channel_roi = df[df['Channel'] == selected_channel]['ROI'].mean()
                st.markdown(f'<div class="insight-highlight">📊 Historical ROI: {channel_roi:.2f}</div>', unsafe_allow_html=True)
            
            with col_b:
                audiences = df['Audience'].unique().tolist()
                selected_audience = st.selectbox("Target Audience", options=audiences)
                audience_roi = df[df['Audience'] == selected_audience]['ROI'].mean()
                st.markdown(f'<div class="insight-highlight">👥 Historical ROI: {audience_roi:.2f}</div>', unsafe_allow_html=True)
            
            col_c, col_d = st.columns(2)
            with col_c:
                similar_campaigns = df[(df['Channel'] == selected_channel) & (df['Audience'] == selected_audience)]
                suggested_budget = int(similar_campaigns['Cost'].median()) if len(similar_campaigns) > 0 else 10000
                budget = st.number_input(
                    "Campaign Budget ($)", 
                    min_value=1000, 
                    max_value=500000, 
                    value=suggested_budget, 
                    step=1000,
                    help=f"Similar campaigns averaged ${suggested_budget:,}"
                )
                impressions = st.number_input(
                    "Expected Impressions", 
                    min_value=10000, 
                    value=100000, 
                    step=10000
                )
            
            with col_d:
                hist_ctr = df['CTR'].mean() if 'CTR' in df.columns else 2.0
                hist_cvr = df['CVR'].mean() if 'CVR' in df.columns else 5.0
                hist_cpa = df['CPA'].mean() if 'CPA' in df.columns else 75.0
                expected_ctr = st.slider(
                    "Expected CTR (%)", 
                    0.5, 10.0, 
                    float(hist_ctr), 0.1,
                    help=f"Your average: {hist_ctr:.1f}%"
                )
                expected_cvr = st.slider(
                    "Expected CVR (%)", 
                    1.0, 25.0, 
                    float(hist_cvr), 0.1,
                    help=f"Your average: {hist_cvr:.1f}%"
                )
                target_cpa = st.number_input(
                    "Target CPA ($)", 
                    min_value=10.0, 
                    value=float(hist_cpa), 
                    step=5.0,
                    help=f"Your average: ${hist_cpa:.0f}"
                )
        
        with col_right:
            st.markdown("### 🎯 AI Performance Prediction")
            
            if models:
                prediction = predict_campaign_performance(
                    models, df, selected_channel, selected_audience, 
                    budget, impressions, expected_ctr, expected_cvr, target_cpa
                )
                
                if prediction:
                    predicted_roi = prediction['predicted_roi']
                    
                    # Store prediction for later use in tab4
                    st.session_state.last_prediction = {
                        'campaign_name': campaign_name,
                        'channel': selected_channel,
                        'audience': selected_audience,
                        'budget': budget,
                        'impressions': impressions,
                        'expected_ctr': expected_ctr,
                        'expected_cvr': expected_cvr,
                        'target_cpa': target_cpa,
                        **prediction
                    }
                    
                    if predicted_roi > 2.5:
                        st.markdown(f"""
                            <div class="prediction-excellent">
                            <h2>🚀 Predicted ROI: {predicted_roi:.2f}</h2>
                            <h4>Exceptional Performance Expected!</h4>
                            <p>This campaign configuration shows outstanding potential</p>
                            </div>
                            """, unsafe_allow_html=True)
                    elif predicted_roi > 1.5:
                        st.markdown(f"""
                        <div class="prediction-good">
                            <h2>📈 Predicted ROI: {predicted_roi:.2f}</h2>
                            <h4>Strong Performance Expected</h4>
                            <p>Good campaign setup with positive returns</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-poor">
                            <h2>⚠️ Predicted ROI: {predicted_roi:.2f}</h2>
                            <h4>Optimization Recommended</h4>
                            <p>Consider adjusting parameters for better performance</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("#### 📊 Expected Performance Metrics")
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Expected Revenue", f"${prediction['expected_revenue']:,.0f}")
                        st.metric("Expected Clicks", f"{prediction['expected_clicks']:,.0f}")
                        st.metric("Expected ROAS", f"{prediction['expected_roas']:.1f}x")
                    with metric_col2:
                        st.metric("Expected Profit", f"${prediction['expected_profit']:,.0f}")
                        st.metric("Expected Conversions", f"{prediction['expected_conversions']:.0f}")
                        st.metric("Break-even CPA", f"${prediction['expected_revenue']/prediction['expected_conversions']:.0f}" if prediction['expected_conversions'] > 0 else "N/A")

                    # Show similar campaigns
                    similar_campaigns = find_similar_campaigns(df, selected_channel, selected_audience, budget)
                    if similar_campaigns is not None and len(similar_campaigns) > 0:
                        st.markdown("#### 📊 Similar Historical Campaigns")
                        for idx, row in similar_campaigns.iterrows():
                            st.markdown(f"""
                            - **{row['Campaign_Name']}**: ROI {row['ROI']:.2f}, Cost ${row['Cost']:,.0f}, Revenue ${row['Revenue']:,.0f}
                            """)

                    # Enhanced recommendations
                    st.markdown("#### 🧠 AI-Powered Recommendations")
                    with st.expander("View Detailed Optimization Suggestions"):
                        if prediction['recommendations']:
                            for rec in prediction['recommendations']:
                                st.markdown(f'<div class="insight-highlight">{rec}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-highlight">✅ Current configuration aligns well with historical data!</div>', unsafe_allow_html=True)
                    
                    # Original recommendations for backward compatibility
                    if selected_channel == insights['best_channel']:
                        st.markdown('<div class="insight-highlight">✅ Excellent choice! This is your top-performing channel</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="insight-highlight">💡 Consider {insights["best_channel"]} (your best channel, ROI: {insights["best_channel_roi"]:.2f})</div>', unsafe_allow_html=True)

                    if selected_audience == insights['best_audience']:
                        st.markdown('<div class="insight-highlight">✅ Great targeting! This is your best audience</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="insight-highlight">🎯 Also consider {insights["best_audience"]} audience (ROI: {insights["best_audience_roi"]:.2f})</div>', unsafe_allow_html=True)

                    if st.button("💾 Save Prediction", type="primary"):
                        campaign_id = f"{campaign_name}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
                        predicted_campaign = {
                            'campaign_id': campaign_id,
                            'created_date': dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
                            **st.session_state.last_prediction
                        }
                        st.session_state.predicted_campaigns.append(predicted_campaign)
                        st.markdown('<div class="success-banner">🚀 Campaign prediction saved!</div>', unsafe_allow_html=True)
                        st.rerun()
                else:
                    st.error("❌ Unable to generate prediction. Please check your model configuration.")
            else:
                st.error("❌ No AI models available for prediction. Please check your data and model training.")

    with tab2:
        st.markdown("## 📈 Your Company's Performance Analytics")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig = px.histogram(
                df, x='ROI', nbins=30, 
                title="ROI Distribution - Your Campaigns",
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            if 'Channel' in df.columns:
                channel_perf = df.groupby('Channel').agg({
                    'ROI': 'mean',
                    'Cost': 'sum',
                    'Revenue': 'sum'
                }).reset_index()
                
                fig = px.bar(
                    channel_perf, x='Channel', y='ROI', 
                    title="Average ROI by Channel",
                    color='ROI', 
                    color_continuous_scale='viridis',
                    text='ROI'
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏆 Performance Highlights")
        
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        
        with col_perf1:
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>🥇 Top Channel</h3>
                <h2>{insights['best_channel']}</h2>
                <p>ROI: {insights['best_channel_roi']:.2f}</p>
                <p>Investment: ${insights['best_channel_spend']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_perf2:
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>🎯 Best Audience</h3>
                <h2>{insights['best_audience']}</h2>
                <p>ROI: {insights['best_audience_roi']:.2f}</p>
                <p>Winning combination for your brand</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_perf3:
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>🚀 Success Rate</h3>
                <h2>{insights['success_rate']:.0f}%</h2>
                <p>{insights['profitable_campaigns']} profitable campaigns</p>
                <p>out of {insights['total_campaigns']} total</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Top 10 Performing Campaigns")
        top_campaigns = df.nlargest(10, 'ROI')
        display_cols = ['Campaign_Name', 'Channel', 'Audience', 'Cost', 'Revenue', 'ROI']
        display_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(top_campaigns[display_cols], use_container_width=True)

    with tab3:
        st.markdown("## 🎯 Campaign Intelligence & Insights")
        
        col_model1, col_model2 = st.columns(2)
        
        with col_model1:
            st.markdown(f"""
            <div class="new-campaign-card">
                <h3>🤖 AI Model Performance</h3>
                <p><strong>Primary Model:</strong> {models['best_model_name']}</p>
                <p><strong>R² Score:</strong> {models['best_r2']:.1%}</p>
                <p><strong>RMSE:</strong> {models['best_rmse']:.2f}</p>
                <p><strong>Training Data:</strong> {insights['total_campaigns']} campaigns</p>
                <p><strong>Confidence Level:</strong> {"High" if models['best_r2'] > 0.8 else "Good" if models['best_r2'] > 0.6 else "Moderate"}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Model Comparison")
            model_comparison = pd.DataFrame({
                'Model': ['Random Forest', 'Linear Regression', 'XGBoost'],
                'R² Score': [models['rf_r2'], models['lr_r2'], models['xgb_r2']],
                'RMSE': [models['rf_rmse'], models['lr_rmse'], models['xgb_rmse']]
            })
            model_comparison['R² Score'] = model_comparison['R² Score'].apply(lambda x: f"{x:.1%}")
            model_comparison['RMSE'] = model_comparison['RMSE'].apply(lambda x: f"{x:.2f}")
            st.dataframe(model_comparison, use_container_width=True)
        
        with col_model2:
            if models['feature_importance']:
                feature_imp_df = pd.DataFrame(
                    list(models['feature_importance'].items()), 
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(8)
                
                fig = px.bar(
                    feature_imp_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    title="Most Important Factors for ROI Prediction",
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            fig = px.bar(
                model_comparison,
                x='Model',
                y=['R² Score', 'RMSE'],
                title="Model Performance Comparison",
                barmode='group',
                color_discrete_sequence=['#636EFA', '#EF553B']
            )
            fig.update_traces(
                texttemplate='%{y:.2f}',
                textposition='outside',
                selector=dict(name='RMSE')
            )
            fig.update_traces(
                texttemplate='%{y:.1%}',
                textposition='outside',
                selector=dict(name='R² Score')
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("## ➕ Campaign Data & AI-Powered Comparison")
        st.markdown("Compare AI predictions with actual results and add new campaign data to improve model accuracy")
        
        if st.session_state.last_prediction:
            st.markdown("### 🔮 Latest AI Prediction Available for Comparison")
            
            prediction_data = st.session_state.last_prediction
            
            col_pred_display, col_pred_metrics = st.columns([2, 1])
            
            with col_pred_display:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>🎯 Predicted Campaign: {prediction_data['campaign_name']}</h3>
                    <p><strong>Channel:</strong> {prediction_data['channel']} | <strong>Audience:</strong> {prediction_data['audience']}</p>
                    <p><strong>Budget:</strong> ${prediction_data['budget']:,.2f} | <strong>Impressions:</strong> {prediction_data['impressions']:,}</p>
                    <h4>📊 AI Predictions:</h4>
                    <p>• ROI: <strong>{prediction_data['predicted_roi']:.2f}</strong></p>
                    <p>• Revenue: <strong>${prediction_data['expected_revenue']:,.2f}</strong></p>
                    <p>• Clicks: <strong>{prediction_data['expected_clicks']:,.0f}</strong></p>
                    <p>• Conversions: <strong>{prediction_data['expected_conversions']:.0f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pred_metrics:
                st.markdown("### 🎯 Target Metrics")
                st.metric("Target ROI", f"{prediction_data['predicted_roi']:.2f}")
                st.metric("Target Revenue", f"${prediction_data['expected_revenue']:,.0f}")
                st.metric("Target Profit", f"${prediction_data['expected_profit']:,.0f}")
                st.metric("Expected CTR", f"{prediction_data['expected_ctr']:.1f}%")
                st.metric("Expected CVR", f"{prediction_data['expected_cvr']:.1f}%")
        
        else:
            st.info("💡 Create a campaign prediction in the 'Smart Campaign Builder' tab first to enable comparison features!")
        
        col_form, col_analysis = st.columns([3, 2])
        
        with col_form:
            st.markdown("### 📝 Enter Actual Campaign Results")
            
            with st.form("campaign_comparison_form"):
                st.markdown("#### Basic Campaign Information")
                col_basic1, col_basic2 = st.columns(2)
                
                with col_basic1:
                    if st.session_state.last_prediction:
                        default_name = st.session_state.last_prediction['campaign_name']
                        default_channel = st.session_state.last_prediction['channel']
                    else:
                        default_name = f"Campaign_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
                        default_channel = df['Channel'].unique()[0]
                    
                    new_campaign_name = st.text_input("Campaign Name*", value=default_name)
                    new_channel = st.selectbox("Channel*", options=df['Channel'].unique().tolist(), 
                                             index=df['Channel'].unique().tolist().index(default_channel) if default_channel in df['Channel'].unique() else 0)
                
                with col_basic2:
                    if st.session_state.last_prediction:
                        default_audience = st.session_state.last_prediction['audience']
                    else:
                        default_audience = df['Audience'].unique()[0]
                    
                    new_audience = st.selectbox("Audience*", options=df['Audience'].unique().tolist(),
                                              index=df['Audience'].unique().tolist().index(default_audience) if default_audience in df['Audience'].unique() else 0)
                    start_date = st.date_input("Campaign Start Date*", value=dt.datetime.now().date())
                    end_date = st.date_input("Campaign End Date*", value=dt.datetime.now().date() + dt.timedelta(days=30))
                
                # Validate dates
                if end_date < start_date:
                    st.error("End date cannot be before start date!")
                
                st.markdown("#### 💰 Financial Results")
                col_finance1, col_finance2 = st.columns(2)
                
                with col_finance1:
                    if st.session_state.last_prediction:
                        default_cost = st.session_state.last_prediction['budget']
                        default_revenue = st.session_state.last_prediction['expected_revenue']
                    else:
                        default_cost = 5000.0
                        default_revenue = 7500.0
                    
                    actual_cost = st.number_input("Actual Cost ($)*", min_value=0.0, value=float(default_cost), step=100.0)
                    actual_revenue = st.number_input("Actual Revenue ($)*", min_value=0.0, value=float(default_revenue), step=100.0)
                
                with col_finance2:
                    if st.session_state.last_prediction:
                        default_cpa = st.session_state.last_prediction['target_cpa']
                    else:
                        default_cpa = 50.0
                    
                    actual_cpa = st.number_input("Actual CPA ($)*", min_value=0.0, value=float(default_cpa), step=1.0)
                    calculated_roi = (actual_revenue - actual_cost) / actual_cost if actual_cost > 0 else 0
                    
                    if st.session_state.last_prediction:
                        target_roi = st.session_state.last_prediction['predicted_roi']
                        roi_diff = calculated_roi - target_roi
                        st.metric("Actual ROI", f"{calculated_roi:.2f}", f"{roi_diff:+.2f} vs target")
                    else:
                        st.metric("Calculated ROI", f"{calculated_roi:.2f}")
                
                st.markdown("#### 📊 Performance Metrics")
                col_perf1, col_perf2 = st.columns(2)
                
                with col_perf1:
                    if st.session_state.last_prediction:
                        default_impressions = st.session_state.last_prediction['impressions']
                        default_clicks = st.session_state.last_prediction['expected_clicks']
                    else:
                        default_impressions = 100000
                        default_clicks = 2000
                    
                    actual_impressions = st.number_input("Actual Impressions*", min_value=0, value=int(default_impressions), step=1000)
                    actual_clicks = st.number_input("Actual Clicks*", min_value=0, value=int(default_clicks), step=10)
                
                with col_perf2:
                    if st.session_state.last_prediction:
                        default_conversions = max(1, int(st.session_state.last_prediction['expected_conversions']))
                    else:
                        default_conversions = 100
                    
                    actual_conversions = st.number_input("Actual Conversions*", min_value=0, value=default_conversions, step=1)
                    
                    calculated_ctr = (actual_clicks / actual_impressions) * 100 if actual_impressions > 0 else 0
                    calculated_cvr = (actual_conversions / actual_clicks) * 100 if actual_clicks > 0 else 0
                    calculated_roas = actual_revenue / actual_cost if actual_cost > 0 else 0
                    
                    if st.session_state.last_prediction:
                        target_ctr = st.session_state.last_prediction['expected_ctr']
                        target_cvr = st.session_state.last_prediction['expected_cvr']
                        ctr_diff = calculated_ctr - target_ctr
                        cvr_diff = calculated_cvr - target_cvr
                        st.metric("Actual CTR", f"{calculated_ctr:.2f}%", f"{ctr_diff:+.1f}% vs target")
                        st.metric("Actual CVR", f"{calculated_cvr:.2f}%", f"{cvr_diff:+.1f}% vs target")
                    else:
                        st.metric("Calculated CTR", f"{calculated_ctr:.2f}%")
                        st.metric("Calculated CVR", f"{calculated_cvr:.2f}%")
                
                add_to_database = st.checkbox(
                    "Add this campaign to the training database", 
                    value=True,
                    help="This will improve future AI predictions"
                )
                
                retrain_models = st.checkbox(
                    "Retrain AI models after adding data",
                    value=True,
                    help="Recommended for improved accuracy"
                )
                
                submitted = st.form_submit_button("🔍 Analyze Performance & Add Data", type="primary")
                
                if submitted and new_campaign_name and actual_cost > 0 and end_date >= start_date:
                    actual_results = {
                        'campaign_name': new_campaign_name,
                        'channel': new_channel,
                        'audience': new_audience,
                        'cost': actual_cost,
                        'revenue': actual_revenue,
                        'impressions': actual_impressions,
                        'clicks': actual_clicks,
                        'conversions': actual_conversions,
                        'cpa': actual_cpa,
                        'roi': calculated_roi,
                        'ctr': calculated_ctr,
                        'cvr': calculated_cvr,
                        'roas': calculated_roas
                    }
                    
                    if add_to_database:
                        new_campaign_data = {
                            'Campaign_Name': new_campaign_name,
                            'Channel': new_channel,
                            'Audience': new_audience,
                            'Start_Date': start_date,
                            'End_Date': end_date,
                            'Cost': actual_cost,
                            'Revenue': actual_revenue,
                            'Impressions': actual_impressions,
                            'Clicks': actual_clicks,
                            'Conversions': actual_conversions,
                            'CPA': actual_cpa,
                            'CTR': calculated_ctr,
                            'CVR': calculated_cvr,
                            'ROI': calculated_roi,
                            'Profit': actual_revenue - actual_cost,
                            'Profit_Margin': ((actual_revenue - actual_cost) / actual_revenue) * 100 if actual_revenue > 0 else 0,
                            'ROAS': calculated_roas
                        }
                        
                        updated_df = add_new_campaign(df, new_campaign_data)
                        if updated_df is not None:
                            st.session_state.company_data = updated_df
                            st.success("✅ Campaign added to database successfully!")
                            
                            if retrain_models:
                                with st.spinner("🤖 Retraining AI models with new data..."):
                                    new_models = train_company_models(updated_df)
                                    if new_models:
                                        st.session_state.company_models = new_models
                                        st.session_state.company_insights = get_company_insights(updated_df)
                                        st.success(f"🎯 Models retrained! New R²: {new_models['best_r2']:.1%}, RMSE: {new_models['best_rmse']:.2f}")
                    
                    if st.session_state.last_prediction and (
                        new_channel == st.session_state.last_prediction['channel'] and 
                        new_audience == st.session_state.last_prediction['audience']
                    ):
                        ai_analysis = generate_ai_analysis(st.session_state.last_prediction, actual_results)
                        
                        st.markdown("---")
                        st.markdown("### 🤖 AI Performance Analysis")
                        
                        if ai_analysis['performance_category'] == 'excellent':
                            st.markdown(f"""
                            <div class="comparison-excellent">
                                <h3>🎯 {ai_analysis['performance_message']}</h3>
                                <p><strong>Prediction Accuracy:</strong> {ai_analysis['roi_accuracy']:.1f}%</p>
                                <p><strong>Revenue Accuracy:</strong> {ai_analysis['revenue_accuracy']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif ai_analysis['performance_category'] == 'good':
                            st.markdown(f"""
                            <div class="comparison-good">
                                <h3>👍 {ai_analysis['performance_message']}</h3>
                                <p><strong>Prediction Accuracy:</strong> {ai_analysis['roi_accuracy']:.1f}%</p>
                                <p><strong>Revenue Accuracy:</strong> {ai_analysis['revenue_accuracy']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="comparison-needs-improvement">
                                <h3>📊 {ai_analysis['performance_message']}</h3>
                                <p><strong>Prediction Accuracy:</strong> {ai_analysis['roi_accuracy']:.1f}%</p>
                                <p><strong>Revenue Accuracy:</strong> {ai_analysis['revenue_accuracy']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        col_insights, col_recs = st.columns(2)
                        
                        with col_insights:
                            st.markdown("#### 🔍 Performance Insights")
                            for insight in ai_analysis['insights']:
                                st.markdown(f"• {insight}")
                        
                        with col_recs:
                            st.markdown("#### 🚀 AI Recommendations")
                            for rec in ai_analysis['recommendations']:
                                st.markdown(f"• {rec}")
                        
                        comparison_data = {
                            'Metric': ['ROI', 'Revenue', 'Clicks', 'Conversions'],
                            'Predicted': [
                                st.session_state.last_prediction['predicted_roi'],
                                st.session_state.last_prediction['expected_revenue'],
                                st.session_state.last_prediction['expected_clicks'],
                                st.session_state.last_prediction['expected_conversions']
                            ],
                            'Actual': [
                                calculated_roi,
                                actual_revenue,
                                actual_clicks,
                                actual_conversions
                            ]
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        comparison_df['Accuracy'] = (
                            100 - abs(comparison_df['Actual'] - comparison_df['Predicted']) / 
                            comparison_df['Predicted'].abs().replace(0, 1) * 100
                        )
                        
                        fig = px.bar(
                            comparison_df, 
                            x='Metric', 
                            y=['Predicted', 'Actual'], 
                            title="Predicted vs Actual Performance",
                            barmode='group',
                            color_discrete_sequence=['#636EFA', '#EF553B']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.success("✅ Campaign data processed successfully!")
                        if not st.session_state.last_prediction:
                            st.info("💡 Create a prediction first to enable AI-powered comparison analysis!")
                        else:
                            st.info("ℹ️ Campaign details don't match the latest prediction. Comparison analysis is available for matching campaigns only.")
                elif submitted:
                    st.error("Please provide a valid campaign name and cost, and ensure end date is not before start date.")
        
        with col_analysis:
            st.markdown("### 📈 Campaign Enhancement Benefits")
            
            st.markdown(f"""
            <div class="ai-analysis-card">
                <h4>🎯 Current AI Status</h4>
                <p><strong>Model:</strong> {models['best_model_name']}</p>
                <p><strong>R² Score:</strong> {models['best_r2']:.1%}</p>
                <p><strong>RMSE:</strong> {models['best_rmse']:.2f}</p>
                <p><strong>Training Data:</strong> {insights['total_campaigns']} campaigns</p>
                <p><strong>Confidence:</strong> {"High" if models['best_r2'] > 0.8 else "Good" if models['best_r2'] > 0.6 else "Moderate"}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🚀 Why Compare & Add Data?")
            st.markdown("""
            **Improved AI Predictions:**
            - Learn from actual vs predicted performance
            - Adapt to market changes and trends
            - Increase prediction accuracy over time
            
            **Better Strategic Insights:**
            - Identify consistent over/under-performance patterns
            - Optimize budget allocation based on real data
            - Understand which factors drive success
            
            **Competitive Advantage:**
            - Data-driven campaign optimization
            - Reduced risk in future campaigns
            - Faster identification of high-ROI opportunities
            """)
            
            # Display saved predictions
            if st.session_state.predicted_campaigns:
                st.markdown("### 📜 Saved Predictions")
                saved_df = pd.DataFrame(st.session_state.predicted_campaigns)
                display_cols = ['campaign_name', 'channel', 'audience', 'budget', 'predicted_roi', 'expected_revenue']
                saved_df = saved_df[display_cols]
                saved_df.columns = ['Campaign Name', 'Channel', 'Audience', 'Budget ($)', 'Predicted ROI', 'Expected Revenue ($)']
                st.dataframe(saved_df, use_container_width=True)
                
                # Allow downloading saved predictions
                csv = saved_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Saved Predictions",
                    csv,
                    f"saved_predictions_{dt.datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

else:
    st.error("❌ No company data loaded. Please ensure the data file is available and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by xAI | Company Campaign Intelligence Platform | © 2025</p>
</div>
""", unsafe_allow_html=True)
