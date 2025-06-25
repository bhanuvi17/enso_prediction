import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import io
import base64

st.set_page_config(
    page_title="ENSO Prediction Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_and_model():
    """Load the trained model and data"""
    try:
        df_enso = pd.read_csv('models/enso_data.csv', index_col=0, parse_dates=True)
        
        model = load_model('models/lstm_enso_model.keras')
        X_scaler = joblib.load('models/X_scaler.pkl')
        y_scaler = joblib.load('models/y_scaler.pkl')
        model_info = joblib.load('models/model_info.pkl')
        
        return df_enso, model, X_scaler, y_scaler, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run train_model.py first to train the model.")
        return None, None, None, None, None

def classify_enso(oni_values):
    """Classify ONI values into ENSO categories"""
    if isinstance(oni_values, (int, float)):
        oni_values = np.array([oni_values])
    
    conditions = [
        oni_values >= 0.5,   # El Niño
        oni_values <= -0.5,  # La Niña
    ]
    choices = ['El Niño', 'La Niña']
    return np.select(conditions, choices, default='Neutral')

def get_enso_color(category):
    """Get color for ENSO category"""
    colors = {
        'El Niño': '#FF6B6B',
        'La Niña': '#4ECDC4', 
        'Neutral': '#45B7D1'
    }
    return colors.get(category, '#888888')

def create_oni_timeseries_plot(df, start_date, end_date):
    """Create ONI time series plot with ENSO events"""
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available for selected date range", 
                                         xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    filtered_df['ENSO_Category'] = classify_enso(filtered_df['ONI'].values)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['ONI'],
        mode='lines',
        name='ONI',
        line=dict(color='darkblue', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>ONI:</b> %{y:.2f}<br><extra></extra>'
    ))
    
    for category in ['El Niño', 'La Niña', 'Neutral']:
        category_data = filtered_df[filtered_df['ENSO_Category'] == category]
        if not category_data.empty:
            fig.add_trace(go.Scatter(
                x=category_data.index,
                y=category_data['ONI'],
                mode='markers',
                name=category,
                marker=dict(color=get_enso_color(category), size=6, opacity=0.7),
                hovertemplate=f'<b>{category}</b><br><b>Date:</b> %{{x}}<br><b>ONI:</b> %{{y:.2f}}<br><extra></extra>'
            ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.7, 
                  annotation_text="El Niño Threshold (+0.5)", annotation_position="bottom right")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", opacity=0.7,
                  annotation_text="La Niña Threshold (-0.5)", annotation_position="top right")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="red", opacity=0.1, 
                  annotation_text="Moderate El Niño", annotation_position="top left")
    fig.add_hrect(y0=1.0, y1=1.5, fillcolor="red", opacity=0.2, 
                  annotation_text="Strong El Niño", annotation_position="top left")
    fig.add_hrect(y0=1.5, y1=3.0, fillcolor="red", opacity=0.3, 
                  annotation_text="Very Strong El Niño", annotation_position="top left")
    
    fig.add_hrect(y0=-0.5, y1=-1.0, fillcolor="blue", opacity=0.1, 
                  annotation_text="Moderate La Niña", annotation_position="bottom left")
    fig.add_hrect(y0=-1.0, y1=-1.5, fillcolor="blue", opacity=0.2, 
                  annotation_text="Strong La Niña", annotation_position="bottom left")
    fig.add_hrect(y0=-1.5, y1=-3.0, fillcolor="blue", opacity=0.3, 
                  annotation_text="Very Strong La Niña", annotation_position="bottom left")
    
    fig.update_layout(
        title=dict(text="ONI Time Series with ENSO Events", font=dict(size=20, color='#1f77b4')),
        xaxis_title="Date",
        yaxis_title="ONI Value",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_prediction_plot(df, model_info, model, X_scaler, y_scaler):
    """Create prediction results plot with forecasting"""
    try:
        n_in = model_info['n_in']
        n_out = model_info['n_out']
        train_end = model_info['train_end']
        valid_end = model_info['valid_end']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Historical Data with Predictions', 'Forecast for Next Periods'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        train_data = df.iloc[:train_end]
        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data['ONI'],
            mode='lines',
            name='Training Data',
            line=dict(color='blue', width=1.5),
            opacity=0.7
        ), row=1, col=1)
        
        valid_data = df.iloc[train_end:valid_end]
        fig.add_trace(go.Scatter(
            x=valid_data.index,
            y=valid_data['ONI'],
            mode='lines',
            name='Validation Data',
            line=dict(color='orange', width=1.5),
            opacity=0.7
        ), row=1, col=1)
        
        test_start_idx = valid_end
        test_dates = df.index[test_start_idx:test_start_idx + len(model_info['y_true'])]
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=model_info['y_true'][:, 0],
            mode='lines+markers',
            name='Actual (Test)',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=model_info['y_pred'][:, 0],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2),
            marker=dict(size=4, symbol='square')
        ), row=1, col=1)
        
        fig.add_vline(x=df.index[train_end], line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_vline(x=df.index[valid_end], line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        
        last_sequence = df['ONI'].tail(n_in).values.reshape(1, -1)
        last_sequence_scaled = X_scaler.transform(last_sequence)
        last_sequence_scaled = last_sequence_scaled.reshape(1, n_in, 1)
        
        forecast_scaled = model.predict(last_sequence_scaled, verbose=0)
        forecast = y_scaler.inverse_transform(forecast_scaled)
        
        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_out,
            freq='MS'
        )
        
        recent_data = df['ONI'].tail(24)
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data.values,
            mode='lines',
            name='Recent History',
            line=dict(color='blue', width=2),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast[0],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2),
            marker=dict(size=8, symbol='diamond')
        ), row=2, col=1)
        
        fig.add_vline(x=df.index[-1], line_dash="dash", line_color="gray", opacity=0.7, row=2, col=1)
        
        for row in [1, 2]:
            fig.add_hline(y=0.5, line_dash="dot", line_color="red", opacity=0.5, row=row, col=1)
            fig.add_hline(y=-0.5, line_dash="dot", line_color="blue", opacity=0.5, row=row, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3, row=row, col=1)
        
        fig.update_layout(
            title=dict(text="ENSO Prediction Results & Forecast", font=dict(size=20, color='#1f77b4')),
            height=800,
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="ONI Value", row=1, col=1)
        fig.update_yaxes(title_text="ONI Value", row=2, col=1)
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast_ONI': forecast[0],
            'ENSO_Category': classify_enso(forecast[0])
        })
        
        return fig, forecast_df
        
    except Exception as e:
        st.error(f"Error creating prediction plot: {e}")
        return go.Figure(), pd.DataFrame()

def calculate_summary_stats(df, start_date, end_date):
    """Calculate summary statistics for selected date range"""
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df[mask]
    
    if filtered_df.empty:
        return {}
    
    stats = {
        'count': len(filtered_df),
        'mean': filtered_df['ONI'].mean(),
        'min': filtered_df['ONI'].min(),
        'max': filtered_df['ONI'].max(),
        'std': filtered_df['ONI'].std(),
        'min_date': filtered_df['ONI'].idxmin(),
        'max_date': filtered_df['ONI'].idxmax()
    }
    
    enso_categories = classify_enso(filtered_df['ONI'].values)
    unique, counts = np.unique(enso_categories, return_counts=True)
    enso_counts = dict(zip(unique, counts))
    
    stats.update(enso_counts)
    return stats

def create_download_link(df, filename, text):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    st.markdown('<h1 class="main-header">🌊 ENSO Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.spinner("Loading model and data..."):
        df_enso, model, X_scaler, y_scaler, model_info = load_data_and_model()
    
    if df_enso is None:
        st.stop()
    
    st.sidebar.markdown("## 📊 Dashboard Controls")
    
    min_date = df_enso.index.min().date()
    max_date = df_enso.index.max().date()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    plot_option = st.sidebar.selectbox(
        "Select Visualization",
        ["Both Plots", "ONI Time Series", "Predictions & Forecast"]
    )
    
    st.sidebar.markdown("## 🎯 Model Performance")
    if model_info:
        metrics = model_info['metrics']
        accuracy = model_info['accuracy']
        
        st.sidebar.metric("Classification Accuracy", f"{accuracy:.1%}")
        st.sidebar.metric("R² Score", f"{metrics['R²']:.3f}")
        st.sidebar.metric("MAE", f"{metrics['MAE']:.3f}")
        st.sidebar.metric("RMSE", f"{metrics['RMSE']:.3f}")
    
    st.sidebar.markdown("## 📈 Data Summary")
    stats = calculate_summary_stats(df_enso, pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    if stats:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Average ONI", f"{stats['mean']:.2f}")
            st.metric("Minimum ONI", f"{stats['min']:.2f}")
        with col2:
            st.metric("Maximum ONI", f"{stats['max']:.2f}")
            st.metric("Std Deviation", f"{stats['std']:.2f}")
        
        st.sidebar.markdown("### ENSO Event Counts")
        for category in ['El Niño', 'La Niña', 'Neutral']:
            count = stats.get(category, 0)
            percentage = (count / stats['count']) * 100 if stats['count'] > 0 else 0
            st.sidebar.write(f"**{category}:** {count} ({percentage:.1f}%)")
    
    if plot_option in ["Both Plots", "ONI Time Series"]:
        st.markdown("## 🌊 ONI Time Series with ENSO Events")
        with st.spinner("Creating ONI time series plot..."):
            oni_fig = create_oni_timeseries_plot(df_enso, pd.Timestamp(start_date), pd.Timestamp(end_date))
            st.plotly_chart(oni_fig, use_container_width=True)
        
        filtered_data = df_enso[(df_enso.index >= pd.Timestamp(start_date)) & 
                               (df_enso.index <= pd.Timestamp(end_date))].copy()
        if not filtered_data.empty:
            filtered_data['ENSO_Category'] = classify_enso(filtered_data['ONI'].values)
            st.markdown(create_download_link(filtered_data, f"oni_data_{start_date}_{end_date}.csv", 
                                           "📥 Download ONI Data"), unsafe_allow_html=True)
    
    if plot_option in ["Both Plots", "Predictions & Forecast"]:
        st.markdown("## 🔮 Predictions & Forecast")
        with st.spinner("Creating prediction and forecast plot..."):
            pred_fig, forecast_df = create_prediction_plot(df_enso, model_info, model, X_scaler, y_scaler)
            st.plotly_chart(pred_fig, use_container_width=True)
        
        if not forecast_df.empty:
            st.markdown("### 📅 Forecast Values")
            
            forecast_display = forecast_df.copy()
            forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m')
            forecast_display['Forecast_ONI'] = forecast_display['Forecast_ONI'].round(3)
            forecast_display = forecast_display.rename(columns={
                'Date': 'Month',
                'Forecast_ONI': 'Predicted ONI',
                'ENSO_Category': 'Predicted Event'
            })
            
            st.dataframe(
                forecast_display,
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown(create_download_link(forecast_df, "enso_forecast.csv", 
                                           "📥 Download Forecast Data"), unsafe_allow_html=True)
            
            st.markdown("### 🔍 Forecast Interpretation")
            for _, row in forecast_display.iterrows():
                month = row['Month']
                oni_val = row['Predicted ONI']
                event = row['Predicted Event']
                
                if event == 'El Niño':
                    color = "#FF6B6B"
                    icon = "🔥"
                elif event == 'La Niña':
                    color = "#4ECDC4"
                    icon = "❄️"
                else:
                    color = "#45B7D1"
                    icon = "⚖️"
                
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>{icon} {month}:</strong> ONI = {oni_val:.3f} → <strong>{event}</strong>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("## 💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    
    with col1:
        st.markdown("""
        ### 🌊 ENSO Categories
        - **El Niño:** ONI ≥ +0.5°C (warm)
        - **La Niña:** ONI ≤ -0.5°C (cool)  
        - **Neutral:** -0.5°C < ONI < +0.5°C
        - **Intensity levels:** Weak, Moderate, Strong, Very Strong
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Understanding ONI
        - **ONI:** Oceanic Niño Index
        - **Measures:** Sea surface temperature anomalies
        - **Region:** Central Pacific Ocean
        - **Impact:** Global weather patterns
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌊 ENSO Prediction Dashboard | Built with Streamlit & LSTM Neural Networks</p>
        <p>Data Source: NOAA Climate Prediction Center | Model: Deep Learning LSTM</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()