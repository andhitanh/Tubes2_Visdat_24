import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Indonesia Crime Data Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load and preprocess crime data"""
    try:
        df = pd.read_csv('crime_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M')
        return df
    except FileNotFoundError:
        st.error("Crime data file not found. Please ensure 'crime_data.csv' exists.")
        return pd.DataFrame()

def create_metrics_cards(df):
    """Create top metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = len(df)
        st.metric(
            label="Total Kasus",
            value=f"{total_cases:,}",
            delta=f"+{len(df[df['year'] == df['year'].max()]):,} (2024)"
        )
    
    with col2:
        most_common_crime = df['crime_type'].mode()[0] if not df.empty else "N/A"
        crime_count = df['crime_type'].value_counts().iloc[0] if not df.empty else 0
        st.metric(
            label="Kejahatan Terbanyak",
            value=most_common_crime,
            delta=f"{crime_count:,} kasus"
        )
    
    with col3:
        ongoing_cases = len(df[df['status'] == 'Dalam Proses'])
        st.metric(
            label="Kasus Sedang Ditangani",
            value=f"{ongoing_cases:,}",
            delta=f"{(ongoing_cases/total_cases*100):.1f}%" if total_cases > 0 else "0%"
        )
    
    with col4:
        resolved_cases = len(df[df['status'] == 'Selesai'])
        resolution_rate = (resolved_cases/total_cases*100) if total_cases > 0 else 0
        st.metric(
            label="Tingkat Penyelesaian",
            value=f"{resolution_rate:.1f}%",
            delta=f"{resolved_cases:,} kasus"
        )

def create_crime_trend_chart(df):
    """Create crime trend over time chart"""
    monthly_data = df.groupby(['year_month', 'crime_type']).size().reset_index(name='count')
    monthly_data['year_month_str'] = monthly_data['year_month'].astype(str)
    
    fig = px.line(
        monthly_data, 
        x='year_month_str', 
        y='count', 
        color='crime_type',
        title='Tren Pertumbuhan Kejahatan dari Waktu ke Waktu',
        labels={'year_month_str': 'Bulan-Tahun', 'count': 'Jumlah Kasus', 'crime_type': 'Jenis Kejahatan'}
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_location_distribution_chart(df):
    """Create location distribution chart (simulated choropleth)"""
    location_data = df['location'].value_counts().reset_index()
    location_data.columns = ['Province', 'Cases']
    
    fig = px.bar(
        location_data.head(15), 
        x='Cases', 
        y='Province',
        orientation='h',
        title='Distribusi Kejahatan per Provinsi (Top 15)',
        labels={'Cases': 'Jumlah Kasus', 'Province': 'Provinsi'}
    )
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_demographics_charts(df):
    """Create demographic analysis charts"""
    # Age distribution
    age_bins = [0, 18, 25, 35, 45, 55, 100]
    age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55+']
    df['age_group'] = pd.cut(df['perp_age'], bins=age_bins, labels=age_labels, right=False)
    
    age_data = df['age_group'].value_counts().reset_index()
    age_data.columns = ['Age Group', 'Count']
    
    fig_age = px.bar(
        age_data, 
        x='Age Group', 
        y='Count',
        title='Distribusi Usia Pelaku',
        labels={'Count': 'Jumlah Kasus', 'Age Group': 'Kelompok Usia'}
    )
    
    # Gender distribution
    gender_data = df['perp_gender'].value_counts().reset_index()
    gender_data.columns = ['Gender', 'Count']
    
    fig_gender = px.pie(
        gender_data, 
        values='Count', 
        names='Gender',
        title='Distribusi Jenis Kelamin Pelaku'
    )
    
    # Occupation distribution
    occupation_data = df['perp_occupation'].value_counts().head(10).reset_index()
    occupation_data.columns = ['Occupation', 'Count']
    
    fig_occupation = px.bar(
        occupation_data, 
        x='Count', 
        y='Occupation',
        orientation='h',
        title='Distribusi Pekerjaan Pelaku (Top 10)',
        labels={'Count': 'Jumlah Kasus', 'Occupation': 'Pekerjaan'}
    )
    
    # Apply dark theme
    for fig in [fig_age, fig_gender, fig_occupation]:
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    return fig_age, fig_gender, fig_occupation

def create_motive_wordcloud(df):
    """Create word cloud for crime motives"""
    motive_text = ' '.join(df['motive'].tolist())
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='black',
        colormap='Blues',
        max_words=50
    ).generate(motive_text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud Motif Kejahatan', color='white', fontsize=16, pad=20)
    fig.patch.set_facecolor('black')
    
    return fig

def create_time_series_chart(df):
    """Create detailed time series analysis"""
    daily_data = df.groupby(df['date'].dt.date).size().reset_index(name='count')
    daily_data.columns = ['date', 'count']
    
    fig = px.line(
        daily_data, 
        x='date', 
        y='count',
        title='Tren Harian Kejahatan',
        labels={'date': 'Tanggal', 'count': 'Jumlah Kasus'}
    )
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def main():
    # Load CSS
    try:
        load_css()
    except FileNotFoundError:
        pass  # CSS file is optional
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.stop()
    
    # Sidebar filters
    st.sidebar.title("🔍 Filter Data")
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Location filter
    locations = ['Semua'] + sorted(df['location'].unique().tolist())
    selected_location = st.sidebar.selectbox("Pilih Provinsi", locations)
    
    # Crime type filter
    crime_types = df['crime_type'].unique().tolist()
    selected_crimes = st.sidebar.multiselect(
        "Pilih Jenis Kejahatan",
        crime_types,
        default=crime_types[:5]  # Default to top 5
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Location filter
    if selected_location != 'Semua':
        filtered_df = filtered_df[filtered_df['location'] == selected_location]
    
    # Crime type filter
    if selected_crimes:
        filtered_df = filtered_df[filtered_df['crime_type'].isin(selected_crimes)]
    
    # Main dashboard
    st.title("🚔 Dashboard Data Kejahatan Indonesia")
    st.markdown("---")
    
    # Metrics cards
    create_metrics_cards(filtered_df)
    st.markdown("---")
    
    # Crime trend chart
    st.plotly_chart(create_crime_trend_chart(filtered_df), use_container_width=True)
    
    # Location distribution
    st.plotly_chart(create_location_distribution_chart(filtered_df), use_container_width=True)
    
    # Demographics section
    st.markdown("## 👥 Analisis Demografi Pelaku")
    
    col1, col2, col3 = st.columns(3)
    
    fig_age, fig_gender, fig_occupation = create_demographics_charts(filtered_df)
    
    with col1:
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col3:
        st.plotly_chart(fig_occupation, use_container_width=True)
    
    # Motive analysis
    st.markdown("## 🎯 Analisis Motif Kejahatan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            fig_wordcloud = create_motive_wordcloud(filtered_df)
            st.pyplot(fig_wordcloud)
        except Exception as e:
            st.error(f"Error creating word cloud: {e}")
            # Fallback to bar chart
            motive_data = filtered_df['motive'].value_counts().head(10).reset_index()
            motive_data.columns = ['Motive', 'Count']
            fig_motive = px.bar(motive_data, x='Count', y='Motive', orientation='h')
            fig_motive.update_layout(template='plotly_dark')
            st.plotly_chart(fig_motive, use_container_width=True)
    
    with col2:
        st.markdown("### Top 10 Motif")
        motive_counts = filtered_df['motive'].value_counts().head(10)
        for i, (motive, count) in enumerate(motive_counts.items(), 1):
            st.write(f"{i}. **{motive}**: {count} kasus")
    
    # Time series analysis
    st.markdown("## 📈 Analisis Tren Waktu")
    st.plotly_chart(create_time_series_chart(filtered_df), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Dashboard Data Kejahatan Indonesia | Data simulasi untuk tujuan demonstrasi
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()