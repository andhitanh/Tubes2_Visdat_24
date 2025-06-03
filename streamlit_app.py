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

@st.cache_data
def load_polda_data():
    """Load police headquarters location data"""
    try:
        polda_df = pd.read_csv('data_polda.csv')
        return polda_df
    except FileNotFoundError:
        st.error("Polda data file not found. Using fallback coordinates.")
        # Fallback data if file not found
        return pd.DataFrame({
            'province': ['DKI JAKARTA', 'JAWA BARAT', 'JAWA TIMUR'],
            'latitude': [-6.2214, -6.9386, -7.322222],
            'longitude': [106.8098, 107.7033, 112.730757]
        })

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

def prepare_map_data(df):
    """Prepare data for map visualization with detailed analytics"""
    # Load polda coordinates
    polda_df = load_polda_data()
    
    # Aggregate crime data by province
    province_stats = []
    
    for province in df['location'].unique():
        province_data = df[df['location'] == province]
        
        if len(province_data) == 0:
            continue
            
        # Basic stats
        total_cases = len(province_data)
        
        # Top 3 crime types
        crime_counts = province_data['crime_type'].value_counts()
        top_3_crimes = crime_counts.head(3)
        top_crimes_text = []
        for crime, count in top_3_crimes.items():
            percentage = (count / total_cases) * 100
            top_crimes_text.append(f"• {crime}: {count} ({percentage:.1f}%)")
        
        # Time analysis - assuming you have a 'time' or 'hour' column in your data
        if 'time' in province_data.columns:
            # Convert time to hour if it's in time format
            province_data['hour'] = pd.to_datetime(province_data['time']).dt.hour
        elif 'hour' in province_data.columns:
            # Use existing hour column
            pass
        else:
            # Create dummy hour data for demonstration
            province_data = province_data.copy()
            province_data['hour'] = np.random.randint(0, 24, len(province_data))
        
        # Determine peak time range
        hour_counts = province_data['hour'].value_counts()
        peak_hours = hour_counts.nlargest(8).index.tolist()  # Top 8 hours
        peak_hours.sort()
        
        if len(peak_hours) > 0:
            time_range = f"{min(peak_hours):02d}:00–{max(peak_hours):02d}:59"
        else:
            time_range = "Data tidak tersedia"
        
        # Most common location type
        if 'location_type' in province_data.columns:
            most_common_location = province_data['location_type'].mode()[0]
            location_counts = province_data['location_type'].value_counts()
            location_percentage = (location_counts.iloc[0] / total_cases) * 100
            typical_location = f"{most_common_location} ({location_percentage:.1f}%)"
        else:
            # Create dummy location data for demonstration
            locations = ['Rumah', 'Jalan Umum', 'Area Perkantoran', 'Pusat Perbelanjaan', 'Tempat Umum']
            province_data = province_data.copy()
            province_data['location_type'] = np.random.choice(locations, len(province_data))
            most_common_location = province_data['location_type'].mode()[0]
            typical_location = most_common_location
        
        # Get coordinates from polda data
        polda_match = polda_df[polda_df['province'] == province]
        if not polda_match.empty:
            lat = polda_match.iloc[0]['latitude']
            lon = polda_match.iloc[0]['longitude']
        else:
            # Default coordinates for Indonesia center if not found
            lat = -0.7893
            lon = 113.9213
        
        province_stats.append({
            'province': province,
            'total_cases': total_cases,
            'latitude': lat,
            'longitude': lon,
            'top_3_crimes': '<br>'.join(top_crimes_text),
            'time_range': time_range,
            'typical_location': typical_location,
            'cases_per_100k': total_cases  # You can adjust this with real population data
        })
    
    return pd.DataFrame(province_stats)

def create_indonesia_crime_map(df):
    """Create interactive choropleth map of Indonesia crime data"""
    
    # Prepare map data
    map_data = prepare_map_data(df)
    
    if map_data.empty:
        st.warning("Tidak ada data untuk ditampilkan pada peta")
        return None
    
    # Create the map
    fig = go.Figure()
    
    # Add scatter plot with sized circles for each province
    fig.add_trace(go.Scattergeo(
        lon=map_data['longitude'],
        lat=map_data['latitude'],
        text=map_data['province'],
        mode='markers',
        marker=dict(
            size=map_data['total_cases'] / 10,  # Scale marker size
            sizemin=8,
            color=map_data['total_cases'],
            colorscale='Reds',
            colorbar=dict(
                title=dict(
                    text="Jumlah Kasus",
                    font=dict(color='white')
                ),
                tickfont=dict(color='white')
            ),
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        hovertemplate=
        "<b>%{text}</b><br>" +
        "<b>Total Kasus:</b> %{customdata[0]:,}<br>" +
        "<b>3 Kejahatan Teratas:</b><br>%{customdata[1]}<br>" +
        "<b>Waktu Puncak:</b> %{customdata[2]}<br>" +
        "<b>Lokasi Paling Umum:</b> %{customdata[3]}<br>" +
        "<extra></extra>",
        customdata=list(zip(
            map_data['total_cases'],
            map_data['top_3_crimes'],
            map_data['time_range'],
            map_data['typical_location']
        )),
        name=""
    ))
    
    # Update layout for Indonesia focus
    fig.update_layout(
        title={
            'text': 'Peta Distribusi Kejahatan di Indonesia',
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(40, 40, 40)',
            showocean=True,
            oceancolor='rgb(20, 20, 30)',
            showlakes=True,
            lakecolor='rgb(20, 20, 30)',
            showrivers=True,
            rivercolor='rgb(20, 20, 30)',
            showcountries=True,
            countrycolor='rgb(60, 60, 60)',
            countrywidth=1,
            center=dict(lat=-2.5, lon=118),  # Center on Indonesia
            lonaxis=dict(range=[95, 141]),   # Indonesia longitude range
            lataxis=dict(range=[-11, 6]),    # Indonesia latitude range
            bgcolor='rgba(0,0,0,0)'
        ),
        height=600,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_province_summary_table(df):
    """Create a summary table alongside the map"""
    map_data = prepare_map_data(df)
    
    if map_data.empty:
        return None
    
    # Sort by total cases
    map_data = map_data.sort_values('total_cases', ascending=False)
    
    # Create a styled table
    table_data = []
    for _, row in map_data.head(10).iterrows():  # Top 10 provinces
        table_data.append({
            'Provinsi': row['province'],
            'Total Kasus': f"{row['total_cases']:,}",
            'Waktu Puncak': row['time_range'],
            'Lokasi Umum': row['typical_location']
        })
    
    return pd.DataFrame(table_data)

def create_location_distribution_section(df):
    """Enhanced location distribution with map and summary"""
    
    st.markdown("## 🗺️ Distribusi Kejahatan per Provinsi")
    
    # Create two columns: map and summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display the interactive map
        map_fig = create_indonesia_crime_map(df)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.error("Gagal membuat peta. Periksa data koordinat provinsi.")
    
    with col2:
        st.markdown("### 📊 Top 10 Provinsi")
        summary_table = create_province_summary_table(df)
        if summary_table is not None:
            st.dataframe(
                summary_table,
                use_container_width=True,
                hide_index=True
            )
            
            # Add some key insights
            st.markdown("### 🔍 Insights Cepat")
            map_data = prepare_map_data(df)
            if not map_data.empty:
                highest_crime = map_data.loc[map_data['total_cases'].idxmax()]
                st.info(f"📍 **Provinsi dengan kasus terbanyak:** {highest_crime['province']} ({highest_crime['total_cases']:,} kasus)")
                
                avg_cases = map_data['total_cases'].mean()
                st.info(f"📊 **Rata-rata kasus per provinsi:** {avg_cases:.0f}")
        else:
            st.warning("Data tidak tersedia untuk tabel ringkasan")

def create_demographics_charts(df):
    """Create demographic analysis charts with consistent heights"""
    
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
        labels={'Count': 'Jumlah Kasus', 'Age Group': 'Kelompok Usia'},
        color='Count',
        color_continuous_scale='Blues'
    )
    
    fig_age.update_layout(
        height=400,  # Fixed height
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Gender distribution
    gender_data = df['perp_gender'].value_counts().reset_index()
    gender_data.columns = ['Gender', 'Count']
    
    fig_gender = px.pie(
        gender_data, 
        values='Count', 
        names='Gender',
        title='Distribusi Jenis Kelamin Pelaku',
        color_discrete_sequence=['#3182ce', '#4299e1', '#63b3ed']
    )
    
    fig_gender.update_layout(
        height=400,  # Fixed height
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Occupation distribution
    occupation_data = df['perp_occupation'].value_counts().head(8).reset_index()  # Reduced to 8 for better fit
    occupation_data.columns = ['Occupation', 'Count']
    
    fig_occupation = px.bar(
        occupation_data, 
        x='Count', 
        y='Occupation',
        orientation='h',
        title='Top 8 Pekerjaan Pelaku',
        labels={'Count': 'Jumlah Kasus', 'Occupation': 'Pekerjaan'},
        color='Count',
        color_continuous_scale='Blues'
    )
    
    fig_occupation.update_layout(
        height=400,  # Fixed height
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=120, r=50, t=80, b=50),  # More left margin for occupation names
        yaxis={'categoryorder': 'total ascending'}  # Sort bars by value
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
    ax.set_title('Motif Kejahatan', color='white', fontsize=16, pad=20)
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
    
    # Time series analysis
    st.markdown("## 📈 Analisis Tren Waktu")
    st.plotly_chart(create_time_series_chart(filtered_df), use_container_width=True)
    
    # Location distribution - REPLACED WITH MAP
    create_location_distribution_section(filtered_df)
    
    # Demographics section - IMPROVED VERSION
    st.markdown("## 👥 Analisis Demografi Pelaku")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📊 Distribusi Umum", "💼 Analisis Pekerjaan"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        fig_age, fig_gender, fig_occupation = create_demographics_charts(filtered_df)
        
        with col1:
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_gender, use_container_width=True)
    
    with tab2:
        # Full width for occupation chart since it needs more space
        st.plotly_chart(fig_occupation, use_container_width=True)
        
        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_age = filtered_df['perp_age'].mean()
            st.metric(
                label="Rata-rata Usia Pelaku",
                value=f"{avg_age:.1f} tahun"
            )
        
        with col2:
            most_common_gender = filtered_df['perp_gender'].mode()[0]
            gender_pct = (filtered_df['perp_gender'].value_counts().iloc[0] / len(filtered_df) * 100)
            st.metric(
                label="Jenis Kelamin Dominan",
                value=most_common_gender,
                delta=f"{gender_pct:.1f}%"
            )
        
        with col3:
            most_common_job = filtered_df['perp_occupation'].mode()[0]
            job_count = filtered_df['perp_occupation'].value_counts().iloc[0]
            st.metric(
                label="Pekerjaan Terbanyak",
                value=most_common_job,
                delta=f"{job_count} kasus"
            )
    
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