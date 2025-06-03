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
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file is optional

@st.cache_data
def load_all_data():
    """Load all CSV files and return as dictionary"""
    data_files = {
        'polda': 'data/data_polda.csv',
        'age': 'data/age.csv',
        'sex': 'data/sex.csv',
        'occupation': 'data/occupation.csv',
        'location': 'data/location.csv',
        'time': 'data/time.csv',
        'status': 'data/status.csv',
        'motive': 'data/motive.csv'
    }
    
    data = {}
    for key, file_path in data_files.items():
        try:
            df = pd.read_csv(file_path)
            data[key] = df
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            data[key] = pd.DataFrame()
    
    return data

def initialize_session_state(min_date, max_date, provinces, crime_types):
    if 'applied_filters' not in st.session_state:
        st.session_state.applied_filters = {
            'start_date': min_date,
            'end_date': max_date,
            'selected_province': 'Semua',
            'selected_crimes': []
        }
    
    if 'current_filters' not in st.session_state:
        st.session_state.current_filters = st.session_state.applied_filters.copy()

def get_date_range(data):
    """Get the min and max date range from all data"""
    all_dates = []
    
    for key in ['age', 'sex', 'occupation', 'location', 'time', 'status', 'motive']:
        if not data[key].empty and 'year' in data[key].columns and 'month' in data[key].columns:
            df = data[key]
            # Create datetime from year and month
            df_dates = pd.to_datetime(df[['year', 'month']].assign(day=1))
            all_dates.extend(df_dates.tolist())
    
    if all_dates:
        return min(all_dates), max(all_dates)
    else:
        # Fallback dates
        return datetime(2022, 1, 1), datetime(2024, 12, 1)

def filter_data_by_date_and_location(data, start_date, end_date, selected_province=None, selected_polda=None):
    """Filter all datasets by date range and location"""
    filtered_data = {}
    
    for key in ['age', 'sex', 'occupation', 'location', 'time', 'status', 'motive']:
        if data[key].empty:
            filtered_data[key] = pd.DataFrame()
            continue
            
        df = data[key].copy()
        
        # Filter by date range - improved logic
        mask = (
            (df['year'] > start_date.year) | 
            ((df['year'] == start_date.year) & (df['month'] >= start_date.month))
        ) & (
            (df['year'] < end_date.year) | 
            ((df['year'] == end_date.year) & (df['month'] <= end_date.month))
        )
        
        df = df[mask]
        
        # Filter by location if specified
        if selected_polda and selected_polda != 'Semua':
            df = df[df['polda'] == selected_polda]
        elif selected_province and selected_province != 'Semua':
            # Get polda list for the selected province
            if not data['polda'].empty:
                province_poldas = data['polda'][data['polda']['province'] == selected_province]['polda'].tolist()
                df = df[df['polda'].isin(province_poldas)]
        
        filtered_data[key] = df
    
    # Also filter polda data for consistency
    filtered_data['polda'] = data['polda'].copy()
    if selected_province and selected_province != 'Semua':
        filtered_data['polda'] = data['polda'][data['polda']['province'] == selected_province]
    
    return filtered_data

def create_filter_sidebar(min_date, max_date, data):
    """Create sidebar with filters and return current filter values"""
    st.sidebar.title("Filter Data")
    
    # Date range slider
    def get_month_year_options(min_date, max_date):
        """Generate list of (year, month) tuples and their display names"""
        options = []
        current = min_date
        
        while current <= max_date:
            month_name = current.strftime("%b %Y")
            options.append((current.year, current.month, month_name))
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return options
    
    month_options = get_month_year_options(min_date, max_date)
    month_labels = [opt[2] for opt in month_options]

    # Find current applied filter indices
    applied_start = st.session_state.applied_filters['start_date']
    applied_end = st.session_state.applied_filters['end_date']
    
    start_idx = 0
    end_idx = len(month_options) - 1
    
    for i, (year, month, _) in enumerate(month_options):
        if year == applied_start.year and month == applied_start.month:
            start_idx = i
        if year == applied_end.year and month == applied_end.month:
            end_idx = i

    # Double-sided select_slider
    if len(month_options) > 1:
        idx_range = st.sidebar.select_slider(
            "Rentang Waktu Kejadian",
            options=list(range(len(month_options))),
            value=(start_idx, end_idx),
            format_func=lambda x: month_options[x][2],
            key="date_range_slider"
        )

        start_idx, end_idx = idx_range

        # Ensure order
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        start_year, start_month, _ = month_options[start_idx]
        end_year, end_month, _ = month_options[end_idx]

        current_start_date = datetime(start_year, start_month, 1)
        current_end_date = datetime(end_year, end_month, 1)
    else:
        current_start_date = min_date
        current_end_date = max_date
    
    # Province filter
    provinces = ['Semua'] + sorted(data['polda']['province'].unique().tolist())
    current_selected_province = st.sidebar.selectbox(
        "Lokasi", 
        provinces,
        index=provinces.index(st.session_state.applied_filters['selected_province']) 
              if st.session_state.applied_filters['selected_province'] in provinces else 0,
        key="province_select"
    )
    
    # Crime type filter
    all_crime_types = []
    if not data['age'].empty:
        all_crime_types = sorted(data['age']['crime_type'].unique().tolist())
    
    current_selected_crimes = []
    if all_crime_types:
        current_selected_crimes = st.sidebar.multiselect(
            "Jenis Kejahatan",
            all_crime_types,
            default=st.session_state.applied_filters['selected_crimes'],
            key="crime_types_select"
        )
    
    # Terapkan button
    apply_clicked = st.sidebar.button("Terapkan", use_container_width=True, type="secondary")
    
    # Handle button click
    if apply_clicked:
        st.session_state.applied_filters = {
            'start_date': current_start_date,
            'end_date': current_end_date,
            'selected_province': current_selected_province,
            'selected_crimes': current_selected_crimes
        }
        st.rerun()
    
    # Show current applied filters info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filter Aktif")
    applied = st.session_state.applied_filters
    st.sidebar.info(f"""
    **Periode:** {applied['start_date'].strftime('%b %Y')} - {applied['end_date'].strftime('%b %Y')}
    
    **Lokasi:** {applied['selected_province']}
    
    **Jenis Kejahatan:** {len(applied['selected_crimes'])} dipilih
    """)
    
    return st.session_state.applied_filters

def scale_marker_sizes(values, min_size=8, max_size=50, method='sqrt'):
    values = np.array(values)
    
    if len(values) == 0:
        return np.array([])
    if np.all(values == values[0]):  # All values are the same
        return np.full(len(values), (min_size + max_size) / 2)
    
    if method == 'linear':
        min_val, max_val = values.min(), values.max()
        scaled = min_size + (values - min_val) * (max_size - min_size) / (max_val - min_val)
        
    elif method == 'sqrt':
        sqrt_values = np.sqrt(values)
        min_val, max_val = sqrt_values.min(), sqrt_values.max()
        scaled = min_size + (sqrt_values - min_val) * (max_size - min_size) / (max_val - min_val)
        
    elif method == 'log':
        log_values = np.log1p(values)  # log(1 + values)
        min_val, max_val = log_values.min(), log_values.max()
        scaled = min_size + (log_values - min_val) * (max_size - min_size) / (max_val - min_val)
        
    elif method == 'quantile':
        percentiles = [np.percentile(values, p) for p in np.linspace(0, 100, len(values))]
        scaled = np.interp(values, sorted(values), 
                          np.linspace(min_size, max_size, len(values)))
    
    else:
        raise ValueError("Method must be 'linear', 'sqrt', 'log', or 'quantile'")
    
    return scaled

def create_metrics_cards(filtered_data):
    """Create top metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate total cases from any dataset (using age as reference)
    total_cases = filtered_data['age']['count_age'].sum() if not filtered_data['age'].empty else 0
    
    with col1:
        st.metric(
            label="Total Kasus",
            value=f"{total_cases:,.0f}",
            delta="Berdasarkan data terpilih"
        )
    
    with col2:
        if not filtered_data['age'].empty:
            # Most common crime type
            crime_totals = filtered_data['age'].groupby('crime_type')['count_age'].sum()
            most_common_crime = crime_totals.idxmax() if not crime_totals.empty else "N/A"
            crime_count = crime_totals.max() if not crime_totals.empty else 0
            st.metric(
                label="Kejahatan Terbanyak",
                value=most_common_crime[:20] + "..." if len(most_common_crime) > 20 else most_common_crime,
                delta=f"{crime_count:,.0f} kasus"
            )
        else:
            st.metric(label="Kejahatan Terbanyak", value="N/A", delta="0 kasus")
    
    with col3:
        if not filtered_data['status'].empty:
            ongoing_statuses = ['Dalam Proses','Proses Sidik', 'Proses Lidik']
            ongoing_cases = filtered_data['status'][filtered_data['status']['status'].isin(ongoing_statuses)]['count_status'].sum()
            total_status_cases = filtered_data['status']['count_status'].sum()
            percentage = (ongoing_cases/total_status_cases*100) if total_status_cases > 0 else 0
            st.metric(
                label="Kasus Sedang Ditangani",
                value=f"{ongoing_cases:,.0f}",
                delta=f"{percentage:.1f}%"
            )
        else:
            st.metric(label="Kasus Sedang Ditangani", value="N/A", delta="0%")
    
    with col4:
        if not filtered_data['status'].empty:
            completed_statuses = ['Selesai', 'Selesai Perkara (CC)', 'Selesai Perkara']
            resolved_cases = filtered_data['status'][filtered_data['status']['status'].isin(completed_statuses)]['count_status'].sum()
            total_status_cases = filtered_data['status']['count_status'].sum()
            resolution_rate = (resolved_cases/total_status_cases*100) if total_status_cases > 0 else 0
            st.metric(
                label="Tingkat Penyelesaian",
                value=f"{resolution_rate:.1f}%",
                delta=f"{resolved_cases:,.0f} kasus"
            )
        else:
            st.metric(label="Tingkat Penyelesaian", value="0%", delta="0 kasus")

def create_time_series_chart(filtered_data):
    """Create time series chart from filtered data"""
    if filtered_data['age'].empty:
        return None
    
    # Aggregate data by month/year
    df = filtered_data['age'].copy()
    monthly_data = df.groupby(['year', 'month'])['count_age'].sum().reset_index()
    monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
    monthly_data = monthly_data.sort_values('date')
    
    fig = px.line(
        monthly_data,
        x='date',
        y='count_age',
        title='Tren Bulanan Kejahatan',
        labels={'date': 'Tanggal', 'count_age': 'Jumlah Kasus'}
    )
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_indonesia_crime_map(filtered_data, hide_summary=False):
    """Create interactive map of Indonesia crime data"""
    if filtered_data['age'].empty or filtered_data['polda'].empty:
        st.warning("Tidak ada data untuk ditampilkan pada peta")
        return None, None
    
    # Aggregate crime data by polda
    polda_stats = filtered_data['age'].groupby('polda').agg({
        'count_age': 'sum'
    }).reset_index()
    polda_stats.columns = ['polda', 'total_cases']
    
    # Get top 3 crimes for each polda
    top_crimes_by_polda = {}
    for polda in polda_stats['polda'].unique():
        polda_crimes = filtered_data['age'][filtered_data['age']['polda'] == polda]
        crime_totals = polda_crimes.groupby('crime_type')['count_age'].sum().sort_values(ascending=False)
        top_3 = crime_totals.head(3)
        top_crimes_text = []
        for crime, count in top_3.items():
            percentage = (count / polda_crimes['count_age'].sum()) * 100
            top_crimes_text.append(f"• {crime[:30]}: {count:,.0f} ({percentage:.1f}%)")
        top_crimes_by_polda[polda] = '<br>'.join(top_crimes_text)
    
    # Merge with coordinate data
    map_data = pd.merge(polda_stats, filtered_data['polda'], on='polda', how='left')
    map_data['top_3_crimes'] = map_data['polda'].map(top_crimes_by_polda)
    
    marker_sizes = scale_marker_sizes(
        map_data['total_cases'], 
        min_size=8, 
        max_size=50, 
        method='sqrt'
    )
    
    # Create the map
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lon=map_data['longitude'],
        lat=map_data['latitude'],
        text=map_data['polda'],
        mode='markers',
        marker=dict(
            size=marker_sizes, 
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
        "<b>Provinsi:</b> %{customdata[0]}<br>" +
        "<b>Total Kasus:</b> %{customdata[1]:,}<br>" +
        "<b>3 Kejahatan Teratas:</b><br>%{customdata[2]}<br>" +
        "<extra></extra>",
        customdata=list(zip(
            map_data['province'],
            map_data['total_cases'],
            map_data['top_3_crimes']
        )),
        name=""
    ))
    
    # Update layout for Indonesia focus
    fig.update_layout(
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(40, 40, 40)',
            showocean=True,
            oceancolor='rgb(20, 20, 30)',
            showlakes=True,
            lakecolor='rgb(20, 20, 30)',
            showcountries=True,
            countrycolor='rgb(60, 60, 60)',
            center=dict(lat=-2.5, lon=118),
            lonaxis=dict(range=[95, 141]),
            lataxis=dict(range=[-11, 6]),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=600,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Create summary table (only if not hiding)
    summary_table = None
    if not hide_summary and 'province' in map_data.columns:
        # Get province-level statistics
        province_summary = map_data.groupby('province')['total_cases'].sum().sort_values(ascending=False).reset_index()
        province_summary.columns = ['Provinsi', 'Total Kasus']
        province_summary['Total Kasus'] = province_summary['Total Kasus'].apply(lambda x: f"{x:,.0f}")
        summary_table = province_summary.head(10)
    
    return fig, summary_table

def create_demographics_charts(filtered_data):
    """Create demographic analysis charts (age, gender, occupation) with 'unknown' label converted"""

    # ----- Age Distribution -----
    if not filtered_data['age'].empty and {'age', 'count_age'}.issubset(filtered_data['age'].columns):
        age_data = (
            filtered_data['age']
            .groupby('age', as_index=False)['count_age']
            .sum()
            .rename(columns={'age': 'Age Group', 'count_age': 'Count'})
        )

        age_data['Age Group'] = age_data['Age Group'].replace({'unknown': 'Tidak Diketahui'})

        fig_age = px.bar(
            age_data,
            x='Age Group',
            y='Count',
            title='Distribusi Usia Pelaku',
            labels={'Count': 'Jumlah Kasus', 'Age Group': 'Kelompok Usia'},
            color='Count',
            color_continuous_scale='Blues'
        )
    else:
        fig_age = px.bar(title='Distribusi Usia Pelaku - Tidak ada data')

    fig_age.update_layout(
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    # ----- Gender Distribution -----
    if not filtered_data['sex'].empty and {'sex', 'count_sex'}.issubset(filtered_data['sex'].columns):
        gender_data = (
            filtered_data['sex']
            .groupby('sex', as_index=False)['count_sex']
            .sum()
            .rename(columns={'sex': 'Gender', 'count_sex': 'Count'})
        )

        gender_mapping = {'L': 'Laki-laki', 'P': 'Perempuan', 'unknown': 'Tidak Diketahui'}
        gender_data['Gender'] = gender_data['Gender'].map(gender_mapping).fillna(gender_data['Gender'])

        fig_gender = px.pie(
            gender_data,
            values='Count',
            names='Gender',
            title='Distribusi Jenis Kelamin Pelaku',
            color_discrete_sequence=['#3182ce', '#4299e1', '#a0aec0']
        )
    else:
        fig_gender = px.pie(title='Distribusi Jenis Kelamin Pelaku - Tidak ada data')

    fig_gender.update_layout(
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # ----- Occupation Distribution -----
    if not filtered_data['occupation'].empty and {'occupation', 'count_occupation'}.issubset(filtered_data['occupation'].columns):
        occupation_data = (
            filtered_data['occupation']
            .groupby('occupation', as_index=False)['count_occupation']
            .sum()
            .sort_values('count_occupation', ascending=False)
            .head(10)
            .rename(columns={'occupation': 'Occupation', 'count_occupation': 'Count'})
        )

        occupation_data['Occupation'] = occupation_data['Occupation'].replace({'unknown': 'Tidak Diketahui'})

        fig_occupation = px.bar(
            occupation_data,
            x='Count',
            y='Occupation',
            orientation='h',
            title='Top 10 Pekerjaan Pelaku',
            labels={'Count': 'Jumlah Kasus', 'Occupation': 'Pekerjaan'},
            color='Count',
            color_continuous_scale='Blues'
        )
    else:
        fig_occupation = px.bar(title='Top 10 Pekerjaan Pelaku - Tidak ada data')

    fig_occupation.update_layout(
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig_age, fig_gender, fig_occupation

def create_motive_wordcloud(filtered_data):
    """Create word cloud and bar chart for crime motives"""
    if filtered_data['motive'].empty:
        return None, None
    
    motive_data = filtered_data['motive'].groupby('motive')['count_motive'].sum().sort_values(ascending=False)
    
    # Create bar chart as primary visualization
    motive_df = motive_data.head(10).reset_index()
    motive_df.columns = ['Motive', 'Count']
    
    fig_motive = px.bar(
        motive_df,
        x='Count',
        y='Motive',
        orientation='h',
        title='Top 5 Motif Kejahatan',
        labels={'Count': 'Jumlah Kasus', 'Motive': 'Motif'},
        color='Count',
        color_continuous_scale='Reds'
    )
    
    fig_motive.update_layout(
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    # Try to create word cloud
    wordcloud_fig = None
    try:
        # Create text from motives weighted by count
        motive_text = []
        for motive, count in motive_data.items():
            # Repeat each motive based on its frequency (scaled down)
            repeat_count = max(1, int(count / motive_data.sum() * 100))
            motive_text.extend([motive] * repeat_count)
        
        motive_text_str = ' '.join(motive_text)
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Reds',
            max_words=50,
            relative_scaling=0.5
        ).generate(motive_text_str)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        fig.patch.set_facecolor('black')
        wordcloud_fig = fig
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
    
    return fig_motive, wordcloud_fig

def main():
    # Load CSS
    load_css()
    
    # Load all data
    data = load_all_data()
    
    # Check if essential data is loaded
    if data['polda'].empty:
        st.error("Polda data is required but not found!")
        st.stop()
    
    # Get date range
    min_date, max_date = get_date_range(data)
    
    # Get crime types
    all_crime_types = []
    if not data['age'].empty:
        all_crime_types = sorted(data['age']['crime_type'].unique().tolist())
    
    # Get provinces
    provinces = sorted(data['polda']['province'].unique().tolist())
    
    # Initialize session state
    initialize_session_state(min_date, max_date, provinces, all_crime_types)
    
    # Create sidebar with filters
    applied_filters = create_filter_sidebar(min_date, max_date, data)
    
    # Apply filters using the applied (not current) filter values
    filtered_data = filter_data_by_date_and_location(
        data, 
        applied_filters['start_date'], 
        applied_filters['end_date'], 
        applied_filters['selected_province']
    )
    
    # Further filter by crime type
    if applied_filters['selected_crimes']:
        for key in ['age', 'sex', 'occupation', 'location', 'time', 'status', 'motive']:
            if not filtered_data[key].empty and 'crime_type' in filtered_data[key].columns:
                filtered_data[key] = filtered_data[key][filtered_data[key]['crime_type'].isin(applied_filters['selected_crimes'])]
    
    # Main dashboard
    st.title("🚔 Dashboard Data Kejahatan Indonesia 🚔 ")
    st.markdown("")
    # st.markdown(f"**Periode:** {start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}")
    # if selected_province != 'Semua':
    #     st.markdown(f"**Provinsi:** {selected_province}")
    # if selected_polda != 'Semua':
    #     st.markdown(f"**Polda:** {selected_polda}")
    # st.markdown("---")
    
    # Metrics cards
    create_metrics_cards(filtered_data)
    st.markdown("---")
    
    # Time series analysis
    st.markdown("## Analisis Tren Waktu")
    time_fig = create_time_series_chart(filtered_data)
    if time_fig:
        st.plotly_chart(time_fig, use_container_width=True)
    else:
        st.warning("Tidak ada data untuk menampilkan tren waktu")
    
    # Location distribution
    st.markdown("## Distribusi Kejahatan per Wilayah")
    
    # Determine whether to hide summary sections
    hide_summary = (applied_filters['selected_province'] != 'Semua')
    
    if hide_summary:
        # When province is selected, show only the map
        map_fig, _ = create_indonesia_crime_map(filtered_data, hide_summary=True)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk menampilkan peta")
    else:
        # When no province filter, show map + summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            map_fig, summary_table = create_indonesia_crime_map(filtered_data, hide_summary=False)
            if map_fig:
                st.plotly_chart(map_fig, use_container_width=True)
            else:
                st.warning("Tidak ada data untuk menampilkan peta")
        
        with col2:
            if summary_table is not None:
                st.markdown("### Top 10 Provinsi")
                st.dataframe(summary_table, use_container_width=True, hide_index=True)
                
                # Quick insights
                # st.markdown("### Insights Cepat")
                if not summary_table.empty:
                    highest_province = summary_table.iloc[0]
                    # st.info(f"**Provinsi dengan kasus terbanyak:** {highest_province['Provinsi']} ({highest_province['Total Kasus']} kasus)")
                    
                    avg_cases = summary_table['Total Kasus'].apply(lambda x: float(x.replace(',', ''))).mean()
                    st.info(f"**Rata-rata kasus per provinsi:** {avg_cases:,.0f}")
    
    # Demographics section
    st.markdown("## Analisis Demografi Pelaku")
    
    tab1, tab2 = st.tabs(["Distribusi Umum", "Analisis Pekerjaan"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        fig_age, fig_gender, fig_occupation = create_demographics_charts(filtered_data)
        
        with col1:
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_gender, use_container_width=True)
    
    with tab2:
        st.plotly_chart(fig_occupation, use_container_width=True)
        
        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not filtered_data['age'].empty:
                # Calculate weighted average age
                age_groups = filtered_data['age']['age'].unique()
                age_totals = filtered_data['age'].groupby('age')['count_age'].sum()
                dominant_age = age_totals.idxmax() if not age_totals.empty else "N/A"
                st.metric(
                    label="Kelompok Usia Dominan",
                    value=dominant_age
                )
            else:
                st.metric(label="Kelompok Usia Dominan", value="N/A")
        
        with col2:
            if not filtered_data['sex'].empty:
                gender_totals = filtered_data['sex'].groupby('sex')['count_sex'].sum()
                most_common_gender = gender_totals.idxmax()
                gender_mapping = {'L': 'Laki-laki', 'P': 'Perempuan', 'unknown': 'Tidak Diketahui'}
                gender_name = gender_mapping.get(most_common_gender, most_common_gender)
                gender_pct = (gender_totals.max() / gender_totals.sum() * 100)
                st.metric(
                    label="Jenis Kelamin Dominan",
                    value=gender_name,
                    delta=f"{gender_pct:.1f}%"
                )
            else:
                st.metric(label="Jenis Kelamin Dominan", value="N/A", delta="0%")
        
        with col3:
            if not filtered_data['occupation'].empty:
                job_totals = filtered_data['occupation'].groupby('occupation')['count_occupation'].sum()
                most_common_job = job_totals.idxmax()
                job_count = job_totals.max()
                st.metric(
                    label="Pekerjaan Terbanyak",
                    value=most_common_job[:15] + "..." if len(most_common_job) > 15 else most_common_job,
                    delta=f"{job_count:,.0f} kasus"
                )
            else:
                st.metric(label="Pekerjaan Terbanyak", value="N/A", delta="0 kasus")
    
    # Motive analysis
    st.markdown("## Analisis Motif Kejahatan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_motive, wordcloud_fig = create_motive_wordcloud(filtered_data)
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
        elif fig_motive:
            st.plotly_chart(fig_motive, use_container_width=True)
        else:
            st.warning("Tidak ada data motif untuk ditampilkan")
    
    with col2:
        if not filtered_data['motive'].empty:
            st.markdown("### Top 5 Motif")
            motive_counts = filtered_data['motive'].groupby('motive')['count_motive'].sum().sort_values(ascending=False).head(5)
            for i, (motive, count) in enumerate(motive_counts.items(), 1):
                st.write(f"{i}. **{motive}**: {count:,.0f} kasus")
        else:
            st.markdown("### Top 5 Motif")
            st.write("Tidak ada data motif tersedia")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Dashboard Data Kejahatan Indonesia | IF4061 Visualisasi Data
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()