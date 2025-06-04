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
import base64
from pathlib import Path

st.set_page_config(
    page_title="Indonesia Crime Data Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

CRIME_TYPE_MAPPING = {
    'Curanmor R-2': 'Curanmor R2',
    'Kejahatan Terkait Senjata Tajam (Sajam) / Premanisme': 'Premanisme',
    'Kekerasan Dalam Rumah Tangga': 'KDRT',
    'Manipulasi data autentik secara elektronik (ITE)': 'Manipulasi ITE',
    'Membahayakan Keamanan Umum Bagi Orang / Barang': 'Keamanan Umum',
    'Menjual/Edarkan Obat Keras/Bebas Tanpa Izin': 'Obat Ilegal',
    'Narkotika (Narkoba)': 'Narkoba',
    'Pemalsuan Surat Otentik': 'Pemalsuan',
    'Pencurian Biasa': 'Pencurian',
    'Pencurian Dengan Pemberatan (Curat)': 'Curat',
    'Pengancaman': 'Pengancaman',
    'Penganiayaan': 'Penganiayaan',
    'Pengerusakan': 'Pengerusakan',
    'Penggelapan': 'Penggelapan',
    'Penggelapan asal usul': 'Penggelapan',
    'Pengroyokan': 'Pengroyokan',
    'Penipuan / Perbuatan Curang': 'Penipuan',
    'Persetubuhan Terhadap Anak / Cabul Terhadap Anak': 'Pencabulan',
    'Tindak Pidana Dalam Perlindungan Anak': 'Pidana Anak'
}

def set_background_image():
    """Add background image using base64 encoding"""
    try:
        with open('police.png', "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            min-height: 100vh !important;
        }}
        
        .stApp > header {{
            background: transparent !important;
        }}
        
        .stApp > .main {{
            background: transparent !important;
        }}
        
        .stApp > .main > div {{
            background: transparent !important;
        }}
        
        .main .block-container {{
            background-color: rgba(13, 17, 23, 0.4) !important;
            border-radius: 15px;
            # padding: 2rem;
            # margin: 1rem;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }}
        
        .css-1d391kg, .css-1lcbmhc {{
            background-color: rgba(13, 17, 23, 0.5) !important;
            backdrop-filter: blur(5px);
        }}
        
        .stApp [data-testid="stSidebar"] {{
            background-color: rgba(13, 17, 23, 0.6) !important;
            backdrop-filter: blur(8px);
        }}
        
        .stApp [data-testid="stSidebar"] > div {{
            background-color: transparent !important;
        }}
        
        div[data-testid="metric-container"] {{
            background-color: rgba(13, 17, 23, 0.3) !important;
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Background image 'background.png' not found")


def get_short_crime_name(crime_type):
    return CRIME_TYPE_MAPPING.get(crime_type, crime_type)

def load_css():
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass 
    set_background_image()

@st.cache_data
def load_all_data():
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
    all_dates = []
    
    for key in ['age', 'sex', 'occupation', 'location', 'time', 'status', 'motive']:
        if not data[key].empty and 'year' in data[key].columns and 'month' in data[key].columns:
            df = data[key]

            df_dates = pd.to_datetime(df[['year', 'month']].assign(day=1))
            all_dates.extend(df_dates.tolist())
    
    if all_dates:
        return min(all_dates), max(all_dates)
    else:
        return datetime(2022, 1, 1), datetime(2024, 12, 1)

def filter_data_by_date_and_location(data, start_date, end_date, selected_province=None, selected_polda=None):
    filtered_data = {}
    
    for key in ['age', 'sex', 'occupation', 'location', 'time', 'status', 'motive']:
        if data[key].empty:
            filtered_data[key] = pd.DataFrame()
            continue
            
        df = data[key].copy()
        
        mask = (
            (df['year'] > start_date.year) | 
            ((df['year'] == start_date.year) & (df['month'] >= start_date.month))
        ) & (
            (df['year'] < end_date.year) | 
            ((df['year'] == end_date.year) & (df['month'] <= end_date.month))
        )
        
        df = df[mask]
        
        if selected_polda and selected_polda != 'Semua':
            df = df[df['polda'] == selected_polda]
        elif selected_province and selected_province != 'Semua':
            if not data['polda'].empty:
                province_poldas = data['polda'][data['polda']['province'] == selected_province]['polda'].tolist()
                df = df[df['polda'].isin(province_poldas)]
        
        filtered_data[key] = df
    
    filtered_data['polda'] = data['polda'].copy()
    if selected_province and selected_province != 'Semua':
        filtered_data['polda'] = data['polda'][data['polda']['province'] == selected_province]
    
    return filtered_data

def create_filter_sidebar(min_date, max_date, data):
    st.sidebar.title("Filter Data")
    
    def get_month_year_options(min_date, max_date):
        options = []
        current = min_date
        
        while current <= max_date:
            month_name = current.strftime("%b %Y")
            options.append((current.year, current.month, month_name))
            
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return options
    
    month_options = get_month_year_options(min_date, max_date)
    month_labels = [opt[2] for opt in month_options]

    applied_start = st.session_state.applied_filters['start_date']
    applied_end = st.session_state.applied_filters['end_date']
    
    start_idx = 0
    end_idx = len(month_options) - 1
    
    for i, (year, month, _) in enumerate(month_options):
        if year == applied_start.year and month == applied_start.month:
            start_idx = i
        if year == applied_end.year and month == applied_end.month:
            end_idx = i

    if len(month_options) > 1:
        idx_range = st.sidebar.select_slider(
            "Rentang Waktu Kejadian",
            options=list(range(len(month_options))),
            value=(start_idx, end_idx),
            format_func=lambda x: month_options[x][2],
            key="date_range_slider"
        )

        start_idx, end_idx = idx_range

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        start_year, start_month, _ = month_options[start_idx]
        end_year, end_month, _ = month_options[end_idx]

        current_start_date = datetime(start_year, start_month, 1)
        current_end_date = datetime(end_year, end_month, 1)
    else:
        current_start_date = min_date
        current_end_date = max_date
    
    provinces = ['Semua'] + sorted(data['polda']['province'].unique().tolist())
    current_selected_province = st.sidebar.selectbox(
        "Lokasi", 
        provinces,
        index=provinces.index(st.session_state.applied_filters['selected_province']) 
              if st.session_state.applied_filters['selected_province'] in provinces else 0,
        key="province_select"
    )
    
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
    
    apply_clicked = st.sidebar.button("Terapkan", use_container_width=True, type="secondary")
    
    if apply_clicked:
        st.session_state.applied_filters = {
            'start_date': current_start_date,
            'end_date': current_end_date,
            'selected_province': current_selected_province,
            'selected_crimes': current_selected_crimes
        }
        st.rerun()
    
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
    if np.all(values == values[0]):  
        return np.full(len(values), (min_size + max_size) / 2)
    
    if method == 'linear':
        min_val, max_val = values.min(), values.max()
        scaled = min_size + (values - min_val) * (max_size - min_size) / (max_val - min_val)
        
    elif method == 'sqrt':
        sqrt_values = np.sqrt(values)
        min_val, max_val = sqrt_values.min(), sqrt_values.max()
        scaled = min_size + (sqrt_values - min_val) * (max_size - min_size) / (max_val - min_val)
        
    elif method == 'log':
        log_values = np.log1p(values)  
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
    col1, col2, col3, col4 = st.columns(4)
    
    total_cases = filtered_data['age']['count_age'].sum() if not filtered_data['age'].empty else 0
    
    with col1:
        st.metric(
            label="Total Kasus",
            value=f"{total_cases:,.0f}",
            delta="Berdasarkan data terpilih",
            delta_color="off"
        )
    
    with col2:
        if not filtered_data['age'].empty:
            crime_totals = filtered_data['age'].groupby('crime_type')['count_age'].sum()
            most_common_crime = crime_totals.idxmax() if not crime_totals.empty else "N/A"
            crime_count = crime_totals.max() if not crime_totals.empty else 0
            
            short_crime_name = get_short_crime_name(most_common_crime)
            display_name = short_crime_name[:20] + "..." if len(short_crime_name) > 20 else short_crime_name
            
            st.metric(
                label="Kejahatan Terbanyak",
                value=display_name,
                delta=f"{crime_count:,.0f} kasus",
                delta_color="off"
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
                delta=f"{percentage:.1f}%",
                delta_color="off"
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
                delta=f"{resolved_cases:,.0f} kasus",
                delta_color="off"
            )
        else:
            st.metric(label="Tingkat Penyelesaian", value="0%", delta="0 kasus")

def create_time_series_chart(filtered_data):
    if filtered_data['age'].empty:
        return None
    
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
    if filtered_data['age'].empty or filtered_data['polda'].empty:
        st.warning("Tidak ada data untuk ditampilkan pada peta")
        return None, None
    
    polda_stats = filtered_data['age'].groupby('polda').agg({
        'count_age': 'sum'
    }).reset_index()
    polda_stats.columns = ['polda', 'total_cases']
    
    top_crimes_by_polda = {}
    for polda in polda_stats['polda'].unique():
        polda_crimes = filtered_data['age'][filtered_data['age']['polda'] == polda]
        crime_totals = polda_crimes.groupby('crime_type')['count_age'].sum().sort_values(ascending=False)
        top_3 = crime_totals.head(3)
        top_crimes_text = []
        for crime, count in top_3.items():
            percentage = (count / polda_crimes['count_age'].sum()) * 100
            short_name = get_short_crime_name(crime)
            top_crimes_text.append(f"• {short_name}: {count:,.0f} ({percentage:.1f}%)")
        top_crimes_by_polda[polda] = '<br>'.join(top_crimes_text)
    
    map_data = pd.merge(polda_stats, filtered_data['polda'], on='polda', how='left')
    map_data['top_3_crimes'] = map_data['polda'].map(top_crimes_by_polda)
    
    marker_sizes = scale_marker_sizes(
        map_data['total_cases'], 
        min_size=8, 
        max_size=50, 
        method='sqrt'
    )
    
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
                x=1,               
                y=0,                 
                xanchor="right",
                yanchor="bottom", 
                len=0.5,             
                thickness=10,        
                title=dict(
                    text="Kasus",
                    font=dict(color='white', size=10)
                ),
                tickfont=dict(color='white', size=8)
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
    
    fig.update_layout(
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(201, 189, 163)',  
            showocean=True,
            oceancolor='rgb(152, 180, 176)',
            showlakes=True,
            lakecolor='rgb(152, 180, 176)',
            showcountries=True,
            countrycolor='rgb(110, 110, 110)',
            center=dict(lat=-2.5, lon=118),
            lonaxis=dict(range=[95, 141]),
            lataxis=dict(range=[-11, 6]),
            bgcolor='rgba(245, 242, 236, 1)'  
        ),
        height=520,
        autosize=True,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=5, r=5, t=5, b=5)
    )
    
    summary_table = None
    if not hide_summary and 'province' in map_data.columns:
        province_summary = map_data.groupby('province')['total_cases'].sum().sort_values(ascending=False).reset_index()
        province_summary.columns = ['Provinsi', 'Total Kasus']
        province_summary['Total Kasus'] = province_summary['Total Kasus'].apply(lambda x: f"{x:,.0f}")
        summary_table = province_summary.head(10)
    
    return fig, summary_table



def create_demographics_charts(filtered_data):
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
    if filtered_data['motive'].empty:
        return None, None
    
    motive_data = filtered_data['motive'].groupby('motive')['count_motive'].sum().sort_values(ascending=False)
    
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
    
    wordcloud_fig = None
    try:
        motive_text = []
        for motive, count in motive_data.items():

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
    load_css()
    
    data = load_all_data()
    
    if data['polda'].empty:
        st.error("Polda data is required but not found!")
        st.stop()
    
    min_date, max_date = get_date_range(data)
    
    all_crime_types = []
    if not data['age'].empty:
        all_crime_types = sorted(data['age']['crime_type'].unique().tolist())
    
    provinces = sorted(data['polda']['province'].unique().tolist())
    
    initialize_session_state(min_date, max_date, provinces, all_crime_types)
    
    applied_filters = create_filter_sidebar(min_date, max_date, data)
    
    filtered_data = filter_data_by_date_and_location(
        data, 
        applied_filters['start_date'], 
        applied_filters['end_date'], 
        applied_filters['selected_province']
    )
    
    if applied_filters['selected_crimes']:
        for key in ['age', 'sex', 'occupation', 'location', 'time', 'status', 'motive']:
            if not filtered_data[key].empty and 'crime_type' in filtered_data[key].columns:
                filtered_data[key] = filtered_data[key][filtered_data[key]['crime_type'].isin(applied_filters['selected_crimes'])]
    
    st.title("🚔 Dashboard Data Kejahatan Indonesia 🚔 ")
    st.markdown("")
    
    create_metrics_cards(filtered_data)
    st.markdown("---")
    
    st.markdown("## Analisis Tren Waktu")
    time_fig = create_time_series_chart(filtered_data)
    if time_fig:
        st.plotly_chart(time_fig, use_container_width=True)
    else:
        st.warning("Tidak ada data untuk menampilkan tren waktu")
    
    st.markdown("## Distribusi Kejahatan per Wilayah")
    
    hide_summary = (applied_filters['selected_province'] != 'Semua')
    
    if hide_summary:
        map_fig, _ = create_indonesia_crime_map(filtered_data, hide_summary=True)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk menampilkan peta")
    else:
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
                
                if not summary_table.empty:
                    highest_province = summary_table.iloc[0]                    
                    avg_cases = summary_table['Total Kasus'].apply(lambda x: float(x.replace(',', ''))).mean()
                    st.info(f"**Rata-rata kasus per provinsi:** {avg_cases:,.0f}")
    
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not filtered_data['age'].empty:
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