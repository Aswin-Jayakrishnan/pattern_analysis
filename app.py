import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Attendance Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
    .attendance-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# === LOAD AND PROCESS DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("attendance_dataset.csv", parse_dates=["Date"])
    
    # Basic date features
    df['Day'] = df['Date'].dt.day_name()
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.month_name()
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    df['DayOfMonth'] = df['Date'].dt.day
    df['IsWeekend'] = df['Date'].dt.weekday >= 5
    
    # Time processing
    df['CheckIn'] = pd.to_datetime(df['CheckIn'], errors='coerce')
    df['CheckOut'] = pd.to_datetime(df['CheckOut'], errors='coerce')
    df['CheckInTime'] = df['CheckIn'].dt.time
    df['CheckOutTime'] = df['CheckOut'].dt.time
    df['CheckInHour'] = df['CheckIn'].dt.hour + df['CheckIn'].dt.minute / 60
    df['CheckOutHour'] = df['CheckOut'].dt.hour + df['CheckOut'].dt.minute / 60
    
    # Working hours calculation
    df['WorkingHours'] = (df['CheckOut'] - df['CheckIn']).dt.total_seconds() / 3600
    df['WorkingHours'] = df['WorkingHours'].fillna(0)
    
    # Attendance patterns
    late_threshold = datetime.strptime("10:15", "%H:%M").time()
    early_threshold = datetime.strptime("09:00", "%H:%M").time()
    df['IsLate'] = df['CheckInTime'] > late_threshold
    df['IsEarlyBird'] = df['CheckInTime'] < early_threshold
    df['IsOvertimeWorker'] = df['WorkingHours'] > 9
    df['IsShortDay'] = df['WorkingHours'] < 6
    
    # Productivity score (example calculation)
    df['ProductivityScore'] = np.where(
        df['Status'] == 'Present',
        np.clip(8 + (8 - df['CheckInHour']) + np.minimum(df['WorkingHours'] - 8, 2), 0, 10),
        np.where(df['Status'] == 'WFH', 7, 0)
    )
    
    return df

# Load data
try:
    df = load_data()
except FileNotFoundError:
    st.error("📄 Please upload 'attendance_dataset.csv' file")
    st.stop()

# === SIDEBAR CONFIGURATION ===
st.sidebar.markdown("## 🎛️ Dashboard Controls")

# Employee selection
employee_ids = sorted(df['EmployeeID'].unique())
selected_id = st.sidebar.selectbox("👤 Select Employee", ['All Employees'] + list(employee_ids))

# Date range selection
date_range = st.sidebar.date_input(
    "📅 Select Date Range",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Additional filters
status_filter = st.sidebar.multiselect(
    "📋 Filter by Status",
    options=df['Status'].unique(),
    default=df['Status'].unique()
)

# Performance metrics toggle
show_predictions = st.sidebar.toggle("🔮 Show Predictive Analytics", False)
show_detailed_stats = st.sidebar.toggle("📈 Show Detailed Statistics", True)

# === FILTER DATA ===
if len(date_range) == 2:
    data = df[(df['Date'] >= pd.to_datetime(date_range[0])) & 
              (df['Date'] <= pd.to_datetime(date_range[1]))]
else:
    data = df.copy()

data = data[data['Status'].isin(status_filter)]

if selected_id != 'All Employees':
    data = data[data['EmployeeID'] == selected_id]
    dashboard_title = f"📊 Attendance Dashboard - {selected_id}"
else:
    dashboard_title = "📊 Attendance Dashboard - All Employees"

# === HEADER ===
st.markdown(f"""
<div class="attendance-header">
    <h1>{dashboard_title}</h1>
    <p>Advanced Analytics & Insights | {len(data)} Records Analyzed</p>
</div>
""", unsafe_allow_html=True)

# === KEY METRICS ===
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    attendance_rate = (data['Status'].isin(['Present', 'WFH']).sum() / len(data) * 100) if len(data) > 0 else 0
    st.metric("📊 Attendance Rate", f"{attendance_rate:.1f}%")

with col2:
    late_rate = (data['IsLate'].sum() / len(data[data['Status'] == 'Present']) * 100) if len(data[data['Status'] == 'Present']) > 0 else 0
    st.metric("⏰ Late Arrival Rate", f"{late_rate:.1f}%")

with col3:
    avg_working_hours = data['WorkingHours'].mean() if len(data) > 0 else 0
    st.metric("⏱️ Avg Working Hours", f"{avg_working_hours:.1f}h")

with col4:
    wfh_rate = (data['Status'].eq('WFH').sum() / len(data) * 100) if len(data) > 0 else 0
    st.metric("🏠 WFH Rate", f"{wfh_rate:.1f}%")

with col5:
    avg_productivity = data['ProductivityScore'].mean() if len(data) > 0 else 0
    st.metric("⭐ Productivity Score", f"{avg_productivity:.1f}/10")

# === TABS FOR DIFFERENT ANALYSES ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", "📊 Time Analysis", "🔍 Patterns", "🎯 Performance", "🤖 ML Insights", "📑 Reports"
])

with tab1:
    # === OVERVIEW TAB ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Attendance Status Distribution")
        status_counts = data['Status'].value_counts()
        fig_pie = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Daily Attendance Trends")
        daily_attendance = data.groupby('Date')['Status'].value_counts().unstack(fill_value=0)
        fig_area = px.area(
            daily_attendance.reset_index(),
            x='Date',
            y=daily_attendance.columns,
            title="Daily Attendance Breakdown"
        )
        fig_area.update_layout(height=400)
        st.plotly_chart(fig_area, use_container_width=True)
    
    # Weekly heatmap
    st.subheader("🗓️ Weekly Attendance Heatmap")
    if len(data) > 0:
        status_numeric = {'Present': 4, 'WFH': 3, 'Leave': 2, 'Absent': 1}
        data['StatusNumeric'] = data['Status'].map(status_numeric)
        
        pivot_data = data.pivot_table(
            index='Week',
            columns='Day',
            values='StatusNumeric',
            aggfunc='mean'
        )
        
        # Reorder columns for proper day sequence
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex(columns=[col for col in day_order if col in pivot_data.columns])
        
        fig_heatmap = px.imshow(
            pivot_data,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            title="Attendance Pattern by Week and Day"
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

with tab2:
    # === TIME ANALYSIS TAB ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Check-in Time Distribution")
        fig_hist = px.histogram(
            data[data['CheckInHour'].notna()],
            x='CheckInHour',
            nbins=30,
            title="Check-in Time Distribution",
            color_discrete_sequence=['#636EFA']
        )
        fig_hist.add_vline(x=9.25, line_dash="dash", line_color="red", annotation_text="Late Threshold")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Working Hours Distribution")
        fig_working = px.histogram(
            data[data['WorkingHours'] > 0],
            x='WorkingHours',
            nbins=20,
            title="Working Hours Distribution",
            color_discrete_sequence=['#00CC96']
        )
        fig_working.add_vline(x=8, line_dash="dash", line_color="orange", annotation_text="Standard Hours")
        fig_working.update_layout(height=400)
        st.plotly_chart(fig_working, use_container_width=True)
    
    # Time trends
    st.subheader("📈 Time Trends Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monthly_late = data.groupby('MonthName')['IsLate'].mean() * 100
        fig_monthly = px.bar(
            x=monthly_late.index,
            y=monthly_late.values,
            title="Late Arrivals by Month (%)",
            color=monthly_late.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        daily_avg_hours = data.groupby('Day')['WorkingHours'].mean()
        fig_daily_hours = px.bar(
            x=daily_avg_hours.index,
            y=daily_avg_hours.values,
            title="Average Working Hours by Day",
            color=daily_avg_hours.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_daily_hours, use_container_width=True)
    
    with col3:
        overtime_by_day = data.groupby('Day')['IsOvertimeWorker'].mean() * 100
        fig_overtime = px.bar(
            x=overtime_by_day.index,
            y=overtime_by_day.values,
            title="Overtime Workers by Day (%)",
            color=overtime_by_day.values,
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_overtime, use_container_width=True)

with tab3:
    # === PATTERNS TAB ===
    st.subheader("🔍 Behavioral Patterns Analysis")
    
    if selected_id != 'All Employees' and len(data) > 10:
        # Individual patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Weekly Pattern Summary**")
            weekly_stats = data.groupby('Week').agg({
                'IsLate': 'sum',
                'WorkingHours': 'mean',
                'ProductivityScore': 'mean',
                'Status': lambda x: (x == 'Present').sum()
            }).round(2)
            weekly_stats.columns = ['Late Days', 'Avg Hours', 'Productivity', 'Present Days']
            st.dataframe(weekly_stats)
        
        with col2:
            st.write("**🎯 Performance Metrics**")
            metrics_summary = {
                'Total Days': len(data),
                'Present Days': (data['Status'] == 'Present').sum(),
                'WFH Days': (data['Status'] == 'WFH').sum(),
                'Late Arrivals': data['IsLate'].sum(),
                'Early Bird Days': data['IsEarlyBird'].sum(),
                'Overtime Days': data['IsOvertimeWorker'].sum(),
                'Avg Productivity': data['ProductivityScore'].mean()
            }
            for key, value in metrics_summary.items():
                if isinstance(value, float):
                    st.write(f"**{key}:** {value:.2f}")
                else:
                    st.write(f"**{key}:** {value}")
    
    # Pattern correlation analysis
    st.subheader("🔗 Correlation Analysis")
    if len(data) > 0:
        correlation_data = data[['IsLate', 'IsEarlyBird', 'IsOvertimeWorker', 'WorkingHours', 'ProductivityScore']].corr()
        fig_corr = px.imshow(
            correlation_data,
            text_auto=True,
            color_continuous_scale='RdBu',
            title="Behavioral Pattern Correlations"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    # === PERFORMANCE TAB ===
    st.subheader("🎯 Performance Dashboard")
    
    if show_detailed_stats and len(data) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**⭐ Top Performers**")
            if selected_id == 'All Employees':
                top_performers = data.groupby('EmployeeID')['ProductivityScore'].mean().sort_values(ascending=False).head(5)
                for emp, score in top_performers.items():
                    st.write(f"👤 {emp}: {score:.2f}/10")
            else:
                st.write(f"Individual Score: {data['ProductivityScore'].mean():.2f}/10")
        
        with col2:
            st.write("**📈 Consistency Metrics**")
            consistency_score = 100 - (data['ProductivityScore'].std() * 10) if data['ProductivityScore'].std() > 0 else 100
            st.write(f"Consistency: {max(0, consistency_score):.1f}%")
            
            punctuality = (1 - data['IsLate'].mean()) * 100 if len(data) > 0 else 0
            st.write(f"Punctuality: {punctuality:.1f}%")
            
            reliability = data['Status'].isin(['Present', 'WFH']).mean() * 100 if len(data) > 0 else 0
            st.write(f"Reliability: {reliability:.1f}%")
        
        with col3:
            st.write("**🏆 Achievements**")
            achievements = []
            if data['IsEarlyBird'].sum() > 5:
                achievements.append("🌅 Early Bird")
            if data['IsOvertimeWorker'].sum() > 10:
                achievements.append("💪 Dedicated Worker")
            if data['IsLate'].sum() < 3:
                achievements.append("⏰ Punctual Star")
            if data['ProductivityScore'].mean() > 8:
                achievements.append("⭐ High Performer")
            
            for achievement in achievements:
                st.write(achievement)
    
    # Performance trends
    if len(data) > 0:
        st.subheader("📊 Performance Trends")
        
        # Monthly performance
        monthly_performance = data.groupby('MonthName').agg({
            'ProductivityScore': 'mean',
            'WorkingHours': 'mean',
            'IsLate': lambda x: (1 - x.mean()) * 10  # Punctuality score
        }).round(2)
        
        fig_performance = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Productivity Score', 'Working Hours', 'Punctuality Score')
        )
        
        fig_performance.add_trace(
            go.Bar(x=monthly_performance.index, y=monthly_performance['ProductivityScore'], name="Productivity"),
            row=1, col=1
        )
        fig_performance.add_trace(
            go.Bar(x=monthly_performance.index, y=monthly_performance['WorkingHours'], name="Hours"),
            row=1, col=2
        )
        fig_performance.add_trace(
            go.Bar(x=monthly_performance.index, y=monthly_performance['IsLate'], name="Punctuality"),
            row=1, col=3
        )
        
        fig_performance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_performance, use_container_width=True)

with tab5:
    # === ML INSIGHTS TAB ===
    st.subheader("🤖 Machine Learning Insights")
    
    if selected_id != 'All Employees' and len(data) > 20:
        # Apriori Analysis
        st.write("**🔍 Behavioral Pattern Mining**")
        
        def get_behavior_tag(row):
            tags = []
            if row['Status'] == 'Leave':
                tags.append(f"Leave_{row['Day']}")
            elif row['Status'] == 'WFH':
                tags.append(f"WFH_{row['Day']}")
            elif row['Status'] == 'Absent':
                tags.append(f"Absent_{row['Day']}")
            elif row['IsLate']:
                tags.append(f"Late_{row['Day']}")
            else:
                tags.append(f"Present_{row['Day']}")
            
            if row['IsOvertimeWorker']:
                tags.append("Overtime")
            if row['IsEarlyBird']:
                tags.append("EarlyBird")
            
            return tags
        
        try:
            data['BehaviorTags'] = data.apply(get_behavior_tag, axis=1)
            transactions = data.groupby('Week')['BehaviorTags'].apply(lambda x: [item for sublist in x for item in sublist]).tolist()
            
            if len(transactions) > 5:
                te = TransactionEncoder()
                te_data = te.fit(transactions).transform(transactions)
                trans_df = pd.DataFrame(te_data, columns=te.columns_)
                
                frequent_items = apriori(trans_df, min_support=0.1, use_colnames=True)
                
                if len(frequent_items) > 0:
                    rules = association_rules(frequent_items, metric='confidence', min_threshold=0.5)
                    
                    if len(rules) > 0:
                        st.write("**📋 Top Behavioral Rules**")
                        top_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='confidence', ascending=False).head(10)
                        st.dataframe(top_rules)
                    else:
                        st.info("No significant behavioral patterns found with current thresholds.")
                else:
                    st.info("Insufficient data for pattern mining.")
        except Exception as e:
            st.error(f"Pattern analysis failed: {str(e)}")
    
    # Predictive indicators
    if show_predictions and len(data) > 0:
        st.subheader("🔮 Predictive Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Risk Indicators**")
            recent_data = data.tail(10)  # Last 10 records
            
            late_trend = recent_data['IsLate'].mean()
            if late_trend > 0.3:
                st.error(f"🚨 High late arrival risk: {late_trend*100:.1f}%")
            elif late_trend > 0.1:
                st.warning(f"⚠️ Moderate late arrival risk: {late_trend*100:.1f}%")
            else:
                st.success(f"✅ Low late arrival risk: {late_trend*100:.1f}%")
            
            productivity_trend = recent_data['ProductivityScore'].mean()
            if productivity_trend < 6:
                st.error(f"📉 Declining productivity: {productivity_trend:.1f}/10")
            elif productivity_trend < 8:
                st.warning(f"📊 Moderate productivity: {productivity_trend:.1f}/10")
            else:
                st.success(f"📈 High productivity: {productivity_trend:.1f}/10")
        
        with col2:
            st.write("**📈 Trend Analysis**")
            if len(data) >= 4:  # Need at least 4 weeks
                recent_weeks = data.groupby('Week').agg({
                    'ProductivityScore': 'mean',
                    'IsLate': 'mean',
                    'WorkingHours': 'mean'
                }).tail(4)
                
                # Simple trend calculation
                productivity_slope = np.polyfit(range(len(recent_weeks)), recent_weeks['ProductivityScore'], 1)[0]
                if productivity_slope > 0.1:
                    st.success("📈 Productivity trending up")
                elif productivity_slope < -0.1:
                    st.error("📉 Productivity trending down")
                else:
                    st.info("📊 Productivity stable")

with tab6:
    # === REPORTS TAB ===
    st.subheader("📑 Detailed Reports")
    
    # Summary report
    st.write("**📊 Executive Summary**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Attendance Overview**")
        total_days = len(data)
        present_days = (data['Status'] == 'Present').sum()
        wfh_days = (data['Status'] == 'WFH').sum()
        leave_days = (data['Status'] == 'Leave').sum()
        absent_days = (data['Status'] == 'Absent').sum()
        
        st.write(f"• Total Days: {total_days}")
        st.write(f"• Present: {present_days} ({present_days/total_days*100:.1f}%)")
        st.write(f"• Work from Home: {wfh_days} ({wfh_days/total_days*100:.1f}%)")
        st.write(f"• Leave: {leave_days} ({leave_days/total_days*100:.1f}%)")
        st.write(f"• Absent: {absent_days} ({absent_days/total_days*100:.1f}%)")
    
    with col2:
        st.write("**Time Management**")
        avg_checkin = data['CheckInHour'].mean() if data['CheckInHour'].notna().any() else 0
        avg_checkout = data['CheckOutHour'].mean() if data['CheckOutHour'].notna().any() else 0
        late_count = data['IsLate'].sum()
        overtime_count = data['IsOvertimeWorker'].sum()
        
        st.write(f"• Avg Check-in: {avg_checkin:.2f}h")
        st.write(f"• Avg Check-out: {avg_checkout:.2f}h")
        st.write(f"• Late Arrivals: {late_count}")
        st.write(f"• Overtime Days: {overtime_count}")
    
    with col3:
        st.write("**Performance Metrics**")
        avg_productivity = data['ProductivityScore'].mean()
        avg_working_hours = data['WorkingHours'].mean()
        consistency = 100 - (data['ProductivityScore'].std() * 10) if data['ProductivityScore'].std() > 0 else 100
        
        st.write(f"• Avg Productivity: {avg_productivity:.2f}/10")
        st.write(f"• Avg Working Hours: {avg_working_hours:.2f}h")
        st.write(f"• Consistency Score: {max(0, consistency):.1f}%")
    
    # Detailed data table
    st.subheader("📋 Detailed Attendance Records")
    
    # Add filters for the table
    col1, col2, col3 = st.columns(3)
    with col1:
        table_status_filter = st.selectbox("Filter by Status", ['All'] + list(data['Status'].unique()))
    with col2:
        table_month_filter = st.selectbox("Filter by Month", ['All'] + list(data['MonthName'].unique()))
    with col3:
        show_late_only = st.checkbox("Show Late Arrivals Only")
    
    # Apply table filters
    table_data = data.copy()
    if table_status_filter != 'All':
        table_data = table_data[table_data['Status'] == table_status_filter]
    if table_month_filter != 'All':
        table_data = table_data[table_data['MonthName'] == table_month_filter]
    if show_late_only:
        table_data = table_data[table_data['IsLate']]
    
    # Display table
    display_columns = ['Date', 'EmployeeID', 'Status', 'CheckInTime', 'CheckOutTime', 
                      'WorkingHours', 'IsLate', 'ProductivityScore']
    
    if len(table_data) > 0:
        st.dataframe(
            table_data[display_columns].sort_values('Date', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv_data = table_data[display_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv_data,
            file_name=f"attendance_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No data matches the selected filters.")

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📊 Professional Attendance Analytics Dashboard | 
    Powered by Streamlit & Advanced Analytics | 
    Last Updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>
""", unsafe_allow_html=True)