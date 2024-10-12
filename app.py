import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Remote Work and Mental Health Data Visualization")

@st.cache_data
def load_data():
    df = pd.read_csv('remote_work_data.csv')
    return df

data = load_data()

data['Stress_Level'] = data['Stress_Level'].str.capitalize()

st.header("1. Stress Levels by Region (Heatmap)")

if 'Region' in data.columns and 'Stress_Level' in data.columns:
    region_data = data.groupby('Region')['Stress_Level'].apply(lambda x: (x == 'High').mean()).reset_index()
    region_data.rename(columns={'Stress_Level': 'high_stress_proportion'}, inplace=True)

    region_coords = {
        'Europe': {'lat': 54.5260, 'lon': 15.2551},
        'Asia': {'lat': 34.0479, 'lon': 100.6197},
        'North America': {'lat': 54.5260, 'lon': -105.2551},
        'South America': {'lat': -14.2350, 'lon': -51.9253},
        'Africa': {'lat': -8.7832, 'lon': 34.5085},
        'Australia': {'lat': -25.2744, 'lon': 133.7751},
        'Oceania': {'lat': -22.7359, 'lon': 140.0188},
        'Antarctica': {'lat': -75.250973, 'lon': 0.071389}
    }

    region_data['lat'] = region_data['Region'].map(lambda region: region_coords.get(region, {}).get('lat', None))
    region_data['lon'] = region_data['Region'].map(lambda region: region_coords.get(region, {}).get('lon', None))

    region_data = region_data.dropna(subset=['lat', 'lon'])

    fig_map = px.scatter_geo(
        region_data,
        lat='lat',
        lon='lon',
        size='high_stress_proportion',
        color='high_stress_proportion',
        hover_name='Region',
        color_continuous_scale="Viridis",
        title="Proportion of High Stress Levels by Region",
        size_max=50,
        labels={'high_stress_proportion': 'Proportion of High Stress'}
    )

    fig_map.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        margin={"r":0,"t":50,"l":0,"b":0}
    )

    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("The dataset must contain 'Region' and 'Stress_Level' columns for the map visualization.")

st.header("2. Stress Levels by Selected Category")

category_options = {
    'Age Group': 'Age_Group',
    'Gender': 'Gender',
    'Job Role': 'Job_Role',
    'Industry': 'Industry'
}

bar_chart_category_display = st.selectbox(
    "Select Category to Visualize Stress Levels",
    options=list(category_options.keys()),
    index=0,
    help="Choose the category to visualize stress levels."
)

bar_chart_category = category_options[bar_chart_category_display]

if bar_chart_category == 'Age_Group':
    # Define age groups
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

bar_data = data.groupby([bar_chart_category, 'Stress_Level']).size().reset_index(name='count')

bar_data['proportion'] = bar_data.groupby(bar_chart_category)['count'].transform(lambda x: x / x.sum())
bar_data['percentage'] = bar_data['proportion'] * 100

fig_bar = px.bar(
    bar_data,
    x=bar_chart_category,
    y='proportion',
    color='Stress_Level',
    barmode='group',
    color_discrete_map={'Low': '#90EE90', 'Medium': '#FFA07A', 'High': '#F08080'},
    title=f'Stress Level Distribution by {bar_chart_category_display}',
    labels={'proportion': 'Proportion', bar_chart_category: bar_chart_category_display},
    height=600,
    text=bar_data['percentage'].round(1),
    category_orders={  
        bar_chart_category: sorted(data[bar_chart_category].dropna().unique())
    }
)

fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')

fig_bar.update_layout(
    xaxis_title=bar_chart_category_display,
    yaxis_title='Proportion',
    legend_title='Stress Level',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12)
)

st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("""
**Note on Proportion Calculation:**
The proportion for each stress level within a category is calculated as follows:
1. Count the number of individuals in each stress level for each category.
2. Divide the count for each stress level by the total count for that category.
3. This gives a proportion value between 0 and 1, which is then converted to a percentage.

For example, if there are 100 people in the '26-35' age group, and 20 of them report high stress,
the proportion for high stress in this age group would be 20/100 = 0.2, or 20%.
""")

st.header("3. Correlation Heatmap of Numerical Variables")

numerical_cols = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 
                 'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating', 
                 'Social_Isolation_Rating', 'Satisfaction_with_Remote_Work',
                 'Company_Support_for_Remote_Work', 'Physical_Activity', 'Sleep_Quality']

numerical_data = data[numerical_cols].apply(pd.to_numeric, errors='coerce')

corr_matrix = numerical_data.corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu',
    title='Correlation Heatmap',
    labels=dict(color="Correlation"),
)

fig_corr.update_layout(
    margin={"r":20,"t":50,"l":20,"b":20}
)

st.plotly_chart(fig_corr, use_container_width=True)

st.header("4. Relationship Between Work Hours and Stress Level")

hours_stress_data = data.groupby('Hours_Worked_Per_Week')['Stress_Level'].value_counts(normalize=True).unstack()
hours_stress_data = hours_stress_data.reset_index()

hours_stress_data_melted = hours_stress_data.melt(
    id_vars=['Hours_Worked_Per_Week'],
    value_vars=['Low', 'Medium', 'High'],
    var_name='Stress_Level',
    value_name='Proportion'
)

hours_stress_data_melted['Percentage'] = hours_stress_data_melted['Proportion'] * 100

fig_line_work_hours = px.line(
    hours_stress_data_melted,
    x='Hours_Worked_Per_Week',
    y='Proportion',
    color='Stress_Level',
    title='Stress Level Distribution by Hours Worked per Week',
    labels={
        'Proportion': 'Proportion',
        'Hours_Worked_Per_Week': 'Hours Worked per Week',
        'Stress_Level': 'Stress Level',
        'Percentage': 'Percentage'
    },
    color_discrete_map={'Low': '#90EE90', 'Medium': '#FFA07A', 'High': '#F08080'},
    category_orders={  
        'Stress_Level': ['Low', 'Medium', 'High']
    }
)

fig_line_work_hours.update_layout(
    xaxis_title='Hours Worked per Week',
    yaxis_title='Proportion',
    legend_title='Stress Level',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12)
)

st.plotly_chart(fig_line_work_hours, use_container_width=True)

st.markdown("""
**Note on Proportion Calculation for Work Hours:**
The proportion for each stress level at each hour point is calculated as follows:
1. Group the data by 'Hours Worked Per Week'.
2. For each hour point, calculate the proportion of each stress level.
3. The proportions for Low, Medium, and High stress at each hour point sum to 1 (or 100%).

This allows us to see how the distribution of stress levels changes as the number of hours worked increases.
""")

st.header("5. Stress Level Distribution by Industry")

industry_data = data.groupby(['Industry', 'Stress_Level']).size().reset_index(name='count')
industry_data['proportion'] = industry_data.groupby('Industry')['count'].transform(lambda x: x / x.sum())
industry_data['percentage'] = industry_data['proportion'] * 100

fig_industry = px.bar(
    industry_data,
    x='Industry',
    y='proportion',
    color='Stress_Level',
    barmode='group',
    color_discrete_map={'Low': '#90EE90', 'Medium': '#FFA07A', 'High': '#F08080'},
    title='Stress Level Distribution by Industry',
    labels={'proportion': 'Proportion', 'Industry': 'Industry'},
    height=600,
    text=industry_data['percentage'].round(1),
    category_orders={  
        'Industry': sorted(data['Industry'].dropna().unique())
    }
)

fig_industry.update_traces(texttemplate='%{text}%', textposition='outside')

fig_industry.update_layout(
    xaxis_title='Industry',
    yaxis_title='Proportion',
    legend_title='Stress Level',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12)
)

st.plotly_chart(fig_industry, use_container_width=True)

st.markdown("""
**Note on Proportion Calculation for Industry:**
The proportion for each stress level within an industry is calculated as follows:
1. Count the number of individuals in each stress level for each industry.
2. Divide the count for each stress level by the total count for that industry.
3. This gives a proportion value between 0 and 1, which is then converted to a percentage.

For example, if there are 200 people in the 'Technology' industry, and 60 of them report high stress,
the proportion for high stress in this industry would be 60/200 = 0.3, or 30%.
""")

st.header("6. Years of Experience vs. Satisfaction with Remote Work")

fig_boxplot = px.box(
    data,
    x='Years_of_Experience',
    y='Satisfaction_with_Remote_Work',
    title='Satisfaction with Remote Work by Years of Experience',
    labels={
        'Years_of_Experience': 'Years of Experience',
        'Satisfaction_with_Remote_Work': 'Satisfaction with Remote Work'
    }
)

fig_boxplot.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12)
)

st.plotly_chart(fig_boxplot, use_container_width=True)

st.header("7. Work Location vs. Productivity Change")

# Convert to Plotly Bar Chart
productivity_counts = data.groupby(['Work_Location', 'Productivity_Change']).size().reset_index(name='count')

fig_barplot = px.bar(
    productivity_counts,
    x='Work_Location',
    y='count',
    color='Productivity_Change',
    barmode='group',
    title='Work Location vs. Productivity Change',
    labels={
        'Work_Location': 'Work Location',
        'count': 'Count',
        'Productivity_Change': 'Productivity Change'
    },
    color_discrete_map={
        'Increased': 'lightgreen',
        'Decreased': 'salmon',
        'No Change': 'lightblue'
    },
    category_orders={'Productivity_Change': ['Increased', 'No Change', 'Decreased']}  # Corrected parameter name
)

fig_barplot.update_layout(
    xaxis_title='Work Location',
    yaxis_title='Count',
    legend_title='Productivity Change',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12)
)

st.plotly_chart(fig_barplot, use_container_width=True)

st.header("8. Work-Life Balance Rating Distribution")

# Convert to Plotly Bar Chart
work_life_balance_counts = data['Work_Life_Balance_Rating'].value_counts().reset_index()
work_life_balance_counts.columns = ['Work_Life_Balance_Rating', 'Count']

fig_countplot = px.bar(
    work_life_balance_counts,
    x='Work_Life_Balance_Rating',
    y='Count',
    title='Work-Life Balance Rating',
    labels={
        'Work_Life_Balance_Rating': 'Work-Life Balance Rating',
        'Count': 'Count'
    },
    color='Work_Life_Balance_Rating',
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig_countplot.update_layout(
    xaxis_title='Work-Life Balance Rating',
    yaxis_title='Count',
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12)
)

st.plotly_chart(fig_countplot, use_container_width=True)

st.header("9. Mental Health Condition Distribution")

mental_health_category = st.selectbox(
    "Select Category for Mental Health Condition Distribution",
    options=list(category_options.keys()),
    index=0,
    help="Choose the category to visualize mental health condition distribution."
)

mental_health_column = category_options[mental_health_category]

if mental_health_column == 'Age_Group':
    if 'Age_Group' not in data.columns:
        bins = [18, 25, 35, 45, 55, 65, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
        data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

mental_health_data = data.groupby([mental_health_column, 'Mental_Health_Condition']).size().reset_index(name='count')

fig_mental_health = px.bar(
    mental_health_data,
    x=mental_health_column,
    y='count',
    color='Mental_Health_Condition',
    barmode='group',
    title=f'Mental Health Condition by {mental_health_category}',
    labels={
        mental_health_column: mental_health_category,
        'count': 'Count',
        'Mental_Health_Condition': 'Mental Health Condition'
    },
    color_discrete_sequence=px.colors.qualitative.Set3,
    category_orders={
        'Mental_Health_Condition': sorted(data['Mental_Health_Condition'].dropna().unique())
    }
)

fig_mental_health.update_layout(
    xaxis_title=mental_health_category,
    yaxis_title='Count',
    legend_title='Mental Health Condition',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12)
)

st.plotly_chart(fig_mental_health, use_container_width=True)

st.header("10. Access to Mental Health Resources")

access_counts = data['Access_to_Mental_Health_Resources'].value_counts().reset_index()
access_counts.columns = ['Access_to_Mental_Health_Resources', 'Count']

fig_pie = px.pie(
    access_counts,
    names='Access_to_Mental_Health_Resources',
    values='Count',
    title='Access to Mental Health Resources',
    color='Access_to_Mental_Health_Resources',
    color_discrete_map={
        'Yes': 'lightgreen',
        'No': 'salmon'
    },
    hole=0.4
)

fig_pie.update_traces(textposition='inside', textinfo='percent+label')

fig_pie.update_layout(
    legend_title='Access to Mental Health Resources',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12)
)

st.plotly_chart(fig_pie, use_container_width=True)
