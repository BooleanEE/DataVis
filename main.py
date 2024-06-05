import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Data pre-processing

life_expectancy_at_0_45_60_all_df = pd.read_csv("./data/life_expectancy_specific_ages_all.csv")

life_expectancy_at_0_45_60_all_df["Value"] = life_expectancy_at_0_45_60_all_df["Value"].str.replace(',', '.').astype(float)

# Group by 'Time', 'Age', and 'Sex', and rank countries based on 'Value' (life expectancy)
life_expectancy_at_0_45_60_all_df['Rank'] = life_expectancy_at_0_45_60_all_df.groupby(['Time', 'Age', 'Sex'])['Value'].rank(ascending=False, method='min')

# Convert ranks to integers
life_expectancy_at_0_45_60_all_df['Rank'] = life_expectancy_at_0_45_60_all_df['Rank'].astype(int)

# Clean the 'Sex' column to contain only the expected values
expected_sexes = ['Female', 'Male', 'Both sexes']
life_expectancy_at_0_45_60_all_df = life_expectancy_at_0_45_60_all_df[life_expectancy_at_0_45_60_all_df['Sex'].isin(expected_sexes)]


df = life_expectancy_at_0_45_60_all_df

# Get unique location, sexes and specific ages
ages = df['Age'].unique()

locations = df['Location'].unique()
location_highest_length = max(df['Location'].unique(), key=len)
char_pad_digits_rank = 3
highest_char_count = len(location_highest_length) + char_pad_digits_rank
sexes = ["Female", "Male", "Both sexes"]

# Custom shade of black (light black)
light_dark_blue = 'rgba(0, 51, 102, 0.1)'  # Adjust the alpha value for lightness
dark_blue = 'rgb(0, 51, 102)'

# Streamlit app
def main():
        
    st.title('Life Expectancy - Ranking Time Series Visualization')

    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #0d214d;
        }
        .st-emotion-cache-bm2z3a {
            padding-right: 30rem;
        }
        .st-emotion-cache-1pbsqtx {
            color: white;
        }
        .st-emotion-cache-1jmvea6 {
            color: white;
        }
        .st-emotion-cache-bm2z3a {
            padding-right: 1rem;
        }
        .st-emotion-cache-13ln4jf {
            max-width: 100rem;
        }
        .st-emotion-cache-13ln4jf {
            padding-left: 10rem;
            padding-right: 1rem;
        }
        h2 {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with multi-select dropdown menu
    st.sidebar.header('Ranking Time Series')
    selected_age_ts = st.sidebar.selectbox('Select the specific age:', ages, key='age_ts')
    selected_countries = st.sidebar.multiselect('Select a country or countries:', locations, default=None, key='countries')
    selected_sex = st.sidebar.selectbox('Select the sex:', sexes, key='sex')

    st.sidebar.header('Ranking Table')
    selected_age_table = st.sidebar.selectbox('Select the specific age:', ages, key='age_table')
    selected_year = st.sidebar.selectbox('Select the specific year:', list(range(1955, 2024)), key='year')

    # Filter dataframe based on selections for time series
    filtered_df_ts = life_expectancy_at_0_45_60_all_df[
        (life_expectancy_at_0_45_60_all_df['Sex'] == selected_sex) &
        (life_expectancy_at_0_45_60_all_df['Age'] == selected_age_ts)
    ]

    # Plot time series chart
    fig = px.line(filtered_df_ts, x='Time', y='Value', color='Location',
                  title=f'Life expectancy at age {selected_age_ts} for {selected_sex}',
                  labels={'Time': 'Year', 'Value': 'Years expected to live at the specific age', 'Location': 'Country'},
                  width=1100, height=800)
    
    # Set line color to custom shade of dark blue for all lines
    fig.update_traces(line=dict(color=light_dark_blue))
    
    # If a specific country is selected, change its line color to normal dark blue
    for country in selected_countries:
        fig.for_each_trace(lambda trace: trace.update(line=dict(color=dark_blue)) if trace.name == country else ())

    # Remove legend
    fig.update_layout(showlegend=False)

    # Set x-axis range to end at 2023
    fig.update_layout(xaxis=dict(range=[filtered_df_ts['Time'].min(), filtered_df_ts['Time'].max()]))

    # Change margin
    fig.update_layout(margin=dict(r=225))

    # Add annotations for selected countries
    years_to_annotate = list(range(1950, 2021, 10))  # Annotate every decade from 1950 to 2020
    if selected_countries:
        for country in selected_countries:
            country_df = filtered_df_ts[filtered_df_ts['Location'] == country]
            if not country_df.empty:
                for year in years_to_annotate:
                    if year in country_df['Time'].values:
                        year_data = country_df[country_df['Time'] == year].iloc[0]
                        latest_rank = str(year_data['Rank'])
                        
                        fig.add_annotation(x=year, y=year_data['Value'],
                                           text=latest_rank, showarrow=False,
                                           yshift=5,
                                           xshift=2,
                                           font=dict(family="Courier New, monospace", size=10, color=dark_blue))
                                            
    
    # Show the selected country name at the end of the time series
    if selected_countries:
            for country in selected_countries:
                country_df = filtered_df_ts[filtered_df_ts['Location'] == country]
                if not country_df.empty:
                    max_time = country_df['Time'].max()
                    latest_value = country_df[country_df['Time'] == max_time]['Value'].values[0]
                    latest_rank = str(country_df[country_df['Time'] == max_time]['Rank'].values[0])
                    char_sum_value_with_rank = len(latest_rank) + len(country)
            
                    if char_sum_value_with_rank != highest_char_count:
                        padding = " " * (highest_char_count - char_sum_value_with_rank)
                        text_annotation = f"{latest_rank} {country}{padding}"
                    else:
                        text_annotation = f"{latest_rank} {country}"

                    fig.add_annotation(x=max_time, y=float(latest_value),
                                    text=text_annotation, showarrow=False,
                                    xshift=117,
                                    font=dict(family="Courier New, monospace", size=10, color=dark_blue))
                                            

    st.plotly_chart(fig)

    # Title for the table
    st.title('Life Expectancy - Ranking Table')

    # Calculate the year 5 years before the selected year
    previous_year = selected_year - 5

    # Filter data for selected year and previous year
    current_df = df[(df['Time'] == selected_year) & (df['Age'] == selected_age_table)]
    previous_df = df[(df['Time'] == previous_year) & (df['Age'] == selected_age_table)]

    # Merge current and previous dataframes to compute rank changes
    merge_df = pd.merge(current_df, previous_df, on=['Location', 'Age', 'Sex'], suffixes=('', '_prev'))

    # Function to get rank change symbol
    def get_change_symbol(change):
        if change > 0:
            return f"▲{change}"
        elif change < 0:
            return f"▼{abs(change)}"
        else:
            return "="

    # Prepare data for the table
    male_data = []
    female_data = []
    for location in current_df['Location'].unique():
        male_current = merge_df[(merge_df['Location'] == location) & (merge_df['Sex'] == 'Male')]
        female_current = merge_df[(merge_df['Location'] == location) & (merge_df['Sex'] == 'Female')]

        male_rank = male_current['Rank'].values[0] if not male_current.empty else None
        male_value = round(male_current['Value'].values[0], 2) if not male_current.empty else None
        male_rank_prev = male_current['Rank_prev'].values[0] if not male_current.empty else None
        male_change = get_change_symbol(male_rank_prev - male_rank) if male_rank and male_rank_prev else None

        female_rank = female_current['Rank'].values[0] if not female_current.empty else None
        female_value = round(female_current['Value'].values[0], 2) if not female_current.empty else None
        female_rank_prev = female_current['Rank_prev'].values[0] if not female_current.empty else None
        female_change = get_change_symbol(female_rank_prev - female_rank) if female_rank and female_rank_prev else None

        male_data.append([male_rank, location, male_value, male_change])
        female_data.append([female_rank, location, female_value, female_change])

    male_data = sorted(male_data, key=lambda x: x[0])  # Sort by male rank
    female_data = sorted(female_data, key=lambda x: x[0])  # Sort by female rank

    table_data = []
    for male, female in zip(male_data, female_data):
        table_data.append(male + female)

    table_columns = ['RANK (MALE)', 'COUNTRY (MALE)', 'YEARS EXPECTED TO LIVE (MALE)', f'CHANGE IN 5 YEARS ({previous_year}-{selected_year}) (MALE)',
                     'RANK (FEMALE)', 'COUNTRY (FEMALE)', 'YEARS EXPECTED TO LIVE (FEMALE)', f'CHANGE IN 5 YEARS ({previous_year}-{selected_year}) (FEMALE)']

    # Convert to DataFrame
    table_df = pd.DataFrame(table_data, columns=table_columns)

    # Create Plotly table
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=table_columns,
                    fill_color=[dark_blue,dark_blue,dark_blue,dark_blue,'red','red','red','red'],
                    font=dict(color='white'),
                    align='left'),
        cells=dict(values=[table_df[col] for col in table_df.columns],
                   fill_color='white',
                   align='center'))
    ])

    # Add scroll to the table
    table_fig.update_layout(
        height=600,  # Adjust height as needed
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(table_fig)

if __name__ == "__main__":
    main()
