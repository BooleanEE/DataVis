import numpy as np
import pandas as pd
import plotly.express as px
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
        
    st.title('Life Expectancy')

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
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with multi-select dropdown menu
    selected_age = st.sidebar.selectbox('Select the specific age:', ages)
    selected_countries = st.sidebar.multiselect('Select a country or countries:', locations, default=None)
    selected_sex = st.sidebar.selectbox('Select the sex:', sexes)

    # Filter dataframe based on selections
    filtered_df = life_expectancy_at_0_45_60_all_df[
        (life_expectancy_at_0_45_60_all_df['Sex'] == selected_sex) &
        (life_expectancy_at_0_45_60_all_df['Age'] == selected_age)
    ]

    # Plot time series chart
    fig = px.line(filtered_df, x='Time', y='Value', color='Location',
                  title=f'Life Expectancy at age {selected_age} for {selected_sex}',
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
    fig.update_layout(xaxis=dict(range=[filtered_df['Time'].min(), filtered_df['Time'].max()]))

    # Change margin
    fig.update_layout(margin=dict(r=225))
    
    # Show the selected country name at the end of the time series
    if selected_countries:
            for country in selected_countries:
                country_df = filtered_df[filtered_df['Location'] == country]
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
                                    font=dict(family="Courier New, monospace", size=10))
                                            

    #fig.update_annotations(align="left")

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
