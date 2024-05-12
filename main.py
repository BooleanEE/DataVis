#import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

#from dash import dcc, html
#from dash.dependencies import Input, Output

# Data pre-processing

life_expectancy_at_birth_both_sexes_df = pd.read_csv("./data/life_expectancy_at_birth_both_sexes.csv")

life_expectancy_at_birth_both_sexes_df["Value"] = life_expectancy_at_birth_both_sexes_df["Value"].str.replace(',', '.')

df = life_expectancy_at_birth_both_sexes_df

# Get unique location names
locations = df['Location'].unique()

# Custom shade of black (light black)
custom_black = 'rgba(0, 0, 0, 0.075)'  # Adjust the alpha value for lightness

# Streamlit app
def main():
    st.title('Life Expectancy at Birth')
    
    # Sidebar with multi-select dropdown menu
    selected_countries = st.sidebar.multiselect('Select a country or countries:', locations)
    
    # Plot time series chart
    fig = px.line(df, x='Time', y='Value', color='Location',
                  title='Life Expectancy at Birth',
                  labels={'Time': 'Year', 'Value': 'Age', 'Location': 'Country'})
    
    # Set line color to custom shade of black for all lines
    fig.update_traces(line=dict(color=custom_black))
    
    # Change color to normal black for selected countries
    for country in selected_countries:
        fig.for_each_trace(lambda trace: trace.update(line=dict(color='black')) if trace.name == country else ())
    
    # Increase the size of the chart
    fig.update_layout(height=800, width=1100)
    
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()

