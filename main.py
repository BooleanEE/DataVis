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
dark_red = 'rgb(155,0,0)'

# Arrays de países por região e sub-região
africa = {
    "All": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo", "Dem. Rep. of the Congo", "Côte d'Ivoire", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mayotte", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Réunion", "Rwanda", "Saint Helena", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Western Sahara", "Zambia", "Zimbabwe"],
    "Eastern Africa": ["Burundi", "Comoros", "Djibouti", "Eritrea", "Ethiopia", "Kenya", "Madagascar", "Malawi", "Mauritius", "Mayotte", "Mozambique", "Réunion", "Rwanda", "Seychelles", "Somalia", "South Sudan", "Uganda", "United Republic of Tanzania", "Zambia", "Zimbabwe"],
    "Middle Africa": ["Angola", "Cameroon", "Central African Republic", "Chad", "Congo", "Dem. Rep. of the Congo", "Equatorial Guinea", "Gabon", "Sao Tome and Principe"],
    "Northern Africa": ["Algeria", "Egypt", "Libya", "Morocco", "Sudan", "Tunisia", "Western Sahara"],
    "Southern Africa": ["Botswana", "Eswatini", "Lesotho", "Namibia", "South Africa"],
    "Western Africa": ["Benin", "Burkina Faso", "Cabo Verde", "Côte d'Ivoire", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania", "Niger", "Nigeria", "Saint Helena", "Senegal", "Sierra Leone", "Togo"]
}

asia = {
    "All ": ["Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", "Brunei Darussalam", "Cambodia", "China", "China, Hong Kong SAR", "China, Macao SAR", "China, Taiwan Province of China", "Cyprus", "Dem. People's Rep. of Korea", "Georgia", "India", "Indonesia", "Iran (Islamic Republic of)", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyzstan", "Lao People's Dem. Republic", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "Oman", "Pakistan", "Philippines", "Qatar", "Republic of Korea", "Saudi Arabia", "Singapore", "Sri Lanka", "State of Palestine", "Syrian Arab Republic", "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan", "Türkiye", "United Arab Emirates", "Uzbekistan", "Viet Nam", "Yemen"],
    "Eastern Asia": ["China", "China, Hong Kong SAR", "China, Macao SAR", "China, Taiwan Province of China", "Dem. People's Rep. of Korea", "Japan", "Mongolia", "Republic of Korea"],
    "Central Asia": ["Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan"],
    "Southern Asia": ["Afghanistan", "Bangladesh", "Bhutan", "India", "Iran (Islamic Republic of)", "Maldives", "Nepal", "Pakistan", "Sri Lanka"],
    "South-Eastern Asia": ["Brunei Darussalam", "Cambodia", "Indonesia", "Lao People's Dem. Republic", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Timor-Leste", "Viet Nam"],
    "Western Asia": ["Armenia", "Azerbaijan", "Bahrain", "Cyprus", "Georgia", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Qatar", "Saudi Arabia", "State of Palestine", "Syrian Arab Republic", "Türkiye", "United Arab Emirates", "Yemen"]
}

europe = {
    "All  ": ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Czechia", "Denmark", "Estonia", "Faroe Islands", "Finland", "France", "Germany", "Gibraltar", "Greece", "Guernsey", "Hungary", "Iceland", "Ireland", "Isle of Man", "Italy", "Jersey", "Kosovo (under UNSC res. 1244)", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Republic of Moldova", "Romania", "Russian Federation", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom"],
    "Eastern Europe": ["Belarus", "Bulgaria", "Czechia", "Hungary", "Poland", "Republic of Moldova", "Romania", "Russian Federation", "Slovakia", "Ukraine"],
    "Northern Europe": ["Denmark", "Estonia", "Faroe Islands", "Finland", "Guernsey", "Iceland", "Ireland", "Isle of Man", "Jersey", "Latvia", "Lithuania", "Norway", "Sweden", "United Kingdom"],
    "Southern Europe": ["Albania", "Andorra", "Bosnia and Herzegovina", "Croatia", "Gibraltar", "Greece", "Italy", "Kosovo (under UNSC res. 1244)", "Malta", "Montenegro", "North Macedonia", "Portugal", "San Marino", "Serbia", "Slovenia", "Spain"],
    "Western Europe": ["Austria", "Belgium", "France", "Germany", "Liechtenstein", "Luxembourg", "Monaco", "Netherlands", "Switzerland"]
}

america = {
    "All   ": ["Argentina", "Belize", "Bermuda", "Bolivia (Plurinational State of)", "Brazil", "Canada", "Chile", "Colombia", "Costa Rica", "Ecuador", "Falkland Islands (Malvinas)", "French Guiana", "Greenland", "Guatemala", "Guyana", "Honduras", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru", "Saint Pierre and Miquelon", "Suriname", "United States of America", "Uruguay", "Venezuela (Bolivarian Republic of)"],
    "Central America": ["Belize", "Costa Rica", "El Salvador", "Guatemala", "Honduras", "Mexico", "Nicaragua", "Panama"],
    "South America": ["Argentina", "Bolivia (Plurinational State of)", "Brazil", "Chile", "Colombia", "Ecuador", "Falkland Islands (Malvinas)", "French Guiana", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela (Bolivarian Republic of)"],
    "North America": ["Bermuda", "Canada", "Greenland", "Saint Pierre and Miquelon", "United States of America"]
}


oceania = {
    "All    ": ["American Samoa", "Australia", "Cook Islands", "Fiji", "French Polynesia", "Guam", "Kiribati", "Marshall Islands", "Micronesia (Fed. States of)", "Nauru", "New Caledonia", "New Zealand", "Niue", "Northern Mariana Islands", "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tokelau", "Tonga", "Tuvalu", "Vanuatu", "Wallis and Futuna Islands"],
    "Australia/New Zealand": ["Australia", "New Zealand"],
    "Melanesia": ["Fiji", "New Caledonia", "Papua New Guinea", "Solomon Islands", "Vanuatu"],
    "Micronesia": ["Guam", "Kiribati", "Marshall Islands", "Micronesia (Fed. States of)", "Nauru", "Northern Mariana Islands", "Palau"],
    "Polynesia": ["American Samoa", "Cook Islands", "French Polynesia", "Niue", "Samoa", "Tokelau", "Tonga", "Tuvalu", "Wallis and Futuna Islands"]
}

regions = {
    "Africa": africa,
    "Asia": asia,
    "Europe": europe,
    "America": america,
    "Oceania": oceania
}



# Função para a visualização do gráfico de séries temporais
def time_series_chart():
    st.title('Ranking Time Series')

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
            padding-left: 5rem;
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

    # Filter dataframe based on selections for time series
    filtered_df_ts = life_expectancy_at_0_45_60_all_df[
        (life_expectancy_at_0_45_60_all_df['Sex'] == selected_sex) &
        (life_expectancy_at_0_45_60_all_df['Age'] == selected_age_ts)
    ]

    # Plot time series chart
    fig = px.line(filtered_df_ts, x='Time', y='Value', color='Location',
                  title=f'Life expectancy at age {selected_age_ts} for {selected_sex.lower()}',
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

# Função para a visualização da tabela de classificação

def ranking_table():
    st.title('Ranking Table')

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
            padding-left: 5rem;
            padding-right: 1rem;
        }
        h2 {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar para a tabela de classificação
    st.sidebar.header('Ranking Table')
    selected_age_table = st.sidebar.selectbox('Select the specific age:', ages, key='age_table')
    selected_year = st.sidebar.selectbox('Select the specific year:', list(range(1955, 2024)), key='year')

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
        male_value = f"{male_current['Value'].values[0]:.2f}" if not male_current.empty else None
        male_rank_prev = male_current['Rank_prev'].values[0] if not male_current.empty else None
        male_change = get_change_symbol(male_rank_prev - male_rank) if male_rank and male_rank_prev else None

        female_rank = female_current['Rank'].values[0] if not female_current.empty else None
        female_value = f"{female_current['Value'].values[0]:.2f}" if not female_current.empty else None
        female_rank_prev = female_current['Rank_prev'].values[0] if not female_current.empty else None
        female_change = get_change_symbol(female_rank_prev - female_rank) if female_rank and female_rank_prev else None

        male_data.append([male_rank, location, male_value, male_change])
        female_data.append([female_rank, location, female_value, female_change])

    male_data = sorted(male_data, key=lambda x: x[0])  # Sort by male rank
    female_data = sorted(female_data, key=lambda x: x[0])  # Sort by female rank

    table_data = []
    for male, female in zip(male_data, female_data):
        table_data.append(male + female)

    table_columns = ['RANK (MALE)', 'COUNTRY', 'YEARS EXPECTED TO LIVE', f'CHANGE IN 5 YEARS ({previous_year}-{selected_year})',
                     'RANK (FEMALE)', 'COUNTRY ', 'YEARS EXPECTED TO LIVE ', f'CHANGE IN 5 YEARS ({previous_year}-{selected_year}) ']

    # Convert to DataFrame
    table_df = pd.DataFrame(table_data, columns=table_columns)

    # Create Plotly table
    table_fig = go.Figure(data=[go.Table(
        columnwidth=[15, 25, 12, 15, 15, 25, 12, 15],  # Adjust widths as needed
        header=dict(values=table_columns,
                    fill_color=[dark_blue, dark_blue, dark_blue, dark_blue, 'black', 'black', 'black', 'black'],
                    font=dict(color='white'),
                    align=['center', 'left', 'right', 'center', 'center', 'left', 'right', 'center']),
        cells=dict(values=[table_df[col] for col in table_df.columns],
                   fill_color='white',
                   align=['center', 'left', 'right', 'center', 'center', 'left', 'right', 'center'])
    )])

    # Add scroll to the table
    table_fig.update_layout(
        height=600,  # Adjust height as needed
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(table_fig)

def other_grafic():
    st.title('Dot Graphic')

    st.sidebar.header('Dot Graphic')

    # Cria listas para as regiões e subdivisões
    region_names = list(regions.keys())
    region_dict = {region: list(subregions.keys()) if isinstance(subregions, dict) else [region]
                   for region, subregions in regions.items()}
    subregion_dict = {subregion: countries for region, subregions in regions.items()
                      for subregion, countries in (subregions.items() if isinstance(subregions, dict) else [(region, subregions)])}

    # Adiciona seleção de região e subdivisão no sidebar
    selected_region = st.sidebar.selectbox('Select the region:', region_names)
    selected_subregion = st.sidebar.selectbox('Select the subregion:', region_dict[selected_region])

    # Obtém os países da subregião selecionada
    selected_countries = subregion_dict[selected_subregion]

    # Seleção de idade e ano
    selected_age = st.sidebar.selectbox('Select the specific age:', ages)
    selected_year = st.sidebar.selectbox('Select the year:', list(range(1950, 2024)))

    # Aplicando estilo ao sidebar
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
            padding-left: 5rem;
            padding-right: 1rem;
        }
        h2 {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # Filtra o dataframe com base nas seleções
    filtered_df = df[
        (df['Time'] == selected_year) &
        (df['Location'].isin(selected_countries)) &
        (df['Age'] == selected_age)
    ]

    # Criação do gráfico de dispersão
    scatter_fig = go.Figure()

    # Separação dos dados por sexo
    female_df = filtered_df[filtered_df['Sex'] == 'Female']
    male_df = filtered_df[filtered_df['Sex'] == 'Male']

    scatter_fig.add_trace(go.Scatter(
        x=female_df['Location'],
        y=female_df['Value'],
        mode='markers',
        name='Female',
        marker=dict(size=10, symbol='circle', color='lightblue')
    ))

    scatter_fig.add_trace(go.Scatter(
        x=male_df['Location'],
        y=male_df['Value'],
        mode='markers',
        name='Male',
        marker=dict(size=10, symbol='diamond', color=dark_blue)
    ))

    # Adiciona linhas conectando pontos para cada país
    for country in selected_countries:
        female_country_df = female_df[female_df['Location'] == country]
        male_country_df = male_df[male_df['Location'] == country]

        if not female_country_df.empty and not male_country_df.empty:
            female_value = female_country_df['Value'].iloc[0]
            male_value = male_country_df['Value'].iloc[0]

            scatter_fig.add_trace(go.Scatter(
                x=[country, country],
                y=[female_value, male_value],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))
            scatter_fig.add_trace(go.Scatter(
                x=[country, country],
                y=[0, female_value],
                mode='lines',
                line=dict(color='black', width=1, dash='dot'),
                showlegend=False
            ))
            scatter_fig.add_trace(go.Scatter(
                x=[country, country],
                y=[0, male_value],
                mode='lines',
                line=dict(color='black', width=1, dash='dot'),
                showlegend=False
            ))

    scatter_fig.update_layout(
        title=f'Life expectancy comparison at the specific age {selected_age} in {selected_year}',
        xaxis_title='Country',
        yaxis_title='Years expected to live',
        height=800,
        width=1100
    )

    st.plotly_chart(scatter_fig)

# Função de visualização 'Time Series' atualizada
def time_series():
    st.title('Time Series')

     # Aplicando estilo ao sidebar
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
            padding-left: 3rem;
            padding-right: 1rem;
        }
        .st-emotion-cache-j6qv4b {
              color: white;
        }
        h2 {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar para seleção de sexo e país
    st.sidebar.header('Time Series Filters')
    selected_sex_ts = st.sidebar.radio('Select the sex:', ['Female', 'Male'])
    selected_country_ts = st.sidebar.selectbox('Select a country:', locations)

    # Filtrar dados para o país e sexo selecionados
    country_data = df[
        (df['Location'] == selected_country_ts) &
        (df['Sex'] == selected_sex_ts)
    ]

    # Preparar dados para o gráfico
    ages_to_plot = [0, 45, 65]
    fig = go.Figure()

    for age in ages_to_plot:
        age_data = country_data[country_data['Age'] == age].sort_values('Time')
        # Calcular a mudança percentual do ano anterior
        age_data['Percent Change'] = age_data['Value'].pct_change() * 100

        fig.add_trace(go.Scatter(
            x=age_data['Time'],
            y=age_data['Value'],
            mode='lines+markers',
            name=f'Age {age}',
            hovertemplate='<b>Year:</b> %{x}<br>' +
                          '<b>Years expected to live:</b> %{y}<br>' +
                          '<b>Percent change from last year:</b> %{text}',
            text=age_data['Percent Change'].apply(lambda x: f'Increased {x:.2f}%' if x > 0 else f'Decreased {-x:.2f}%').fillna('No data')
        ))

    # Configurações do layout do gráfico
    fig.update_layout(
        title=f'Life expectancy over time for {selected_sex_ts.lower()} in {selected_country_ts}',
        xaxis_title='Year',
        yaxis_title='Years expected to live',
        legend_title='Age'
    )

    st.plotly_chart(fig)

# Função principal para controlar a navegação entre as páginas
def main():
    st.title('Life Expectancy Visualization')

    # Adicionando opções para navegação entre as páginas
    st.sidebar.header('Visualization')
    page = st.sidebar.selectbox("Choose a visualization", ["Ranking Time Series", "Time Series", "Ranking Table", "Dot Graphic"])

    if page == "Ranking Time Series":
        time_series_chart()
    elif page == "Ranking Table":
        ranking_table()
    elif page == "Dot Graphic":
        other_grafic()
    elif page == "Time Series":
        time_series()

if __name__ == "__main__":
    main()

