# imports
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# app instantiation
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# preparing dataframes
poverty = pd.read_csv('data/poverty.csv', low_memory=False)
series = pd.read_csv('data/PovStatsSeries.csv')
population = 'Population, total'
population_df = poverty[poverty[population].notna()]

#dataframe definitions for poverty gap graph
perc_pov_cols = poverty.filter(regex='Poverty gap').columns
perc_pov_df = poverty[poverty['is_country']].dropna(subset=perc_pov_cols)
perc_pov_years = sorted(set(perc_pov_df['year']))
cividis0 = px.colors.sequential.Cividis[0]

def make_empty_fig():
    fig = go.Figure()
    fig.layout.paper_bgcolor = '#ecf0f1'
    fig.layout.plot_bgcolor = '#ecf0f1'
    return fig

# app layout (list of HTML and interactive components)
app.layout = html.Div([
    dbc.Col([
        html.Br(),
        html.H2('POVERTY AND EQUITY DATABASE'),
        html.H3('THE WORLD BANK'),
    ], style={'textAlign': 'center'}),
    dbc.Tabs([
        dbc.Tab([
            dbc.Row([
                dbc.Col(lg=2),
                dbc.Col([
                    html.Br(),
                    dcc.Dropdown(id='indicator_dropdown',
                                 value='GINI index (World Bank estimate)',
                                 options=[{'label': indicator, 'value': indicator}
                                          for indicator in poverty.columns[3:54]]),
                    dcc.Graph(id='indicator_map_chart'),
                    html.Br(),
                    dcc.Markdown(id='indicator_map_details_md',
                                 style={'backgroundColor': '#ecf0f1'})
                ], lg=8),
                html.Br(),
            ]), # gini index map
        ], label='Index map'),
        dbc.Tab([
        html.Br(),
        dbc.Row([
            dbc.Col(lg=1),
            dbc.Col([
                dbc.Label('Select the year'),
                dcc.Slider(id='year_cluster_slider',
                           dots=True, min=1974, max=2018, step=1, included=False,
                           value=2018,
                           marks={year: str(year)
                                  for year in range(1974, 2019, 5)})
            ], lg=5),
            dbc.Col([
                dbc.Label('Select the number of clusters'),
                dcc.Slider(id='ncluster_cluster_slider',
                           dots=True, min=2, max=15, step=1, included=False,
                           value=4,
                           marks={n: str(n) for n in range(2, 16)}),
            ], lg=5),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(lg=1),
            dbc.Col([
                dbc.Label('Select indicators'),
                dcc.Dropdown(id='cluster_indicator_dropdown',
                             optionHeight=30,
                             multi=True,
                             value=['GINI index (World Bank estimate)'],
                             options=[{'label': indicator, 'value': indicator}
                                      for indicator in poverty.columns[3:54]]),
            ], lg=8),
            dbc.Col([
                html.Br(),
                dbc.Button("Submit", id='clustering_submit_button', size="me-1"),
                ], lg=2),
        ]),
        dbc.Row([
            dbc.Col(lg=1),
            dbc.Col([
                dcc.Loading([
                    dcc.Graph(id='clustered_map_graph'),
                    html.Br(),
                    dcc.Markdown(id='clustered_map_details',
                                 style={'backgroundColor': '#ecf0f1'})
                ]),
            ], lg=10),
        ]),
        html.Br(),


        ], label='Countries clustering'),
    ]),
    html.Br(),
    dbc.Col([
        html.Br(),
        html.H3('Population trend (1974-2018)'),
        ], style={'textAlign': 'center'}),
    dbc.Row([
        dbc.Col(lg=2),
        dbc.Col([
            dbc.Label('Countries'),
            dcc.Dropdown(id='population_country_dropdown',
                         multi=True,
                         value=['Ukraine'],
                         placeholder='Select one or more countries',
                         options=[{'label': country, 'value': country}
                                  for country in population_df['Country Name'].unique()]),
            html.Br(),
            dcc.Graph(id='population_country_barchart',
                      figure=make_empty_fig())
        ], lg=8),
    ]), # country population graph
    html.Br(),
    html.Br(),
    html.H3('Poverty Gap at $1.9, $3.2, and $5.5 (% of population)',
            style={'textAlign': 'center'}),
    html.Br(),
    dbc.Row([
        dbc.Col(lg=2),
        dbc.Col([
            dbc.Label('Select poverty level:'),
            dcc.Slider(id='perc_pov_indicator_slider',
                       min=0,
                       max=2,
                       step=1,
                       value=0,
                       # dots=True,
                       included=False,
                       marks={0: {'label': '$1.9', 'style': {'color': cividis0, 'fontWeight': 'bold', 'fontSize': 15}},
                              1: {'label': '$3.2', 'style': {'color': cividis0, 'fontWeight': 'bold', 'fontSize': 15}},
                              2: {'label': '$5.5',
                                  'style': {'color': cividis0, 'fontWeight': 'bold', 'fontSize': 15}}}),

        ], lg=2),
        dbc.Col([
            dbc.Label('Select year:'),
            dcc.Slider(id='perc_pov_year_slider',
                       min=perc_pov_years[0],
                       max=perc_pov_years[-1],
                       step=1,
                       included=False,
                       value=2018,
                       marks={year: {'label': str(year), 'style': {'color': cividis0, 'fontSize': 14}}
                              for year in perc_pov_years[::5]}),

        ], lg=5),
    ]), # poverty chart settings
    dbc.Row([
        dbc.Col(lg=1),
        dbc.Col([
            dcc.Graph(id='perc_pov_scatter_chart',
                      figure=make_empty_fig())
        ], lg=10)
    ]), # poverty scatter graph
    dbc.Tabs([
        dbc.Tab([
            html.Ul([
                html.Br(),
                html.Li('Number of Economies: 170'),
                html.Li('Temporal Coverage: 1974 - 2019'),
                html.Li('Update Frequency: Quarterly'),
                html.Li('Last Updated: March 19, 2020'),
                html.Li(['Source: ',
                    html.A('https://datacatalog.worldbank.org/dataset/poverty-and-equity-database',
                           href='https://datacatalog.worldbank.org/dataset/poverty-and-equity-database')
                    ]),
                ]),
        ], label='Key Facts'),
        dbc.Tab([
            html.Ul([
                html.Br(),
                html.Li('Book title: Interactive Dashboards and Data Apps with Plotly and Dash'),
                html.Li(['GitHub repo: ',
                    html.A('https://github.com/PacktPublishing/Interactive-Dashboards-and-Data-Apps-with-Plotly-and-Dash',
                           href='https://github.com/PacktPublishing/Interactive-Dashboards-and-Data-Apps-with-Plotly-and-Dash')
                    ]),
                ]),
        ], label='Project Info')
    ]), # key facts and project info

], style={'backgroundColor': '#ecf0f1'}) # country population barchart

@app.callback(Output('indicator_map_chart', 'figure'),
              Output('indicator_map_details_md', 'children'),
              Input('indicator_dropdown', 'value'))
def display_generic_map_chart(indicator):
    df = poverty[poverty['is_country']]
    fig = px.choropleth(df,
                        locations='Country Code',
                        color_continuous_scale='cividis',
                        color=indicator,
                        animation_frame='year',
                        title=indicator,
                        hover_name='Country Name',
                        height=700
                        )

    # remove the rectangular frame arount the map
    fig.layout.geo.showframe = False
    # show the country borders
    fig.layout.geo.showcountries = True
    # use a different projection of the earth
    fig.layout.geo.projection.type = 'natural earth'
    # limit the vertical and horizontal range of the chart to focus more on countries
    fig.layout.geo.lataxis.range = [-53, 76]
    fig.layout.geo.lonaxis.range = [-137, 168]
    # change the color of the land to white
    fig.layout.geo.landcolor = 'white'
    # change background color of the map and the "paper" background
    fig.layout.geo.bgcolor = '#ecf0f1'
    fig.layout.paper_bgcolor = '#ecf0f1'
    # set the color of the country borders
    fig.layout.geo.countrycolor = 'gray'
    fig.layout.geo.coastlinecolor = 'gray'
    # split title of the color bar
    fig.layout.coloraxis.colorbar.title = indicator.replace(' ', '<br>')

    # dataframe definitions for chart description
    series_df = series[series['Indicator Name'].eq(indicator)]

    if series_df.empty:
        markdown = 'No details available on this indicator'
    else:
        limitations = series_df['Limitations and exceptions'].fillna('N/A').str.replace('\n\n', ' ').values[0]

        markdown = f"""
        #### {series_df['Indicator Name'].values[0]}  

        {series_df['Long definition'].values[0]}  

        * **Unit of measure:** {series_df['Unit of measure'].fillna('count').values[0]}
        * **Periodicity:** {series_df['Periodicity'].fillna('N/A').values[0]}
        * **Source:** {series_df['Source'].values[0]}

        #### Limitations and exceptions:  

        {limitations}  

        """
    return fig, markdown

@app.callback(Output('clustered_map_graph', 'figure'),
              Output('clustered_map_details', 'children'),
              Input('clustering_submit_button', 'n_clicks'),
              State('year_cluster_slider', 'value'),
              State('ncluster_cluster_slider', 'value'),
              State('cluster_indicator_dropdown', 'value'))
def clustered_map(n_clicks, year, n_clusters, indicators):
    if not indicators:
        raise PreventUpdate
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters)
    df = poverty[poverty['is_country'] & poverty['year'].eq(year)][indicators + ['Country Name', 'year']]
    data = df[indicators]
    if df.isna().all().any():
        return px.scatter(title='No available data for the selected combination of year/indicators.')
    data_no_na = imp.fit_transform(data)
    scaled_data = scaler.fit_transform(data_no_na)
    kmeans.fit(scaled_data)

    fig = px.choropleth(df,
                        locations='Country Name',
                        locationmode='country names',
                        color=[str(x) for x in kmeans.labels_],
                        labels={'color': 'Cluster'},
                        hover_data=indicators,
                        height=650,
                        title=f'Country clusters - {year} <br>Number of clusters: {n_clusters}<br>Inertia: {kmeans.inertia_:,.2f}',
                        color_discrete_sequence=px.colors.qualitative.T10
                        )
    fig.add_annotation(x=0, y=-0.15,
                       xref='paper', yref='paper',
                       text=f'Selected indicators:<br>' + "<br>".join(indicators),
                       showarrow=False)
    fig.layout.geo.showframe = False
    fig.layout.geo.showcountries = True
    fig.layout.geo.projection.type = 'natural earth'
    fig.layout.geo.lataxis.range = [-53, 76]
    fig.layout.geo.lonaxis.range = [-137, 168]
    fig.layout.geo.landcolor = 'white'
    fig.layout.geo.bgcolor = '#ecf0f1'
    fig.layout.paper_bgcolor = '#ecf0f1'
    fig.layout.geo.countrycolor = 'gray'
    fig.layout.geo.coastlinecolor = 'gray'

    markdown = f"""
            #### Clustering countries by indicators 
            Clustering will create groups of countries with similar characteristics and will assign 'clusters' to them.   
            * Select **year**, **number of clusters**,  **indicators** and press **submit**
            * Increase **number of clusters** if **Inertia** attribute is not close to 0. 
            """

    return fig, markdown

@app.callback(Output('population_country_barchart', 'figure'),
              Input('population_country_dropdown', 'value'))
def plot_population_country_barchart(countries):
    if not countries:
        raise PreventUpdate
    df = population_df[population_df['Country Name'].isin(countries)].dropna(subset=[population])
    fig = px.bar(df,
                 x='year',
                 y=population,
                 height=100 + (250 * len(countries)),
                 facet_row='Country Name',
                 color='Country Name',
                 labels={population: 'Population, total'},
                 title=''.join([population, '<br><b>', ', '.join(countries), '</b>']))
    fig.layout.paper_bgcolor = '#ecf0f1'
    fig.layout.plot_bgcolor = '#ecf0f1'
    return fig

@app.callback(Output('perc_pov_scatter_chart', 'figure'),
              Input('perc_pov_year_slider', 'value'),
              Input('perc_pov_indicator_slider', 'value'))
def plot_poverty_gap_chart(year, indicator):
    indicator = perc_pov_cols[indicator]
    df = (perc_pov_df
          [perc_pov_df['year'].eq(year)]
          .dropna(subset=[indicator])
          .sort_values(indicator))
    if df.empty:
        raise PreventUpdate

    fig = px.scatter(df,
                     x=indicator,
                     y='Country Name',
                     color='Population, total',
                     size=[30]*len(df),
                     size_max=15,
                     hover_name='Country Name',
                     height=250 +(20*len(df)),
                     color_continuous_scale='cividis',
                     title=indicator + '<b>: ' + f'{year}' +'</b>')
    fig.layout.paper_bgcolor = '#ecf0f1'
    fig.layout.plot_bgcolor = '#ecf0f1'
    fig.layout.xaxis.ticksuffix = '%'
    return fig

# running the app
if __name__ == '__main__':
    app.run_server(debug=True)