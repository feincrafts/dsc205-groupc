import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium as fm
from streamlit_folium import st_folium
import plotly.graph_objects as go
import seaborn as sns
import time
import geopandas as gpd
import os
from shapely.geometry import Polygon, MultiPolygon
import branca
from sklearn.preprocessing import StandardScaler

#Helper function to remove glitchy parts of country shape files
def keep_large_parts(gdf, area_km2=100):
    """
    For every (multi)polygon in *gdf* return only the sub‑polygons whose
    area is larger than *area_km2* (square kilometres).
    """
    if gdf.empty:
        return gdf

    # 1) explode multipolygons → one row per simple polygon
    exploded = gdf.explode(index_parts=False)

    # 2) project to an equal‑area CRS so 'geometry.area' is meaningful
    #    (EPSG:6933 = World Cylindrical Equal Area; works globally)
    ea = exploded.to_crs(epsg=6933)

    # 3) compute area in km²
    ea["area_km2"] = ea["geometry"].area / 1e6

    # 4) filter
    ea = ea[ea["area_km2"] >= area_km2]

    # 5) dissolve back to one multi‑polygon per original country
    cleaned = (
        ea.drop(columns="area_km2")
          .to_crs(gdf.crs)          # convert back to original CRS
          .dissolve(by=gdf.index.name or "country", as_index=False)
    )
    return cleaned
#Helper function to parse 'k'/'M'/'B' suffixes in the data
def parse_suffix(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(',', '').strip()
    factor = 1
    if s.endswith(('k', 'M', 'B')):
        factor = {'k': 1e3, 'M': 1e6, 'B': 1e9}[s[-1]]
        s = s[:-1] #Drop the suffix
    try:
        return float(s) * factor
    except ValueError:
        return np.nan  
world = gpd.read_file('./world-administrative-boundaries.geojson')
world = world[['name', 'geometry']]
world = world.rename(columns={'name': 'country'})
#Rename countries so they don't get dropped in the merge
world["country"] = world["country"].replace(
    {"United States of America": "USA",
     'Iran (Islamic Republic of)': 'Iran',
     'Russian Federation': 'Russia',
     'Libyan Arab Jamahiriya': 'Libya',
     'United Republic of Tanzania': 'Tanzania',
     'Congo': 'Congo, Rep.',
     'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
     'U.K. of Great Britain and Northern Ireland': 'UK',
     "Lao People's Democratic Republic": 'Lao',
     'Moldova, Republic of': 'Moldova',
     'Slovakia': 'Slovak Republic',
     'Syrian Arab Republic': 'Syria',
     'United Arab Emirates': 'UAE',
     'Kyrgyzstan': 'Kyrgyz Republic'}
)
#Add in North and South Korea
NK = gpd.read_file('./gadm41_PRK.gpkg')
NK = NK[['COUNTRY', 'geometry']]
NK.rename(columns={'COUNTRY': 'country'}, inplace=True)
SK = gpd.read_file('./gadm41_KOR.gpkg')
SK = SK[['COUNTRY', 'geometry']]
SK.rename(columns={'COUNTRY': 'country'}, inplace=True)
NK_clean = keep_large_parts(NK, area_km2=100)
SK_clean = keep_large_parts(SK, area_km2=100)
world = gpd.GeoDataFrame(
    pd.concat([world, NK_clean, SK_clean], ignore_index=True),
    crs=world.crs
)
internet_usage_pct = pd.read_csv("./datasets/internet_users.csv") #percent of population
gdp = pd.read_csv("./datasets/gdp_pcap.csv")
pop = pd.read_csv("./datasets/pop.csv")

def convert_count(value):
  if isinstance(value, float):
    pass
  elif value[-1] == 'µ':
    return float(value[:-1]) * .000001
  elif value[-1].upper() == 'B':
    return float(value[:-1]) * 1000000000
  elif value[-1].upper() == 'M':
    return float(value[:-1]) * 1000000
  elif value[-1].upper() == 'K':
    return float(value[:-1]) * 1000
  else:
    return float(value)
  
pop.update(pop.drop(columns='country').map(convert_count))

cellphones = pd.read_csv('./datasets/cell_phones_per_100_people.csv')
cellphones = cellphones.melt(id_vars=['country'],
                             var_name='year',
                             value_name='cellphones_per_100k')
cellphones['year'] = cellphones['year'].astype(int)
mask = (cellphones["country"] == "China") & (cellphones["year"] == 1987)
cellphones = cellphones[~mask]
cellphones['cellphones_per_100k'] = cellphones['cellphones_per_100k'].astype(float)

education = pd.read_csv("./datasets/mean_years_in_school_men_25_years_and_older.csv")
education = education.melt(id_vars=['country'],
                             var_name='year',
                             value_name='years_of_schooling')
education['year'] = education['year'].astype(int)
education['years_of_schooling'] = education['years_of_schooling'].astype(float)

idx = internet_usage_pct.columns.get_loc('1990')
internet_pct_lists = [internet_usage_pct.columns[0]] + list(internet_usage_pct.columns[idx:])
internet_usage_pct = internet_usage_pct[internet_pct_lists]

#using gdp as an example
df_country = gdp['country'].unique()

co2 = pd.read_csv('./datasets/co2_countries.csv')
co2 = co2.melt(id_vars=['country'], var_name='year', value_name='co2_emission_per_cap')
co2['year'] = co2['year'].astype(int)
co2['co2_emission_per_cap'] = co2['co2_emission_per_cap'].replace('−', '-', regex=True).astype(float)

#importing energy dataset
energy = pd.read_csv("./datasets/eg_use_elec_kh_pc.csv")
energy = energy.melt(id_vars=['country'], var_name="year", value_name="energy_use")
energy['year'] = energy['year'].astype(int)



#defaults 
min_year = 1990
max_year = 2020
if 'timeslider' not in st.session_state: #avoids timeslider not initialized error
    st.session_state.timeslider = min_year
if 'animToggle' not in st.session_state: #avoids any animation related errors
    st.session_state.animToggle = True 
#note: I don't know why the timeslider doesn't show up for the animation, maybe because too fast to render? - Kaye

animToggle = st.toggle("Toggle Animation?", value=False)

st.sidebar.title('Navigation')
page_select = st.sidebar.selectbox('Select a page:', ['Home', 'CO2 Emissions', 'GDP and Tech', 'Cellphone Usage and Education', 'Correlation', 'Energy Consumption'])

if page_select != 'Home':
  options = ['All Countries'] + list(df_country)
  country_select = st.sidebar.multiselect(
    'Filter Country:',
    options=options,
    default=['All Countries']
  )
  
if page_select == 'Home':
    st.title('Investigating Intersections of Technology and Industrialization')
    st.write("In the last two centuries, many of the world's nations have been transformed by the industrial revolution and the invention of digital technologies. This dashboard allows one to explore the development of countries individually and track these changes over time.")
elif page_select == 'CO2 Emissions': 
    st.title("CO2 Emissions (tonnes per Capita)")
    year_placeholder = st.empty()
    map_placeholder = st.empty()
    year_select = st.slider(
    'Select a year',
    min_value=1800,
    max_value=2022,
    value = 1800,
    step=1
    )
    co2_year = co2[co2['year']==year_select]
    co2_year = co2_year.merge(world, on='country', how='inner')
    co2_year = gpd.GeoDataFrame(co2_year, geometry='geometry')
    co2_year['geometry'] = co2_year['geometry'].simplify(tolerance=0.001, preserve_topology=True)
    cmin, cmax = co2_year['co2_emission_per_cap'].min(), co2_year['co2_emission_per_cap'].max()
    cmap = branca.colormap.linear.YlOrRd_09.scale(cmin, cmax)
    year_placeholder.subheader(f'Year: {year_select}')
    choromap = fm.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron")
    def style_function(feature):
        value = feature["properties"]["co2_emission_per_cap"]
        if value is None or np.isnan(value):
            return {"fillOpacity": 0} #no data → transparent
        return {
            "fillColor": cmap(value),
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.7,
        }
    tooltip = fm.GeoJsonTooltip(
        fields=["country", "co2_emission_per_cap"],
        aliases=["Country", "CO₂ / cap"],
        localize=True,
        sticky=False
    )
    fm.GeoJson(
        co2_year,
        style_function=style_function,
        tooltip=tooltip,
    ).add_to(choromap)
    cmap.add_to(choromap)
    with map_placeholder: 
        _ = st_folium(choromap, width=700, height=500, returned_objects=[])
        
elif page_select == 'GDP and Tech': 
  st.title("RQ2: Does a country’s GDP relate to the country’s usage of technology?")
  usage = internet_usage_pct.copy()
  currGDP = gdp.copy()

  minidx = gdp.columns.get_loc(usage.columns[1])
  min_year = int(usage.columns[1])
  maxidx = gdp.columns.get_loc(usage.columns[-1])
  max_year = int(usage.columns[-1])

  #filters the gdp data to match the cellphone data in terms of years
  gdplist = [gdp.columns[0]] + list(gdp.columns[minidx:maxidx])
  currGDP = gdp[gdplist]

  currYear = st.session_state.timeslider

  tab0, tab1, tab2 = st.tabs(["Selected Scatterplot", "General Scatterplot", "Population Scatterplot"])
  with tab0:
   
    currYear = str(2017)

    usageYear = usage[['country', currYear]].rename(columns={currYear: 'usage'})
    gdpYear = gdp[['country', currYear]].rename(columns={currYear: 'gdp'})
    mergedDF = pd.merge(usageYear, gdpYear, on='country')
    mergedDF = mergedDF.dropna()

    min_year = int(usage.columns[1])
    max_year = int(usage.columns[-1])

    countrylist = country_select.copy()
    if len(countrylist) == 1 and ('All Countries' in countrylist):
      fig = px.scatter(mergedDF, x='usage', y='gdp', hover_name='country', title = "Internet Usage by GDP per country in Year " + str(currYear))
    else:
      allCountry = False
      if ('All Countries' in countrylist and len(countrylist) > 1):
        countrylist.remove('All Countries')
        allCountry = True

      highlight = mergedDF[mergedDF['country'].isin(country_select)]
      others = mergedDF[~mergedDF['country'].isin(country_select)]

      trace1 = go.Scatter(
          x=highlight['usage'],
          y=highlight['gdp'],
          mode='markers',
          marker=dict(color='orange'),
          name='Highlighted',
          text=highlight['country'],
          hoverinfo='text+x+y'
      )

      if allCountry:
        trace2 = go.Scatter(
            x=others['usage'],
            y=others['gdp'],
            mode='markers',
            name='Others',
            text=others['country'],
            hoverinfo='text+x+y',
            opacity = 0.4
        )

        fig = go.Figure(data=[trace2, trace1])
      else:
        fig = go.Figure(data=[trace1])

      fig.update_layout(
          title="Internet Usage by GDP per country in Year " + str(currYear),
          xaxis_title='Internet Usage (by percent)',
          yaxis_title='GDP'
      )
    key = time.time() #unique key
    st.plotly_chart(fig, y='gdp', key=key)
    st.write("This graph seeks to illustrate the start of the shift from a logarithmic to a linear relationship of the variables over time, perhaps as the digital divide narrows.")

  with tab1:
    selected_tab = "General Scatterplot"
    if animToggle and selected_tab == "General Scatterplot":
      fig_goes_here = st.empty()
      while animToggle:
        if int(currYear) < int(max_year):
          currYear = str(int(currYear) + 1)
        elif int(currYear) == int(max_year):
          currYear = str(min_year)

        # don't worry about the redundancy
        usageYear = usage[['country', currYear]].rename(columns={currYear: 'usage'})
        gdpYear = gdp[['country', currYear]].rename(columns={currYear: 'gdp'})
        mergedDF = pd.merge(usageYear, gdpYear, on='country')
        mergedDF = mergedDF.dropna()
        countrylist = country_select.copy()
        if len(countrylist) == 1 and ('All Countries' in countrylist):
          fig = px.scatter(mergedDF, x='usage', y='gdp', hover_name='country', title = "Internet Usage by GDP per country in Year " + str(currYear))
        else:
          allCountry = False
          if ('All Countries' in countrylist and len(countrylist) > 1):
            countrylist.remove('All Countries')
            allCountry = True

          highlight = mergedDF[mergedDF['country'].isin(country_select)]
          others = mergedDF[~mergedDF['country'].isin(country_select)]

          trace1 = go.Scatter(
              x=highlight['usage'],
              y=highlight['gdp'],
              mode='markers',
              marker=dict(color='orange'),
              name='Highlighted',
              text=highlight['country'],
              hoverinfo='text+x+y'
          )
          if allCountry:
            trace2 = go.Scatter(
                x=others['usage'],
                y=others['gdp'],
                mode='markers',
                name='Others',
                text=others['country'],
                hoverinfo='text+x+y',
                opacity = 0.4
            )

            fig = go.Figure(data=[trace2, trace1])
          else:
            fig = go.Figure(data=[trace1])

          fig.update_layout(
              title="Internet Usage by GDP per country in Year " + str(currYear),
              xaxis_title='Internet Usage',
              yaxis_title='GDP'
          )
        key = time.time() #unique key
        fig_goes_here.plotly_chart(fig, key=key) # hope this works
        time.sleep(0.45) 

    else:
      currYear = str(st.session_state.timeslider)

      usageYear = usage[['country', currYear]].rename(columns={currYear: 'usage'})
      gdpYear = gdp[['country', currYear]].rename(columns={currYear: 'gdp'})
      mergedDF = pd.merge(usageYear, gdpYear, on='country')
      mergedDF = mergedDF.dropna()

      min_year = int(usage.columns[1])
      max_year = int(usage.columns[-1])

      countrylist = country_select.copy()
      if len(countrylist) == 1 and ('All Countries' in countrylist):
        fig = px.scatter(mergedDF, x='usage', y='gdp', hover_name='country', title = "Internet Usage by GDP per country in Year " + str(currYear))
      else:
        allCountry = False
        if ('All Countries' in countrylist and len(countrylist) > 1):
          countrylist.remove('All Countries')
          allCountry = True

        highlight = mergedDF[mergedDF['country'].isin(country_select)]
        others = mergedDF[~mergedDF['country'].isin(country_select)]

        trace1 = go.Scatter(
            x=highlight['usage'],
            y=highlight['gdp'],
            mode='markers',
            marker=dict(color='orange'),
            name='Highlighted',
            text=highlight['country'],
            hoverinfo='text+x+y'
        )

        if allCountry:
          trace2 = go.Scatter(
              x=others['usage'],
              y=others['gdp'],
              mode='markers',
              name='Others',
              text=others['country'],
              hoverinfo='text+x+y',
              opacity = 0.4
          )

          fig = go.Figure(data=[trace2, trace1])
        else:
          fig = go.Figure(data=[trace1])

        fig.update_layout(
            title="Internet Usage by GDP per country in Year " + str(currYear),
            xaxis_title='Internet Usage',
            yaxis_title='GDP'
        )
    key = time.time() #unique key
    st.plotly_chart(fig, y='gdp', key=key)
    st.write("This graph seeks to illustrate the development of a relationship between the variables over time. This animation shows that the relationship of the percentage of internet usage compared to gdp appears to develop from almost logarithmic to linear (at around year 2010).")

  with tab2:
    #added selected tab for performance reasons but not sure if it works
    selected_tab = "Population Scatterplot"
    st.write("Due to Streamlit limitations, animation does not work on this page ")
    # I don't know why, but animToggle does not work with this plot, even if it's on tab1
    # and sometimes even with no plots at all
    # baffled and confused
    if animToggle and selected_tab == "Population Scatterplot":
      pass

    else:
      currYear = str(st.session_state.timeslider)

      #technically added it but it doesn't help much
      #min_year = int(usage.columns[1])
      #max_year = int(usage.columns[-1])

      #dataset initialization
      usageYear = usage[['country', currYear]].rename(columns={currYear: 'usage'})
      gdpYear = gdp[['country', currYear]].rename(columns={currYear: 'gdp'})
      mergedDF = pd.merge(usageYear, gdpYear, on='country')
      popYear = pop[['country', currYear]].rename(columns={currYear: 'pop'})
      merged_currYear = pd.merge(mergedDF, popYear, on=['country'], how='inner')
      merged_currYear["pop"] = pd.to_numeric(merged_currYear["pop"]) #error handling
      merged_currYear.dropna()

      countrylist = country_select.copy()
      if len(countrylist) == 1 and ('All Countries' in countrylist):
        fig = px.scatter(merged_currYear, x="gdp", y="usage", size="pop",
          hover_name="country", log_x = True, title = "GDP vs Internet Usage for Year " + str(currYear), size_max=60)
      else:
        allCountry = False
        if ('All Countries' in countrylist and len(countrylist) > 1):
          countrylist.remove('All Countries')
          allCountry = True

        highlight = merged_currYear[merged_currYear['country'].isin(country_select)]
        others = merged_currYear[~merged_currYear['country'].isin(country_select)]
        sizeref = 2. * max(merged_currYear["pop"]) / (60. ** 2)

        trace1 = go.Scatter(
            x=highlight['gdp'],
            y=highlight['usage'],
            mode='markers',
            marker=dict(
                size=highlight['pop'],
                sizemode='area',  
                sizeref=sizeref,
                sizemin=4,
                color = 'orange'
            ),
            name='Highlighted',
            text=highlight['country'],
            hoverinfo='text+x+y'
        )

        if allCountry:
          trace2 = go.Scatter(
              x=others['gdp'],
              y=others['usage'],
              mode='markers',
              marker=dict(
                  size=others['pop'], 
                  sizemode='area',   
                  sizeref=sizeref,
                  sizemin=4      
              ),
              name='Others',
              text=others['country'],
              hoverinfo='text+x+y',
              opacity = 0.4
          )

          fig = go.Figure(data=[trace2, trace1])
        else:
          fig = go.Figure(data=[trace1])

        fig.update_layout(
            title="Internet Usage by GDP per country in Year " + str(currYear),
            xaxis_title='GDP',
            yaxis_title='Internet Usage'
        )

    key = time.time() #unique key
    st.plotly_chart(fig, key=key)
    st.write("This graph seeks to illustrate how GDP and Internet Usage (by percentage) relate over time by reversing the axis and with respect to population size in order to investigate any potential relationships.")

elif page_select == 'Cellphone Usage and Education':
  st.title("RQ3: Does a country's cellphone usage relate to its education outcomes?")
  currYear = int(st.session_state.timeslider)
  tab0, tab1 = st.tabs(["Selected Scatterplot", "General Scatterplot"])

  with tab0:
    currYear = 2006
    merged_df = pd.merge(cellphones, education, on=['country', 'year'], how='inner')
    merged_df = merged_df.dropna()
    #recoded a bit because it wasn't showing up in the animation
    current_year_df = merged_df[merged_df['year'] == int(currYear)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Years of Schooling vs. Cellphones per 100k for year ' + str(currYear))
    ax.set_xlabel('Years of Schooling')
    ax.set_ylabel('Cellphones per 100k')

    #country_select
    countrylist = country_select.copy()
    if len(countrylist) == 1 and ('All Countries' in countrylist):
      ax.scatter(x=current_year_df['years_of_schooling'], y=current_year_df['cellphones_per_100k'])
    else:

      if ('All Countries' in countrylist and len(countrylist) > 1):
        countrylist.remove('All Countries')

      highlight = current_year_df[current_year_df['country'].isin(country_select)]
      others = current_year_df[~current_year_df['country'].isin(country_select)]

      ax.scatter(x=others['years_of_schooling'], y=others['cellphones_per_100k'], alpha=0.4, label='Other country/countries')
      ax.scatter(x=highlight['years_of_schooling'], y=highlight['cellphones_per_100k'], color='orange', s=75, label='Selected country/countries')
      ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    st.pyplot(fig)
    st.write("This graph illustrates the start of the shift from an exponential to linear relationship when comparing cellphone usage to mean years of schooling for men ages 25 or older across the globe.")

  with tab1:
    if animToggle:
      fig_goes_here = st.empty()
      while animToggle:
        if int(currYear) < int(max_year):
          currYear = str(int(currYear) + 1)
        elif int(currYear) == int(max_year):
          currYear = int(min_year)

        merged_df = pd.merge(cellphones, education, on=['country', 'year'], how='inner')
        merged_df = merged_df.dropna()
        #recoded a bit because it wasn't showing up in the animation
        current_year_df = merged_df[merged_df['year'] == int(currYear)]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Years of Schooling vs. Cellphones per 100k for year ' + str(currYear))
        ax.set_xlabel('Years of Schooling')
        ax.set_ylabel('Cellphones per 100k')

        #country_select
        countrylist = country_select.copy()
        if len(countrylist) == 1 and ('All Countries' in countrylist):
          ax.scatter(x=current_year_df['years_of_schooling'], y=current_year_df['cellphones_per_100k'])
        else:

          if ('All Countries' in countrylist and len(countrylist) > 1):
            countrylist.remove('All Countries')

          highlight = current_year_df[current_year_df['country'].isin(country_select)]
          others = current_year_df[~current_year_df['country'].isin(country_select)]

          ax.scatter(x=others['years_of_schooling'], y=others['cellphones_per_100k'], label='Other country/countries')
          ax.scatter(x=highlight['years_of_schooling'], y=highlight['cellphones_per_100k'], color='orange', s=75, label='Selected country/countries')
          ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig_goes_here.pyplot(fig) # hope this works
        plt.close(fig)
        time.sleep(0.45) 

    else:
      currYear = int(st.session_state.timeslider)
      max_year = 2009
      merged_df = pd.merge(cellphones, education, on=['country', 'year'], how='inner')
      merged_df = merged_df.dropna()
      #recoded a bit because it wasn't showing up in the animation
      current_year_df = merged_df[merged_df['year'] == int(currYear)]

      fig, ax = plt.subplots(figsize=(10, 6))
      ax.set_title('Years of Schooling vs. Cellphones per 100k for year ' + str(currYear))
      ax.set_xlabel('Years of Schooling')
      ax.set_ylabel('Cellphones per 100k')

      #country_select
      countrylist = country_select.copy()
      if len(countrylist) == 1 and ('All Countries' in countrylist):
        ax.scatter(x=current_year_df['years_of_schooling'], y=current_year_df['cellphones_per_100k'])
      else:

        if ('All Countries' in countrylist and len(countrylist) > 1):
          countrylist.remove('All Countries')

        highlight = current_year_df[current_year_df['country'].isin(country_select)]
        others = current_year_df[~current_year_df['country'].isin(country_select)]

        ax.scatter(x=others['years_of_schooling'], y=others['cellphones_per_100k'], alpha=0.4, label='Other country/countries')
        ax.scatter(x=highlight['years_of_schooling'], y=highlight['cellphones_per_100k'], color='orange', s=75, label='Selected country/countries')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
      st.pyplot(fig)
    st.write("This graph illustrates the development of the relationship between mean years of schooling for men 25 or older across the globe and cellphone usage, over time.")

elif page_select == 'Correlation':
  st.title("RQ5: Correlation Matrix")
  st.image("./CorrMatrix.png")
  st.write("This image illustrates the potential correlation between various factors on each other. CO2 emissions seems to correlate most strongly with GDP, life expectancy, household income, and internet usage. Internet usage also correlated with GDP, life expectancy, and years of education.")
elif page_select == 'Energy Consumption':

  country_select = False
  #Line Plot of Energy Consumption
  st.title("RQ6: Does the percentage of Internet users affect the average power consumption?")
  #Merging data
  internet_usage_pct = internet_usage_pct.melt(id_vars=['country'], var_name='year', value_name="percentage")
  internet_usage_pct['year']= internet_usage_pct['year'].astype(int)
  merged = energy.merge(internet_usage_pct, on=['country', 'year'])

  #converting values
  merged['energy_use'] = merged['energy_use'].apply(convert_count)

  #Setting the year cutoff
  merged = merged[merged['year'] >=1990]

  #Scaling data
  scaler = StandardScaler()
  merged['energy_use'] = scaler.fit_transform(merged[['energy_use']])
  merged['percentage'] = scaler.fit_transform(merged[['percentage']])
  x = merged['year']
  fig, ax = plt.subplots(figsize= (10,6))
  sns.lineplot(data=merged,y="percentage", x="year", color='blue', ax=ax, label="Internet Users")
  sns.lineplot(data=merged,y="energy_use", x="year", color='orange', ax= ax, label="Energy Consumption")
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  ax.set_title('Percentage of Internet Users and Energy Consumption vs Time')
  ax.set_xlabel('Year (1990-2023)')
  ax.set_ylabel('Percentage')
  st.write("This Lineplot shows the increase in both energy usage and the percentage of Internet users over time." \
  " A key takeaway is that shortly after 2020, there is a spike in both energy usage and internet usage. This can most likely be attributed to the pandemic and the introduction of AI.")
  st.pyplot(fig)



if page_select in ['CO2 Emissions', 'GDP and Tech', 'Cellphone Usage and Education']:
  st.slider(
        'Select a year',
        min_value=min_year,
        max_value=max_year,
        value = st.session_state.timeslider,
        key="timeslider",
        step=1 
  )
