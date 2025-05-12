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
try: 
  import geopandas as gpd
except ImportError:
  import os
  os.system('pip install geopandas')
  import geopandas as gpd
try: 
  from shapely.geometry import Polygon, MultiPolygon
except ImportError:
  import os
  os.system('pip install shapely')
  from shapely.geometry import Polygon, MultiPolygon          

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
world = gpd.read_file('world-adminstrative-boundaries.geojson')
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
NK = gpd.read_file('gadm41_PRK.gpkg')
NK = NK[['COUNTRY', 'geometry']]
NK.rename(columns={'COUNTRY': 'country'}, inplace=True)
SK = gpd.read_file('gadm41_KOR.gpkg')
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

co2 = pd.read_csv('co2_countries.csv')
co2 = co2.melt(id_vars=['country'], var_name='year', value_name='co2_emission_per_cap')
co2['year'] = co2['year'].astype(int)
co2['co2_emission_per_cap'] = co2['co2_emission_per_cap'].replace('−', '-', regex=True).astype(float)

st.sidebar.title('Navigation')
page_select = st.sidebar.selectbox('Select a page:', ['Home', 'RQ1', 'RQ2', 'RQ3', 'RQ4', 'RQ5'])
if page_select == 'Home': st.title('Investigating Intersections of Technology and Industrialization')
elif page_select == 'RQ1': 
    st.title("RQ1")
    year_placeholder = st.empty()
    map_placeholder = st.empty()
    year_select = st.slider(
    'Select a year',
    min_value=1950,
    max_value=2025,
    value = 2000,
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
elif page_select == 'RQ2': 
    st.title("RQ2: Insert Plotly Express Graph")
elif page_select == 'RQ3': st.title("RQ3: Does a country’s GDP relate to the country’s Internet Usage")
elif page_select == 'RQ4': st.title("RQ4: Does a country's cellphone usage relate to its education outcomes?")
elif page_select == 'RQ5': st.title("RQ5: Correlation Matrix")
elif page_select == 'RQ6': st.title("RQ6: Erik's graph(s)")



st.title('Investigating Intersections of Technology and Industrialization')
st.session_state.selectedQ = st.sidebar.selectbox("Question Select",
      ( 'Home',
      'Q1 - Does a country’s R&D expenditure relate to its CO2 emissions?',
      'Q2 - Does a country’s GDP relate to the country’s usage of technology?',
      'Q3 - Does a country’s level of education relate to the country’s use of technology?',
      'Q4 - Does a country’s use of technology relate to the country’s crime rates?',
      'Q5 - How does a country’s use of technology relate to the country’s wealth inequality?'
      )
  )

selectedQ = st.session_state.selectedQ

st.write(selectedQ)

currQ = selectedQ.split(" - ")[0]

#defaults 
min_year = 1990
max_year = 2020

if 'timeslider' not in st.session_state:
    st.session_state.timeslider = min_year
if 'animToggle' not in st.session_state:
    st.session_state.animToggle = True
#note: I don't know why the timeslider doesn't show up for the animation, maybe because too fast to render? - Kaye

animToggle = st.toggle("Toggle Animation?", value=True)

if currQ == "Home":
  pass
else:
  
  if st.session_state.selectedQ != 'Home':
    options = ['All Countries'] + list(df_country)
    country_select = st.sidebar.multiselect(
      'Filter Country:',
      options=options,
      default=['All Countries']
    )

    # TODO - filter functionality per graph x 1, 2, 3, 4, 5

   elif currQ == "Q2":

    # TODO - Q2 graph - could use some visualization later
    # it's also just visualizing cellphone usage per gdp , could also be combined with the pc usage total or another feature in the sidebar
    # I'll work on it - Kaye

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

    tab1, tab2, tab3 = st.tabs(["Scatterplot", "TBD1", "TBD2"])
    with tab1:
      if animToggle:
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
        st.plotly_chart(fig, y='gdp')

  elif currQ == "Q3":

    currYear = int(st.session_state.timeslider)
    tab1, tab2, tab3 = st.tabs(["Scatterplot", "TBD1", "TBD2"])

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

            ax.scatter(x=others['years_of_schooling'], y=others['cellphones_per_100k'])
            ax.scatter(x=highlight['years_of_schooling'], y=highlight['cellphones_per_100k'], color='orange')

          #trying to recreate the bestfit from seaborn regplot
          #TODO - debug line of best fit for errors
          #slope, intercept = np.polyfit(current_year_df['years_of_schooling'], current_year_df['cellphones_per_100k'], 1)
          #line = slope * current_year_df['years_of_schooling'] + intercept
          #ax.plot(current_year_df['years_of_schooling'], line, color='#1f77b4', label='Best Fit Line')
          #TODO shader thing that regplot has

          fig_goes_here.pyplot(fig) # hope this works
          plt.close(fig)
          time.sleep(0.45) 

      else:
        currYear = int(st.session_state.timeslider)
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

          ax.scatter(x=others['years_of_schooling'], y=others['cellphones_per_100k'])
          ax.scatter(x=highlight['years_of_schooling'], y=highlight['cellphones_per_100k'], color='r')

        st.pyplot(fig)
