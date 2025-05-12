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

st.title('Investigating Intersections of Technology and Industrialization')

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

#Data Preprocessing - Remove empty columns

# note for later - with any percentage based file check for µ and auto replace it to 0.0000N
# personally its feasible to replace by hand but also did something for the convert_count, albeit untested

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
  
#example usage
#internet_usage.update(internet_usage.drop(columns='country').map(convert_count))

#Internet Usage by Percent preprocessing
idx = internet_usage_pct.columns.get_loc('1990')
internet_pct_lists = [internet_usage_pct.columns[0]] + list(internet_usage_pct.columns[idx:])
internet_usage_pct = internet_usage_pct[internet_pct_lists]

def yearCleaning(val):
  val = val.lower().replace(",", "" .strip())
  if val[-1] == "M" or val[-1] == "K":
    pass
  return int(val)

#using gdp as an example
df_country = gdp['country'].unique()


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

  if currQ == "Q1":
    # TODO - Q1 graph - r&d expenditure and co2 emissions
    year_placeholder = st.empty()
    m = fm.Map(zoom_start=7)
    st_folium(m, width=700, height=500)
    
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

  elif currQ == "Q4":
    # TODO - Q4 graph - technology usage and crime rates
    pass
  elif currQ == "Q5":
    # TODO - Q5 graph - technology usage and wealth inequality
    pass

  st.slider(
      'Select a year',
      min_value=min_year,
      max_value=max_year,
      value = st.session_state.timeslider,
      key="timeslider",
      step=1 
  )

st.write("Credits to Gapminder for the datasets behind this project")