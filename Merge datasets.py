# load data
import pandas as pd

ev_df = pd.read_csv("EV_Dataset.csv")  # From 'EV by State'
charging_df = pd.read_csv("ev-charging-stations-india.csv")  # From charging stations
top_cities_df = pd.read_csv("top 5 cities.csv")  # Top 5 Cities EV data

# Preprocess each file
ev_df.columns = ['State', 'EV_Type', 'Total_EVs', 'Year']  # Rename for clarity
state_ev_summary = ev_df.groupby('State')['Total_EVs'].sum().reset_index()

# charging station data
charging_df.columns = charging_df.columns.str.strip().str.replace(" ", "_")
state_charging_summary = charging_df.groupby('State/UT')['Station_Name'].count().reset_index()
state_charging_summary.columns = ['State', 'Charging_Stations']

# Merge EV and charging data
merged_df = pd.merge(state_ev_summary, state_charging_summary, on='State', how='inner')

# add top cities info
top_cities_df.columns = ['City', 'Total_EVs_City', 'Charging_Stations_City']
# Merge by state name if applicable or keep separate for city-wise clustering

# Feature engineering
merged_df['EVs_per_Station'] = merged_df['Total_EVs'] / merged_df['Charging_Stations']
merged_df.to_csv("merged_ev_data.csv", index=False)  # Save for clustering
