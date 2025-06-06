# EDA process
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("EV_Dataset.csv")

# Clean data
df["EV_Sales_Quantity"] = pd.to_numeric(df["EV_Sales_Quantity"], errors="coerce").fillna(0)
df["State"] = df["State"].str.strip()

# Aggregate by state
state_sales = df.groupby("State")["EV_Sales_Quantity"].sum().sort_values(ascending=False).head(15)

# Plot
plt.figure(figsize=(14, 6))
sns.barplot(x=state_sales.values, y=state_sales.index, palette="rocket")
plt.title("Top 15 States by Total EV Sales")
plt.xlabel("EV Sales Quantity")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# EV sales dataset
vehicle_sales = df.groupby("Vehicle_Type")["EV_Sales_Quantity"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=vehicle_sales.index, y=vehicle_sales.values, palette="pastel")
plt.title("Total EV Sales by Vehicle Type")
plt.xticks(rotation=45)
plt.ylabel("Sales Quantity")
plt.tight_layout()
plt.show()


# Monthly EV sales trend
df["Month_Year"] = df["Month"].astype(str) + "-" + df["Year"].astype(str)
monthly_sales = df.groupby("Month_Year")["EV_Sales_Quantity"].sum()

plt.figure(figsize=(12, 5))
monthly_sales.plot(kind='line', marker='o', color='darkgreen')
plt.title("Monthly EV Sales Trend")
plt.ylabel("EV Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Vehicle type distribution
top_state = df.groupby("State")["EV_Sales_Quantity"].sum().idxmax()
state_df = df[df["State"] == top_state]
state_vehicle_sales = state_df.groupby("Vehicle_Type")["EV_Sales_Quantity"].sum()

plt.figure(figsize=(8, 6))
state_vehicle_sales.plot(kind="pie", autopct="%1.1f%%", startangle=90, cmap="viridis")
plt.title(f"EV Vehicle Type Share in {top_state}")
plt.ylabel("")
plt.tight_layout()
plt.show()


#Geospatial distribution
import geopandas as gpd

# Load India shapefile (you need to download it externally, e.g., Natural Earth or GADM)
# india = gpd.read_file("path_to_shapefile")

gdf = gpd.GeoDataFrame(
    charging_df,
    geometry=gpd.points_from_xy(charging_df.longitude, charging_df.lattitude),
    crs="EPSG:4326"
)

# Plot stations over map
fig, ax = plt.subplots(figsize=(10, 12))
india.plot(ax=ax, color='whitesmoke', edgecolor='gray')
gdf.plot(ax=ax, color='blue', alpha=0.5, markersize=5)
plt.title("EV Charging Station Distribution in India")
plt.tight_layout()
plt.show()


# Charging stations per city
top_cities = charging_df["city"].value_counts().head(15)
plt.figure(figsize=(12, 5))
sns.barplot(x=top_cities.values, y=top_cities.index, palette="mako")
plt.title("Top 15 Cities by Number of Charging Stations")
plt.xlabel("Number of Stations")
plt.ylabel("City")
plt.tight_layout()
plt.show()


# Public Charging Stations by State
pcs_df = pd.read_csv("RS_Session_265_AU_2151_E.csv")
pcs_df.columns = pcs_df.columns.str.strip()
pcs_df.rename(columns={"State/ UT": "State", "No. of PCS": "Charging_Stations"}, inplace=True)

pcs_df["Charging_Stations"] = pd.to_numeric(pcs_df["Charging_Stations"], errors="coerce").fillna(0)
pcs_df.sort_values(by="Charging_Stations", ascending=False, inplace=True)

plt.figure(figsize=(12, 6))
sns.barplot(y="State", x="Charging_Stations", data=pcs_df, palette="flare")
plt.title("Number of Public Charging Stations by State")
plt.tight_layout()
plt.show()
