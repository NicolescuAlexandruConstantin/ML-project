import folium
import pandas as pd

df = pd.read_csv("earthquake_1995-2023.csv")

avg_lat = df["latitude"].mean()
avg_lon = df["longitude"].mean()

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2)

for _, row in df.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=row.get("sig", ""),
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)


m.save("harta.html")
