import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# Set up code checking
import os
print("Setup Complete")

# Path of the file to read
fifa_filepath = "../datasets/fifa.csv"

# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data)

# Path of the file to read
museum_filepath = "../datasets/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data
museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)
import numpy as np
# Replace infinite values with NaN
museum_data.replace([np.inf, -np.inf], np.nan, inplace=True)

print(museum_data.tail()) # Your code here

# Line chart showing the number of visitors to each museum over time
plt.figure(figsize=(12, 6))
plt.title("Monthly Visitors to Los Angeles City Museums")
plt.ylabel("Individuals")
sns.lineplot(data=museum_data)

# Line plot showing the number of visitors to Avila Adobe over time
plt.figure(figsize=(12, 6))
plt.title("Monthly Visitors to Avila Adobe Museum")
plt.ylabel("Visitors")
sns.lineplot(data=museum_data["Avila Adobe"])

# Use FacetGrid to show the visitor trend for Avila Adobe, separated by years
avila = museum_data.reset_index()[["Date", "Avila Adobe"]].copy()
avila["Year"] = avila["Date"].dt.year
avila["Month"] = avila["Date"].dt.month
#group by year and month
avila = avila.groupby(["Year", "Month"]).agg({"Avila Adobe": "sum"}).reset_index()

g = sns.FacetGrid(avila, col="Year", col_wrap=3, height=4)
g.map(sns.lineplot, "Month", "Avila Adobe")
# Show the plot
plt.show()

# Pivot the avila DataFrame to have years as rows and months as columns
avila_pivot = avila.pivot(index="Year", columns="Month", values="Avila Adobe")

plt.figure(figsize=(10, 6))
sns.heatmap(avila_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Avila Adobe Visitors: Years by Months")
plt.xlabel("Month")
plt.ylabel("Year")
plt.show()


# Path of the file to read
flight_filepath = "../datasets/flight_delays.csv"

# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col="Month")
# Print the data
flight_data

# Bar chart showing average arrival delay for Spirit Airlines flights by month
# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
# palette="tab20" is used to set the color palette
#sns.barplot(x=flight_data.index, y=flight_data['NK'], palette="tab20")
sns.barplot(x=flight_data.index, y=flight_data['NK'], palette=sns.color_palette("husl", len(flight_data)))

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")

# Heatmap showing average arrival delay for each airline by month
# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
# sns.heatmap - This tells the notebook that we want to create a heatmap.
# data=flight_data - This tells the notebook to use all of the entries in flight_data to create the heatmap.
# annot=True - This ensures that the values for each cell appear on the chart. (Leaving this out removes the numbers from each of the cells!)
# sns.heatmap(data=flight_data, annot=True)
sns.heatmap(data=flight_data, annot=True, cmap="YlGnBu")

# Add label for horizontal axis
plt.xlabel("Airline")

