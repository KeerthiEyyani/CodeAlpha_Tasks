import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create screenshots folder
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Load Dataset
df = pd.read_csv("Unemployment_Rate.csv")

# Display Dataset Information
print("First 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Save Dataset Preview as PNG
plt.figure(figsize=(10, 2))
plt.axis('off')

table = plt.table(
    cellText=df.head().values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)

plt.savefig(
    "screenshots/dataset_preview.png",
    bbox_inches='tight'
)
plt.close()

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])

# Unemployment Trend Analysis
plt.figure(figsize=(10, 5))

plt.plot(
    df['Date'],
    df['Estimated Unemployment Rate (%)'],
    marker='o'
)

plt.title("Unemployment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)

plt.savefig(
    "screenshots/unemployment_trend.png"
)

plt.close()

# State-wise Analysis
state_avg = df.groupby('Region')[
    'Estimated Unemployment Rate (%)'
].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=state_avg.values,
    y=state_avg.index
)

plt.title("Average Unemployment Rate by State")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    "screenshots/statewise_analysis.png"
)

plt.close()

# Covid Impact Analysis
covid_data = df[
    (df['Date'] >= '2020-03-01') &
    (df['Date'] <= '2020-12-31')
]

plt.figure(figsize=(10, 5))

plt.plot(
    covid_data['Date'],
    covid_data['Estimated Unemployment Rate (%)'],
    marker='o'
)

plt.title("Covid-19 Impact on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)

plt.savefig(
    "screenshots/covid_impact.png"
)

plt.close()

# Key Insights
print("\nKey Insights:")
print(
    "Average Unemployment Rate:",
    round(
        df['Estimated Unemployment Rate (%)'].mean(),
        2
    ),
    "%"
)

print(
    "Highest Unemployment Rate:",
    df['Estimated Unemployment Rate (%)'].max(),
    "%"
)

print(
    "Lowest Unemployment Rate:",
    df['Estimated Unemployment Rate (%)'].min(),
    "%"
)

print("\nAnalysis Completed Successfully!")

print("\nGenerated Files:")
print("screenshots/dataset_preview.png")
print("screenshots/unemployment_trend.png")
print("screenshots/statewise_analysis.png")
print("screenshots/covid_impact.png")