import pandas as pd
import plotly.express as px

# Convert the data to a Pandas DataFrame
from io import StringIO

df = pd.read_csv(StringIO(data), parse_dates=['timestamp'])
df.columns = df.columns.str.strip()

# Create a plot
fig = px.line(df, x='timestamp', y=['utilization.gpu [%]', 'utilization.memory [%]'], 
              title="GPU and Memory Utilization Over Time", 
              labels={'timestamp': 'Time', 'value': 'Utilization (%)'},
              markers=True)

# Show the plot
fig.show()
