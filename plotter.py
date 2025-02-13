import pandas as pd
import plotly.express as px

# Replace 'your_file.csv' with the path to your CSV file
csv_file = 'C:/Users/midas/midas_model_experimental_umit/time_series/data/person_eight/person_8_sensor_5_speed_1_amplification_3_trace_1_with_headers.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Add a column for row number (starting at 1)
df['row_number'] = df.index + 1

# Create a line plot using Plotly Express and include the row number in the hover data
fig = px.line(df, x='timestamp', y='voltage', hover_data=['row_number'],
              title='Voltage Over Time (Timestamp in Seconds)')

# Update axis titles if needed
fig.update_xaxes(title_text='Timestamp (s)')
fig.update_yaxes(title_text='Voltage')

# Display the plot
fig.show()