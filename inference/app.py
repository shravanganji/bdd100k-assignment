import pandas as pd
import plotly.express as px
import streamlit as st
from io import StringIO

# Provided data
new_data = """Category,Precision,Recall,IoU,AP
pedestrian,0.0,0.0,0.0,0.0
rider,0.001860416861326473,0.0016147919876733443,0.026151263071757625,6.776557646459123e-06
car,0.006588486910972667,0.003940000000000001,0.0620855045690976,6.827306817826244e-05
truck,0.05114412675844106,0.024474000000000003,0.12732470473035876,0.0021974316907494825
bus,0.08005532464723805,0.031188,0.12830838563243568,0.004704280827258397
train,0.0,0.0,0.0,0
motorcycle,0.0,0.0,0.0,0.0
bicycle,0.0,0.0,0.0,0.0
traffic light,0.0020391701579563614,0.0012670000000000005,0.0035454053853680055,5.12001144462955e-06
traffic sign,0.0,0.0,0.011751620077821172,0.0
Overall,0.014168752533593459,0.006248379198767334,0.035916688346683885,0.0006981882155277231
"""

# Convert the provided data to a DataFrame
new_df = pd.read_csv(StringIO(new_data))

# Set 'Category' as the index
new_df.set_index('Category', inplace=True)

# Unstack the DataFrame
pivot_table = new_df.unstack().reset_index()

# Rename columns
pivot_table.columns = ['Metrics', 'Category', 'Value']
df1 = pivot_table.reset_index()

# Streamlit app
st.title('Category Metrics Analysis')
st.write("Note: All the data is static which is generated while testing in my local system, As it is time consuming to calculate metrics while running the docker image, the values are copied here")

# Add a selectbox to the sidebar
category = st.sidebar.selectbox(
    'Select a category',
    df1['Category'].unique(), index=10
)

# Filter the DataFrame based on the selected category
filtered_df = df1[df1['Category'] == category]

# Plot
fig = px.bar(filtered_df, x='Metrics', y='Value', color='Category', barmode='group')
st.plotly_chart(fig)

# Display images
st.subheader('Predicted images:')
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]

for image_path in image_paths:
    st.image(image_path, caption='Image', use_column_width=True)

