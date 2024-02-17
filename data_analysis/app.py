import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load  and validation data
train_data = pd.read_csv('csv_data/trainP.csv')
validation_data = pd.read_csv('csv_data/valP.csv')

# Function to plot distribution


def plot_distribution(dataframe, column_name, title, x_label, y_label):
    value_counts = dataframe[column_name].value_counts()
    uniq_category = value_counts.index
    value_cat = value_counts.values.tolist()
    data = {'cat': uniq_category, 'values': value_cat}
    df_viz = pd.DataFrame(data)

    # Plot value counts as bar plot
    fig = px.bar(
        df_viz,
        x='cat',
        y='values',
        title=title,
        labels={
            'cat': x_label,
            'values': y_label},
        color='cat',
        color_discrete_sequence=px.colors.qualitative.Set3)

    # Add value counts as text on top of each bar
    fig.update_traces(texttemplate='%{y}', textposition='outside')

    # Beautify layout
    fig.update_layout(
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        title=title,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white')
    )

    return fig


# Function to plot histogram

# Function to plot histogram

def plot_histogram(dataframe, title, x_label, y_label):
    fig = px.histogram(
        x=dataframe,
        title=title,
        labels={
            'value': x_label,
            'count': y_label},
        opacity=0.7,
        color_discrete_sequence=['#00CC96'])
    fig.update_traces(
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.8)
    fig.update_layout(showlegend=False)

    # Add vertical line for mean
    mean_val = np.mean(dataframe)
    fig.add_vline(
        x=mean_val,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Mean: {mean_val:.2f}',
        annotation_position='top left')

    # Add vertical line for median
    median_val = np.median(dataframe)
    fig.add_vline(
        x=median_val,
        line_dash='dot',
        line_color='blue',
        annotation_text=f'Median: {median_val:.2f}',
        annotation_position='bottom right')

    # Add quartiles
    quartiles = np.percentile(dataframe, [25, 75])
    for q in quartiles:
        fig.add_vline(x=q, line_dash='dash', line_color='grey')

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig

# Function to plot histogram with logarithmic scale


def plot_histogram_log(dataframe, title, x_label, y_label):
    fig = px.histogram(dataframe, x='area_ratio', color='category',
                       title=title,
                       labels={
                           'area_ratio': x_label,
                           'count': y_label,
                           'category': 'Category'},
                       nbins=20,
                       log_y=True,  # Enable logarithmic scale for y-axis
                       opacity=0.7,
                       color_discrete_sequence=px.colors.qualitative.Set3)

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig


# Set the Seaborn theme
sns.set_theme(style="darkgrid")  # Change 'darkgrid' to other available themes

# Function to generate enhanced box plots


def generate_box_plots(data, column):
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x='category',
        y=column,
        data=data,
        whis=1.5,
        palette='Set3')  # Customize as needed
    # Apply logarithmic scale to y-axis for better visualization
    plt.yscale('log')
    plt.title(f'Box Plot of {column} by Category', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Value (log scale)', fontsize=14)
    # Rotate x-axis labels and adjust font size
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)  # Adjust font size of y-axis labels
    plt.tight_layout()

    plt.legend(fontsize=12)  # Add legend
    st.pyplot(plt.gcf())  # Display the plot
    plt.clf()  # Clear the current figure to avoid overlapping plots
    plt.close()  # Close the plot to release resources


# Streamlit UI
st.title('Object Detection EDA')

# Sidebar for page selection
st.sidebar.markdown("All the filters are independent of each other", unsafe_allow_html=True)

page = st.sidebar.radio("Select Page", (' Data', 'Validation Data'))

if page == ' Training Data':
    st.markdown("""
    ##  Training Data
    In this section, we will analyze the  data for object detection.
    """)
    data = train_data
else:
    st.markdown("""
    ## Validation Data
    In this section, we will analyze the validation data for object detection.
    """)
    data = validation_data


# Display Data
if st.checkbox("Display Data"):
    st.write(data.head())

# Columns Name
if st.checkbox("Columns Name"):
    st.write(data.columns)

# Rows and Columns
dimensions = st.radio(
    "What Dimension Do You Want to Show?", ("Rows", "Columns"))
if dimensions == "Rows":
    st.text("Showing Length of Rows")
    st.write(data.shape[0])
elif dimensions == "Columns":
    st.text("Showing Length of Columns")
    st.write(data.shape[1])

# Missing Values
if st.checkbox("Check for missing values"):
    st.write(data.isnull().sum())

# General Statistics
if st.checkbox("Show Summary of Dataset"):
    st.write(data.describe())

# Sidebar for filtering options
st.sidebar.title("Plots")
option = st.sidebar.selectbox(
    "BASED ON ALL DATA",
    ('None',
     'Bar Chart',
     'Histogram Chart',
     'Scatter Chart',
     'Box PlotChart'))

# Function to plot scatter plot


def plot_scatter(dataframe, x_col, y_col, title, x_label, y_label):
    fig = px.scatter(dataframe, x=x_col, y=y_col, title=title,
                     labels={x_col: x_label, y_col: y_label})
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig


if option == 'Bar Chart':
    # Plot various distributions
    fig1 = plot_distribution(
        data,
        'category',
        'Classes Distribution in  Data',
        'Classes',
        'Count')
    st.plotly_chart(fig1)
    fig2 = plot_distribution(
        data,
        'weather',
        'Weather Distribution in  Data',
        'Weather',
        'Count')
    st.plotly_chart(fig2)
    fig3 = plot_distribution(
        data,
        'scene',
        'Scene Distribution in  Data',
        'Scene',
        'Count')
    st.plotly_chart(fig3)
    fig4 = plot_distribution(
        data,
        'timeofday',
        'Time of Day  Distribution in  Data',
        'Time of Day ',
        'Count')
    st.plotly_chart(fig4)
    fig5 = plot_distribution(
        data,
        'trafficLightColor',
        'Traffic Light Color Distribution in  Data',
        'Traffic Light Color',
        'Count')
    st.plotly_chart(fig5)
    fig6 = plot_distribution(
        data,
        'occluded',
        'Occluded Distribution in  Data',
        'occluded',
        'Count')
    st.plotly_chart(fig6)
    fig7 = plot_distribution(
        data,
        'truncated',
        'Truncated Distribution in  Data',
        'Truncated',
        'Count')
    st.plotly_chart(fig7)



# Markdown text for Object Counts per Image
markdown_object_counts = """
### Object Counts per Image

This histogram displays the distribution of object counts per image in the dataset. It provides insights into the frequency of different object counts, helping understand the complexity and density of objects in the images.
"""

# Markdown text for Ratio of Object Area to Image Area
markdown_area_ratio = """
### Ratio of Object Area to Image Area

The histogram below illustrates the ratio of object area to image area for object detection classes. It offers valuable insights into the relative sizes of detected objects compared to the overall image size. Understanding this ratio is crucial for optimizing object detection models and assessing their performance.
"""


# Markdown text for Box Plot
markdown_box_plot = """
### Box Plot of Object Area, Area Ratio, and Aspect Ratio by Category

This box plot displays the distribution of object area, area ratio, and aspect ratio for each object detection category. It provides insights into the variability and central tendency of these metrics across different categories, aiding in understanding the characteristics of objects within each category.
"""

# Display markdown text for Object Counts per Image
if option == 'Histogram Chart':
    # Plot histogram of object counts per image
    st.markdown(markdown_object_counts)
    object_count_per_image = data['image_name'].value_counts()
    fig2h = plot_histogram(
        dataframe=object_count_per_image,
        title='Distribution of Object Counts per Image',
        x_label='Object Count',
        y_label='Number of Images')
    st.plotly_chart(fig2h)
# Display markdown text for Ratio of Object Area to Image Area
    st.markdown(markdown_area_ratio)
    # Plot histogram for area_ratio with logarithmic scale
    fig3h = plot_histogram_log(
        dataframe=train_data,
        title='Ratio of Object Area to Image Area for Object Detection Classes',
        x_label='Area Ratio',
        y_label='Number of Objects')
    st.plotly_chart(fig3h)


# Plot box plot of object area, area ratio, and aspect ratio by category
if option == 'Box PlotChart':

    # Display markdown text for Box Plot
    st.markdown(markdown_box_plot)

    generate_box_plots(train_data, 'object_area')
    generate_box_plots(train_data, 'area_ratio')
    generate_box_plots(train_data, 'aspect_ratio')


if option == 'Scatter Chart':
    # Plot scatter plot of height versus width
    fig3 = plot_scatter(
        data,
        'width',
        'height',
        'Height vs Width Scatter Plot',
        'Width',
        'Height')
    st.plotly_chart(fig3)


# Close the plot to release resources
plt.close()

st.sidebar.title("Filter Based on Classes")
option_class = st.sidebar.selectbox(
    "Filter",
    ('none',
     'car',
     'traffic sign',
     'traffic light',
     'truck',
     'bus',
     'rider',
     'train'))

# Dictionary to map class names to their respective data
class_mapping = {
    'car': 'car',
    'traffic sign': 'traffic sign',
    'traffic light': 'traffic light',
    'truck': 'truck',
    'bus': 'bus',
    'rider': 'rider',
    'train': 'train'
}

if option_class != 'none':
    class_name = class_mapping.get(option_class)
    if class_name:
        data_n = data[data['category'] == class_name]
        fig1 = plot_distribution(
            data_n,
            'weather',
            f'Weather Distribution for {option_class.capitalize()} in  Data',
            'Weather',
            'Count')
        st.plotly_chart(fig1)
        fig2 = plot_distribution(
            data_n,
            'scene',
            f'Scene Distribution for {option_class.capitalize()} in  Data',
            'Scene',
            'Count')
        st.plotly_chart(fig2)
        fig3 = plot_distribution(
            data_n,
            'timeofday',
            f'Time of Day Distribution for {option_class.capitalize()} in  Data',
            'Time of Day',
            'Count')
        st.plotly_chart(fig3)
        fig4 = plot_distribution(
            data_n,
            'trafficLightColor',
            f'Traffic Light Color Distribution for {option_class.capitalize()} in  Data',
            'Traffic Light Color',
            'Count')
        st.plotly_chart(fig4)
    else:
        st.write("Please select a valid option.")

st.sidebar.title("Filter Based on Truncation")
option_class = st.sidebar.selectbox(
    "Filter", ('none', 'True', 'False'), key="truncation_filter")

# Dictionary to map class names to their respective data
class_mapping = {
    'True': True,
    'False': False
}

if option_class != 'none':
    class_name = class_mapping.get(option_class)
    print(class_name)
    if class_name==True or class_name==False:
        data_n = data[data['truncated'] == class_name]
        print(len(data))
        fig0 = plot_distribution(
            data_n,
            'category',
            f'Weather Distribution for {option_class.capitalize()} in  Data',
            'Category',
            'Count')
        st.plotly_chart(fig0)
        fig1 = plot_distribution(
            data_n,
            'weather',
            f'Weather Distribution for {option_class.capitalize()} in  Data',
            'Weather',
            'Count')
        st.plotly_chart(fig1)
        fig2 = plot_distribution(
            data_n,
            'scene',
            f'Scene Distribution for {option_class.capitalize()} in  Data',
            'Scene',
            'Count')
        st.plotly_chart(fig2)
        fig3 = plot_distribution(
            data_n,
            'timeofday',
            f'Time of Day Distribution for {option_class.capitalize()} in  Data',
            'Time of Day',
            'Count')
        st.plotly_chart(fig3)
        fig4 = plot_distribution(
            data_n,
            'trafficLightColor',
            f'Traffic Light Color Distribution for {option_class.capitalize()} in  Data',
            'Traffic Light Color',
            'Count')
        st.plotly_chart(fig4)
    else:
        st.write("Please select a valid option.")


st.sidebar.title("Filter Based on Occluded")
option_class = st.sidebar.selectbox(
    "Filter", ('none', 'True', 'False'), key="occluded_filter")

# Dictionary to map class names to their respective data
class_mapping = {
    'True': True,
    'False': False
}

if option_class != 'none':
    class_name = class_mapping.get(option_class)
    if class_name==True or class_name==False:
        data_n = data[data['occluded'] == class_name]
        fig0 = plot_distribution(
            data_n,
            'category',
            f'Weather Distribution for {option_class.capitalize()} in  Data',
            'Category',
            'Count')
        st.plotly_chart(fig0)
        fig1 = plot_distribution(
            data_n,
            'weather',
            f'Weather Distribution for {option_class.capitalize()} in  Data',
            'Weather',
            'Count')
        st.plotly_chart(fig1)
        fig2 = plot_distribution(
            data_n,
            'scene',
            f'Scene Distribution for {option_class.capitalize()} in  Data',
            'Scene',
            'Count')
        st.plotly_chart(fig2)
        fig3 = plot_distribution(
            data_n,
            'timeofday',
            f'Time of Day Distribution for {option_class.capitalize()} in  Data',
            'Time of Day',
            'Count')
        st.plotly_chart(fig3)
        fig4 = plot_distribution(
            data_n,
            'trafficLightColor',
            f'Traffic Light Color Distribution for {option_class.capitalize()} in  Data',
            'Traffic Light Color',
            'Count')
        st.plotly_chart(fig4)
    else:
        st.write("Please select a valid option.")




# Close the plot to release resources
plt.close()
