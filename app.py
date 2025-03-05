import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d, DatetimeTickFormatter
from bokeh.models.ranges import DataRange1d
from scipy.stats import norm
from datetime import datetime, timedelta

# Set the page layout to wide
st.set_page_config(layout="wide")

# Custom loss function
def custom_loss(y_true, y_pred):
    return -y_pred.log_prob(y_true)

# Load the LSTM model with custom objects
@st.cache_resource
def load_saved_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'<lambda>': custom_loss, 'DistributionLambda': tfp.layers.DistributionLambda})

def lstm_shape_train_test(dataset, timesteps):    
    features = dataset.columns[:-1]

    # Find the minimum and maximum timestamps in the dataset
    min_timestamp = dataset.index.min()
    max_timestamp = dataset.index.max()

    # Generate a complete time index from the minimum to the maximum timestamp
    complete_time_index = pd.date_range(start=min_timestamp, end=max_timestamp, freq='1H')

    # fill in data gaps so that when that data is shifted, the time series is not misaligned
    dataset = dataset.reindex(complete_time_index)

    # set up dataframe to consider the number of timesteps for each prediction
    for i in range(1,timesteps):
        for j in range(len(features)):
            dataset[f'{features[j]}(t-{i})'] = dataset[features[j]].shift(i)

    if 'y' in dataset.columns:
        dataset = dataset[[col for col in dataset.columns if col != 'y'] + ['y']]

    multi_step_dataset = dataset.dropna()
    return multi_step_dataset

def get_LSTM_shaped_forcing(lstm_forcing_df):

    dataset = lstm_shape_train_test(lstm_forcing_df, timesteps=1)
    dataset = dataset.dropna()

    X_val = dataset
    features = X_val.columns

    X_val = X_val.values
    X_val = X_val.reshape((X_val.shape[0], 1, len(features)))

    return X_val, dataset


def get_predictions_df_generalized(model, X_val, dataset, hour, exposure_flow=8.5):

    u = np.reshape(model(X_val).mean().numpy(),-1)
    sigma = np.reshape(model(X_val).scale.numpy(),-1)

    median = 10**u
    mode = 10**(u-(sigma**2))
    mean = 10**(u+((sigma**2)/2))
    variance = 10**(2*u + 2*(sigma**2)) - 10**(2*u + sigma**2)

    dataset[f'mean_{hour}'] = mean
    dataset[f'variance_{hour}'] = variance
    dataset[f'median_{hour}'] = median
    dataset[f'exceedance_probability_{hour}'] = 1 - norm.cdf(np.log10(exposure_flow), loc=u, scale=sigma) 


    new_index = pd.date_range(start=dataset.index.min(), end=(dataset.index.max() + timedelta(hours=hour)), freq='1H')
    dataset = dataset.reindex(new_index)
    
    dataset[f'mean_{hour}'] = dataset[f'mean_{hour}']
    dataset[f'variance_{hour}'] = dataset[f'variance_{hour}']
    dataset[f'median_{hour}'] = dataset[f'median_{hour}']
    dataset[f'percentile_95_{hour}'] = norm.ppf(0.95, loc=dataset[f'mean_{hour}'], scale=dataset[f'variance_{hour}'])
    dataset[f'exceedance_probability_{hour}'] = dataset[f'exceedance_probability_{hour}']
    norm.ppf([0.1, 0.9, 0.8, 0.5], loc=25, scale=4)

    return dataset

# Streamlit UI
st.title("Data Driven Flood Forecasting Prediction Dashboard")
st.write("Adjust the input parameters and see how the model responds on July 6th.")

# Load model
model_path = "site_crossing_model_24H.h5"
model = load_saved_model(model_path)

# User Inputs
#sequence_length = st.slider("Sequence Length", min_value=10, max_value=100, value=50, step=5)
Hindcast_Precitipation = st.slider("Past 6 Hour Precipitation", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
Forecast_Precipitation = st.slider("Cumulative 24 Hour Forecast Precipitation", min_value=0.0, max_value=200.0, value=0.0, step=5.0)
Temperature = st.slider("Max 24 Hour Forecast Temperature", min_value=5.0, max_value=40.0, value=25.0, step=1.0)
Snow_Pack = st.slider("Snow Water Equivalent", min_value=0.0, max_value=0.1, value=0.05, step=0.005)

exposure_flow = st.number_input("Exposure Flow", min_value=0, max_value=20, value=10, step=1)

# Create input data
forcing = pd.read_csv(f'lstm_forcing_site_crossing_July11_Aug_11.csv', parse_dates=['Datetime']).set_index('Datetime').dropna()

# hindcast
forcing.loc[forcing.index[260+7:266+7], 'capa_w_avg'] = Hindcast_Precitipation  / 6
# forecast
for i in range(24):
    forcing.iloc[265+7+i, 5:29-i] = Forecast_Precipitation / 24
# temperature
max_temp = forcing.loc[forcing.index[265+7:289+7], 'temp_1 days 00:00:00'].max()
conversion = Temperature / max_temp
forcing.loc[forcing.index[265+9:289+9], forcing.columns[53:77]] = forcing.loc[forcing.index[265+9:289+9], forcing.columns[53:77]] * conversion
# snow pack
forcing.loc[forcing.index[265+7:289+7], 'snodas_mean'] = Snow_Pack

X_val, dataset = get_LSTM_shaped_forcing(forcing)

# Run prediction
# Run prediction
if st.button("Predict"):
    df = get_predictions_df_generalized(model, X_val, dataset, hour=24, exposure_flow=exposure_flow)
    df.index = df.index - timedelta(hours=7)
    df = df.loc['2024-06-25 00:00:00':'2024-07-07 00:00:00']

    # Store the current dataset in session state to keep track of previous predictions
    if "prev_df" in st.session_state:
        prev_df = st.session_state.prev_df
    else:
        prev_df = None  # No previous data yet

    # Update session state with the new dataset
    st.session_state.prev_df = df.copy()

    p_x_range = DataRange1d(start=df.index.min(), end=df.index.max() + pd.Timedelta(days=1))

    p1 = figure(width=1800, height=500, x_axis_type="datetime", title=' ', x_range=p_x_range, y_axis_label='Flow (m3/s)')

    # Plot previous prediction (if available)
    if prev_df is not None:
        p1.line(x=prev_df.index, y=prev_df['median_24'], color='orange', line_width=3, legend_label='Previous Prediction (Median)')
        p1.line(x=prev_df.index, y=prev_df['percentile_95_24'], color='orange', line_width=3, line_dash='dashed', legend_label='Previous 95th Percentile')

    # Plot current prediction
    p1.line(x=df.index, y=df['median_24'], color='darkblue', line_width=3, legend_label='Current Prediction (Median)')
    p1.line(x=df.index, y=df['percentile_95_24'], color='darkblue', line_width=3, line_dash='dashed', legend_label='Current 95th Percentile')

    # Plot threshold probability
    p1.varea(x=df['exceedance_probability_24'].index, 
             y2=df['exceedance_probability_24']*100, 
             y1=[0 for _ in range(len(df.index))], 
             y_range_name="exposure_percentile", alpha=0.5, color='grey', legend_label='Threshold Probability')

    # Create a new y-axis for the exposure percentile
    p1.extra_y_ranges = {"exposure_percentile": Range1d(start=100, end=0)}
    p1.add_layout(LinearAxis(y_range_name="exposure_percentile", axis_label='\n\n\n\nThreshold Probability'), 'right')

    # Add threshold line
    p1.line(x=df.index, y=11, color='red', line_width=2, line_dash='dashed', line_alpha=0.5, legend_label='Threshold')

    # Formatting
    p1.yaxis.axis_label_text_font_size = '20px'
    p1.title.text_font_size = "20px"
    p1.yaxis.major_label_text_font_size = '15px'
    p1.xaxis.major_label_text_font_size = '15px'

    p1.xaxis.ticker.desired_num_ticks = 20
    p1.xaxis[0].formatter = DatetimeTickFormatter(days='%b %d')

    leg = p1.legend[0]
    p1.add_layout(leg, 'right')
    p1.legend.label_text_font_size = '15px'

    p2 = figure(width=1800, height=600, x_axis_type="datetime", title=' ', 
                y_axis_label='Hourly Precipitation (mm)', 
                y_range=(10, 0), 
                x_range=DataRange1d(start=df.index.min(), end=df.index.max() + pd.Timedelta(days=1)))

    # forecast precip
    p2.vbar(x=df.index, top=df['0 days 01:00:00'],  width=2, color='#6a51a3', legend_label='Forecasted Precipitation (mm/hr)')

    # hindcast precip
    p2.vbar(x=df.index, top=df['capa_w_avg'], color='blue', width=2, legend_label='Observed Precipitation (mm/hr)')

    # swe
    p2.varea(x=df['snodas_mean'].dropna().index, 
                y1=1000*df['snodas_mean'].dropna(), 
                y2=[0 for i in range(len(df['snodas_mean'].dropna().index))], 
                y_range_name="snow", alpha=0.5, color='darkturquoise', legend_label='Snow Water Equivalent (mm)')

    p2.add_layout(LinearAxis(y_range_name="snow", axis_label='Snow Water Equivalent (mm)'), 'right')

    # soil moisture
    # p2.line(x=df.index, y=df['sm_surface'], color=brbg[-3], line_width=3, legend_label='Soil Moisture Surface',y_range_name="sm")
    # p2.line(x=df.index, y=df['sm_rootzone'], color=brbg[-2], line_width=3, legend_label='Soil Moisture Rootzone',y_range_name="sm")

    # p2.add_layout(LinearAxis(y_range_name="sm", axis_label='Soil Moisture'), 'right')

    # temp forecast

    p2.line(x=df.index, y=df['temp_0 days 01:00:00'],  width=2, color='#e34a33', legend_label='Forecasted Temperature (C)', y_range_name="temp")

    p2.add_layout(LinearAxis(y_range_name="temp", axis_label='Temperature'), 'right')

    p2.extra_y_ranges = {"precip": Range1d(start=20, end=0),
                        "snow": Range1d(start=0, end=500), 
                        "sm": Range1d(start=0.0, end=0.4),
                        "temp": Range1d(start=-10, end=40)}


    p2.yaxis.axis_label_text_font_size = '20px'
    p2.title.text_font_size = "20px"
    p2.yaxis.major_label_text_font_size = '15px'
    p2.xaxis.major_label_text_font_size = '15px'

    p2.xaxis.ticker.desired_num_ticks = 20
    p2.xaxis[0].formatter = DatetimeTickFormatter(days = '%b %d')

    leg = p2.legend[0]
    p2.add_layout(leg,'right')
    p2.legend.label_text_font_size = '15px'

    st.bokeh_chart(p1)
    st.bokeh_chart(p2)

    table = df[['capa_w_avg', 'snodas_mean','0 days 01:00:00',
    'temp_0 days 01:00:00','median_24', 'exceedance_probability_24',
    'percentile_95_24']].rename(columns={'capa_w_avg':'Observed Precipitation (mm/hr)',
                                        'snodas_mean':'Snow Water Equivalent (mm)',
                                        '0 days 01:00:00':'Forecasted Precipitation (mm/hr)',
                                        'temp_0 days 01:00:00':'Forecasted Temperature (C)',
                                        'median_24':'Predicted Flow (m3/s)',
                                        'exceedance_probability_24':'Exceedance Probability',
                                        'percentile_95_24':'95th Percentile Flow (m3/s)'})
    
    st.write("Prediction Results:", table)
