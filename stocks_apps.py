import plotly_express as px
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
import streamlit as st
from datetime import date
from plotly import graph_objs as go
import warnings

START = "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app title
st.title("Stock Prediction in USD($)")

# Select stock and prediction time period
stocks = ("AMD", "AAPL", "AMZN", "EBAY", "TM", "SOFI", "OPEN", "F")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)
n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365


def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data..")
data = load_data(selected_stocks)
data_load_state.text("Loading data...finished")

st.subheader('Raw Data')
st.write(data.tail(100))


#fig = go.Figure()
#fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'].round(2),mode='lines', name='open'))
#fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'].round(2),mode='lines', name='close'))
#fig.show()
#fig.layout.update(title_text='TIme Series',xaxis_rangeslider_visible=True)
#st.plotly_chart(fig2)

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})
df_train.columns = ['ds', 'y']
m=Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecasted Data')
st.write(round(forecast.tail(100),2))

st.write('forecast')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
fig1.update_xaxes(title_text = 'Future Time')
fig1.update_yaxes(title_text = 'Price_Forecast')
