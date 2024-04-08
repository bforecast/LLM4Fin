from yahoo_fin import news
import streamlit as st

st.write(news.get_yf_rss('MSFT'))