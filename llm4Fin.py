import os
import google.generativeai as genai

from dotenv import load_dotenv

import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import ast
import json
from ratelimit import limits, RateLimitException, sleep_and_retry
from backoff import on_exception, expo
import google.api_core.exceptions as google_exceptions
import streamlit as st
from functools import cache

# load_dotenv()
# api_key=os.getenv("GOOGLE_API_KEY")
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

MINUTE = 60
@sleep_and_retry # If there are more request to this function than rate, sleep shortly
@on_exception(expo, google_exceptions.ResourceExhausted, max_tries=10) # if we receive exceptions from Google API, retry
@limits(calls=60, period=MINUTE)
def llm_gemini(system_prompt, messages, stshow=False):
    messages = system_prompt + "\n\n\n" + messages
    response = model.generate_content(messages)
    if stshow:
        # Display user message in chat message container
        st.chat_message("user").markdown(messages)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": messages})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.text)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.text})
    return response.text

@cache
def get_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return article_text
    except:
        return "Error retrieving article text."


def get_stock_data(ticker, years):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=years*365)

    stock = yf.Ticker(ticker)

    # Retrieve historical price data
    hist_data = stock.history(start=start_date, end=end_date)

    # Retrieve balance sheet
    balance_sheet = stock.balance_sheet

    # Retrieve financial statements
    financials = stock.financials

    # Retrieve news articles
    news = stock.news

    return hist_data, balance_sheet, financials, news


def get_claude_comps_analysis(ticker, hist_data, balance_sheet, financials, news):
    system_prompt = f"You are a financial analyst assistant. Analyze the given data for {ticker} and suggest a few comparable companies to consider. Do so in a Python-parseable list."

    news = ""

    for article in news:
        article_text = get_article_text(article['link'])
        news = news + f"\n\n---\n\nTitle: {article['title']}\nText: {article_text}"

    messages = f"Historical price data:\n{hist_data.tail().to_string()}\n\nBalance Sheet:\n{balance_sheet.to_string()}\n\nFinancial Statements:\n{financials.to_string()}\n\nNews articles:\n{news.strip()}\n\n----\n\nNow, suggest a few comparable companies to consider, in a Python-parseable list. Return nothing but the list. Make sure the companies are in the form of their tickers."

    response_text = llm_gemini(system_prompt, messages)
    return ast.literal_eval(response_text)


def compare_companies(main_ticker, main_data, comp_ticker, comp_data):
    system_prompt = f"You are a financial analyst assistant. Compare the data of {main_ticker} against {comp_ticker} and provide a detailed comparison, like a world-class analyst would. Be measured and discerning. Truly think about the positives and negatives of each company. Be sure of your analysis. You are a skeptical investor."

    messages = f"Data for {main_ticker}:\n\nHistorical price data:\n{main_data['hist_data'].tail().to_string()}\n\nBalance Sheet:\n{main_data['balance_sheet'].to_string()}\n\nFinancial Statements:\n{main_data['financials'].to_string()}\n\n----\n\nData for {comp_ticker}:\n\nHistorical price data:\n{comp_data['hist_data'].tail().to_string()}\n\nBalance Sheet:\n{comp_data['balance_sheet'].to_string()}\n\nFinancial Statements:\n{comp_data['financials'].to_string()}\n\n----\n\nNow, provide a detailed comparison of {main_ticker} against {comp_ticker}. Explain your thinking very clearly."

    response_text = llm_gemini(system_prompt, messages)
    return response_text

def get_sentiment_analysis(ticker, news):
    system_prompt = f"You are a sentiment analysis assistant. Analyze the sentiment of the given news articles for {ticker} and provide a summary of the overall sentiment and any notable changes over time. Be measured and discerning. You are a skeptical investor."

    news_text = ""
    for article in news:
        article_text = get_article_text(article['link'])
        timestamp = datetime.fromtimestamp(article['providerPublishTime']).strftime("%Y-%m-%d")
        news_text += f"\n\n---\n\nDate: {timestamp}\nTitle: {article['title']}\nText: {article_text}"

    messages = f"News articles for {ticker}:\n{news_text}\n\n----\n\nProvide a summary of the overall sentiment and any notable changes over time."
    response = model.generate_content(messages)
    response_text = llm_gemini(system_prompt, messages)
    return response_text

@cache
def get_analyst_ratings(ticker):
    stock = yf.Ticker(ticker)
    recommendations = stock.upgrades_downgrades

    if recommendations is None or recommendations.empty:
        return "No analyst ratings available."

    latest_rating = recommendations.iloc[0]

    GradeDate = latest_rating.name
    firm = latest_rating.get('Firm', 'N/A')
    to_grade = latest_rating.get('ToGrade', 'N/A')
    action = latest_rating.get('Action', 'N/A')

    rating_summary = f"Latest analyst rating for {ticker}:\n Grade Date: {GradeDate}\n Firm: {firm}\n To Grade: {to_grade}\n Action: {action}"


    return rating_summary

@cache
def get_industry_analysis(ticker):

    ### update to use search to find recent data!!

    stock = yf.Ticker(ticker)
    industry = stock.info['industry']
    sector = stock.info['sector']

    system_prompt = f"You are an industry analysis assistant. Provide an analysis of the {industry} industry and {sector} sector, including trends, growth prospects, regulatory changes, and competitive landscape. Be measured and discerning. Truly think about the positives and negatives of the stock. Be sure of your analysis. You are a skeptical investor."

    messages = f"Provide an analysis of the {industry} industry and {sector} sector." 

    response_text = llm_gemini(system_prompt, messages)
    return response_text


def get_final_analysis(ticker, comparisons, sentiment_analysis, analyst_ratings, industry_analysis):
    system_prompt = f"You are a financial analyst providing a final investment recommendation for {ticker} based on the given data and analyses. Be measured and discerning. Truly think about the positives and negatives of the stock. Be sure of your analysis. You are a skeptical investor."

    messages = f"Ticker: {ticker}\n\nComparative Analysis:\n{json.dumps(comparisons, indent=2)}\n\nSentiment Analysis:\n{sentiment_analysis}\n\nAnalyst Ratings:\n{analyst_ratings}\n\nIndustry Analysis:\n{industry_analysis}\n\nBased on the provided data and analyses, please provide a comprehensive investment analysis and recommendation for {ticker}. Consider the company's financial strength, growth prospects, competitive position, and potential risks. Provide a clear and concise recommendation on whether to buy, hold, or sell the stock, along with supporting rationale."
    response_text = llm_gemini(system_prompt, messages)
    return response_text

@cache
def generate_ticker_ideas(industry):
    system_prompt = f"You are a financial analyst assistant. Generate a list of 5 ticker symbols for major companies in the {industry} industry, as a Python-parseable list."

    messages = f"Please provide a list of 5 ticker symbols for major companies in the {industry} industry as a Python-parseable list. Only respond with the list, no other text."

    response_text = llm_gemini(system_prompt, messages)
    ticker_list = ast.literal_eval(response_text)
    return [ticker.strip() for ticker in ticker_list]


def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d', interval='1m')
    return data['Close'][-1]

def rank_companies(industry, analyses, prices):
    system_prompt = f"You are a financial analyst providing a ranking of companies in the {industry} industry based on their investment potential. Be discerning and sharp. Truly think about whether a stock is valuable or not. You are a skeptical investor."

    analysis_text = "\n\n".join(
        f"Ticker: {ticker}\nCurrent Price: {prices.get(ticker, 'N/A')}\nAnalysis:\n{analysis}"
        for ticker, analysis in analyses.items()
    )

    messages = f"Industry: {industry}\n\nCompany Analyses:\n{analysis_text}\n\nBased on the provided analyses, please rank the companies from most attractive to least attractive for investment. Provide a professional rationale for your ranking. In each rationale, include the current price (if available) and a price target."
    response_text = llm_gemini(system_prompt, messages)
    # st.write(analysis_text)
    return response_text

def translate_cn(messages):
    system_prompt = "你是一位精通简体中文的专业翻译，尤其擅长将专业学术论文翻译成浅显易懂的科普文章。请你帮我将以下英文段落翻译成中文，风格与中文科普读物相似。\
        规则：- 翻译时要准确传达原文的事实和背景。\
            - 输入格式为 Markdown 格式，输出格式也必须保留原始 Markdown 格式"

    response_text = llm_gemini(system_prompt, messages)
    return response_text

def rank_byLLM(industry, tickers, years=1):
    # Perform analysis for each company
        analyses = {}
        prices = {}
        with st.status("Analyzing the Stocks...", expanded=True) as status:
            for i, ticker in enumerate(tickers):
                try:
                    st.write(f"{i+1}. Analyzing {ticker}...")
                    hist_data, balance_sheet, financials, news = get_stock_data(ticker, years)
                    main_data = {
                        'hist_data': hist_data,
                        'balance_sheet': balance_sheet,
                        'financials': financials,
                        'news': news
                    }
                    tab1,tab2,tab3,tab4 = st.tabs(["Analyst", "Sentiment", "Industry", "Summary"])
                    with tab1:
                        analyst_ratings = get_analyst_ratings(ticker)
                        st.write(analyst_ratings)
                    with tab2:
                        sentiment_analysis = get_sentiment_analysis(ticker, news)
                        st.write(sentiment_analysis)
                    with tab3:
                        industry_analysis = get_industry_analysis(ticker)
                        st.write(industry_analysis)
                    with tab4:
                        final_analysis = get_final_analysis(ticker, {}, sentiment_analysis, analyst_ratings, industry_analysis)
                        st.write(final_analysis)
                    analyses[ticker] = final_analysis
                    prices[ticker] = round(get_current_price(ticker), 2)

                except Exception as e:
                    print(e)
            status.update(label="Analyses completed!", state="complete", expanded=False)


        # Rank the companies based on their analyses
        st.subheader("Rank the companies based on their analyses")
        ranking = rank_companies(industry, analyses, prices)
        print(f"\nRanking of Companies in the {industry} Industry:")
        print(ranking)
        st.markdown(ranking)

if __name__ == "__main__":
    st.title("GENAI in Financial analysis")
    industry = st.text_input("Input Industry")

    if st.button("Analyze") and industry:
        years = 1 # int(input("Enter the number of years for analysis: "))


        # Generate ticker ideas for the industry
        tickers = generate_ticker_ideas(industry)
        st.write(f"\n 5 Ticker Ideas for {industry} Industry:     "+ ", ".join(tickers))

        rank_byLLM(industry, tickers)
