import os
import time
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
import asyncio
import asyncache
from cachetools import TTLCache
import aiohttp


# load_dotenv()
# api_key=os.getenv("GOOGLE_API_KEY")
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
RPM_GEMINI = {
    "gemini-1.0-pro":   15,
    "gemini-1.5-pro-latest":   2,

}
gemini_model = "gemini-1.0-pro"
model = genai.GenerativeModel(gemini_model)

MINUTE = 60

@sleep_and_retry # If there are more request to this function than rate, sleep shortly
@on_exception(expo, google_exceptions.ResourceExhausted, max_tries=10) # if we receive exceptions from Google API, retry
@limits(calls=RPM_GEMINI[gemini_model], period=MINUTE)
async def llm_gemini(system_prompt, messages, stshow=False):
    messages = system_prompt + "\n\n\n" + messages
    response = await model.generate_content_async(messages)
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

@asyncache.cached(TTLCache(1024, 36000))
async def get_article_text(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.read()
                soup = BeautifulSoup(content, 'html.parser')
                article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
                return article_text
    except Exception as e:
        print(e)
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
    # st.write(financials)
    # Retrieve news articles
    news = stock.news

    return hist_data, balance_sheet, financials, news


async def get_claude_comps_analysis(ticker, hist_data, balance_sheet, financials, news):
    system_prompt = f"You are a financial analyst assistant. Analyze the given data for {ticker} and suggest a few comparable companies to consider. Do so in a Python-parseable list."

    news = ""

    for article in news:
        article_text = await get_article_text(article['link'])
        news = news + f"\n\n---\n\nTitle: {article['title']}\nText: {article_text}"

    messages = f"Historical price data:\n{hist_data.tail().to_string()}\n\nBalance Sheet:\n{balance_sheet.to_string()}\n\nFinancial Statements:\n{financials.to_string()}\n\nNews articles:\n{news.strip()}\n\n----\n\nNow, suggest a few comparable companies to consider, in a Python-parseable list. Return nothing but the list. Make sure the companies are in the form of their tickers."

    response_text = await llm_gemini(system_prompt, messages)
    return ast.literal_eval(response_text)


async def compare_companies(main_ticker, main_data, comp_ticker, comp_data):
    system_prompt = f"You are a financial analyst assistant. Compare the data of {main_ticker} against {comp_ticker} and provide a detailed comparison, like a world-class analyst would. Be measured and discerning. Truly think about the positives and negatives of each company. Be sure of your analysis. You are a skeptical investor."

    messages = f"Data for {main_ticker}:\n\nHistorical price data:\n{main_data['hist_data'].tail().to_string()}\n\nBalance Sheet:\n{main_data['balance_sheet'].to_string()}\n\nFinancial Statements:\n{main_data['financials'].to_string()}\n\n----\n\nData for {comp_ticker}:\n\nHistorical price data:\n{comp_data['hist_data'].tail().to_string()}\n\nBalance Sheet:\n{comp_data['balance_sheet'].to_string()}\n\nFinancial Statements:\n{comp_data['financials'].to_string()}\n\n----\n\nNow, provide a detailed comparison of {main_ticker} against {comp_ticker}. Explain your thinking very clearly."

    response_text = await llm_gemini(system_prompt, messages)
    return response_text

async def get_sentiment_analysis(ticker, news):
    system_prompt = f"You are a sentiment analysis assistant. Analyze the sentiment of the given news articles for {ticker} and provide a summary of the overall sentiment and any notable changes over time. Be measured and discerning. You are a skeptical investor."

    news_text = ""
    for article in news:
        article_text = await get_article_text(article['link'])
        timestamp = datetime.fromtimestamp(article['providerPublishTime']).strftime("%Y-%m-%d")
        news_text += f"\n\n---\n\nDate: {timestamp}\nTitle: {article['title']}\nText: {article_text}"

    messages = f"News articles for {ticker}:\n{news_text}\n\n----\n\nProvide a summary of the overall sentiment and any notable changes over time."
    response_text = await llm_gemini(system_prompt, messages)
    return response_text

@asyncache.cached(TTLCache(1024, 36000))
async def get_analyst_ratings(ticker):
    """
    This Python function retrieves the latest analyst rating for a given stock ticker.
    
    :param ticker: The code you provided is a Python function that retrieves the latest analyst rating
    for a given stock ticker symbol. The function uses the yfinance library to fetch analyst ratings
    data for the specified stock
    :return: The function `get_analyst_ratings` returns a summary of the latest analyst rating for a
    given stock ticker. The summary includes the grade date, firm, to grade, and action of the latest
    analyst rating. If there are no analyst ratings available for the given stock ticker, it returns "No
    analyst ratings available."
    """
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

@asyncache.cached(TTLCache(1024, 36000))
async def get_industry_analysis(ticker):
    """
    This Python function retrieves industry analysis for a given stock ticker by prompting the user to
    provide an analysis of the industry and sector.
    
    :param ticker: The `ticker` parameter is a stock symbol that represents a publicly traded company on
    the stock market. It is used to identify and retrieve information about a specific company's stock,
    such as its price, historical data, financial information, and more. In the provided code snippet,
    the `ticker` parameter
    :return: The function `get_industry_analysis` is returning the response text generated by the
    `llm_gemini` function after providing an analysis prompt for the industry and sector based on the
    provided stock ticker.
    """

    ### update to use search to find recent data!!

    stock = yf.Ticker(ticker)
    industry = stock.info['industry']
    sector = stock.info['sector']

    system_prompt = f"You are an industry analysis assistant. Provide an analysis of the {industry} industry and {sector} sector, including trends, growth prospects, regulatory changes, and competitive landscape. Be measured and discerning. Truly think about the positives and negatives of the stock. Be sure of your analysis. You are a skeptical investor."

    messages = f"Provide an analysis of the {industry} industry and {sector} sector." 

    response_text = await llm_gemini(system_prompt, messages)
    return response_text


async def get_final_analysis(ticker, comparisons, sentiment_analysis, analyst_ratings, industry_analysis):
    system_prompt = f"You are a financial analyst providing a final investment recommendation for {ticker} based on the given data and analyses. Be measured and discerning. Truly think about the positives and negatives of the stock. Be sure of your analysis. You are a skeptical investor."

    messages = f"Ticker: {ticker}\n\nComparative Analysis:\n{json.dumps(comparisons, indent=2)}\n\nSentiment Analysis:\n{sentiment_analysis}\n\nAnalyst Ratings:\n{analyst_ratings}\n\nIndustry Analysis:\n{industry_analysis}\n\nBased on the provided data and analyses, please provide a comprehensive investment analysis and recommendation for {ticker}. Consider the company's financial strength, growth prospects, competitive position, and potential risks. Provide a clear and concise recommendation on whether to buy, hold, or sell the stock, along with supporting rationale."
    response_text = await llm_gemini(system_prompt, messages)
    return response_text

@asyncache.cached(TTLCache(1024, 36000))
async def generate_ticker_ideas(industry):
    """
    The function `generate_ticker_ideas` prompts the user to provide a list of 5 ticker symbols for
    major companies in a specified industry and returns the parsed list of ticker symbols.
    
    :param industry: I see that you have a function `generate_ticker_ideas` that prompts the user to
    provide a list of 5 ticker symbols for major companies in a specific industry. The function then
    processes the response and returns a cleaned list of ticker symbols
    :return: A list of 5 ticker symbols for major companies in the specified industry, parsed from the
    user's response.
    """
    system_prompt = f"You are a financial analyst assistant. Generate a list of 5 ticker symbols for major companies in the {industry} industry, as a Python-parseable list."

    messages = f"Please provide a list of 5 ticker symbols for major companies in the {industry} industry as a Python-parseable list. Only respond with the list, no other text."

    response_text = await llm_gemini(system_prompt, messages)
    response_text = response_text.replace("```python", "").strip()
    response_text = response_text.replace("```", "").strip()

    ticker_list = ast.literal_eval(response_text)
    
    return [ticker.strip() for ticker in ticker_list]

@asyncache.cached(TTLCache(1024, 3600))
def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d', interval='1m')
    return data['Close'].iloc[-1]

async def rank_companies(industry, analyses, prices):
    """
    The function `rank_companies` takes in industry, analyses, and prices as input, then prompts the
    user to rank companies in the industry based on investment potential and provide a rationale for the
    ranking.
    
    :param industry: The `industry` parameter in the `rank_companies` function represents the specific
    industry for which you are providing a ranking of companies based on their investment potential.
    This could be any sector or field such as technology, healthcare, finance, etc
    :param analyses: The `analyses` parameter in the `rank_companies` function likely contains a
    dictionary where the keys are stock tickers and the values are the corresponding analysis for each
    company in the industry. This analysis could include information such as financial performance,
    market trends, competitive positioning, and other relevant factors that
    :param prices: The `prices` parameter in the `rank_companies` function likely contains a dictionary
    where the keys are stock tickers and the values are the current prices of the corresponding stocks.
    This information is used to display the current price of each company in the industry analysis
    :return: The function `rank_companies` returns the response text generated by the `llm_gemini`
    function after providing the system prompt and messages related to ranking companies in a specific
    industry based on their investment potential.
    """
    system_prompt = f"You are a financial analyst providing a ranking of companies in the {industry} industry based on their investment potential. Be discerning and sharp. Truly think about whether a stock is valuable or not. You are a skeptical investor."

    analysis_text = "\n\n".join(
        f"Ticker: {ticker}\nCurrent Price: {prices.get(ticker, 'N/A')}\nAnalysis:\n{analysis}"
        for ticker, analysis in analyses.items()
    )

    messages = f"Industry: {industry}\n\nCompany Analyses:\n{analysis_text}\n\nBased on the provided analyses, please rank the companies from most attractive to least attractive for investment. Provide a professional rationale for your ranking. In each rationale, include the current price (if available) and a price target."
    response_text = await llm_gemini(system_prompt, messages)
    return response_text

def translate_cn(messages):
    """
    这个函数是一个用于将英文消息翻译成中文科普风格的专业翻译工具。
    
    :param messages: 这里是以上参数的含义：
    :return: 返回的是经过翻译后的中文段落，风格与中文科普读物相似，保留了原始 Markdown 格式。
    """
    system_prompt = "你是一位精通简体中文的专业翻译，尤其擅长将专业学术论文翻译成浅显易懂的科普文章。请你帮我将以下英文段落翻译成中文，风格与中文科普读物相似。\
        规则：- 翻译时要准确传达原文的事实和背景。\
            - 输入格式为 Markdown 格式，输出格式也必须保留原始 Markdown 格式"

    response_text = llm_gemini(system_prompt, messages)
    return response_text

async def analyze_stock(i, ticker, years):
    """
    The function `analyze_stock` retrieves and analyzes stock data for a given ticker symbol over a
    specified number of years, presenting the analysis in different tabs including analyst ratings,
    sentiment analysis, industry analysis, and a final summary.
    
    :param i: The `i` parameter in the `analyze_stock` function seems to be an index or counter variable
    used in a loop to keep track of the iteration number. It is likely used to display the analysis
    progress for each stock being analyzed
    :param ticker: The `analyze_stock` function you provided seems to be analyzing stock data for a
    given ticker symbol over a specified number of years. It fetches historical data, balance sheet
    information, financial data, and news related to the stock. It then displays this information in
    different tabs for analyst ratings, sentiment analysis
    :param years: The `years` parameter in the `analyze_stock` function represents the number of years
    of historical data you want to analyze for the given stock `ticker`. This parameter allows you to
    specify the time frame for which you want to retrieve and analyze the stock data, such as historical
    price data, balance sheet
    :return: The function `analyze_stock` is returning a dictionary with the keys 'ticker',
    'final_analysis', and 'current_price'. The value of 'ticker' is the stock ticker symbol,
    'final_analysis' is the final analysis result obtained from various data sources and analyses, and
    'current_price' is the current price of the stock.
    """
    final_analysis = ""
    current_price = 0            
    # try:
    if True:
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
                        analyst_ratings = await get_analyst_ratings(ticker)
                        st.write(analyst_ratings)
                    with tab2:
                        sentiment_analysis = await get_sentiment_analysis(ticker, news)
                        st.write(sentiment_analysis)
                    with tab3:
                        industry_analysis = await get_industry_analysis(ticker)
                        st.write(industry_analysis)
                    with tab4:
                        final_analysis = await get_final_analysis(ticker, {}, sentiment_analysis, analyst_ratings, industry_analysis)
                        st.write(final_analysis)
                        current_price = round(get_current_price(ticker), 2)
    # except Exception as e:
    #                 print(e)

    return {'ticker': ticker, 'final_analysis': final_analysis, 'current_price': current_price}


async def rank_byLLM(industry, tickers, years=1):
    """
    The `rank_byLLM` function asynchronously analyzes stock data for a given industry and ranks
    companies based on their analyses.
    
    :param industry: The `rank_byLLM` function you provided is an asynchronous function that ranks
    companies in a given industry based on their stock analyses. It takes three parameters:
    :param tickers: Tickers is a list of stock symbols representing the companies in the industry you
    want to analyze. For example, if you are analyzing the technology industry, tickers could include
    symbols like 'AAPL' for Apple Inc., 'MSFT' for Microsoft Corporation, and so on
    :param years: The `years` parameter in the `rank_byLLM` function specifies the number of years of
    historical data to consider when analyzing the stocks. This parameter allows you to customize the
    time frame for the analysis, which can impact the accuracy and relevance of the results. By
    adjusting the `years` parameter, defaults to 1 (optional)
    """
    # Perform analysis for each company
    analyses = {}
    prices = {}
    with st.status("Analyzing the Stocks...", expanded=True) as status:
            async_functions = []
            for i, ticker in enumerate(tickers):
                async_functions.append(analyze_stock(i, ticker, years))
            results = await asyncio.gather(*async_functions)

            status.update(label="Analyses completed!", state="complete", expanded=False)

    for result in results:
            analyses[result['ticker']] = result['final_analysis']
            prices[result['ticker']] = result['current_price']
             

    # Rank the companies based on their analyses
    st.subheader("Rank the companies based on their analyses")
    ranking = await rank_companies(industry, analyses, prices)
    print(f"\nRanking of Companies in the {industry} Industry:")
    print(ranking)
    st.markdown(ranking)

async def main(industry, years):
    """
    The `main` function in the Python code asynchronously generates ticker ideas for a given industry,
    displays them, and then ranks them based on a certain criteria.
    
    :param industry: The `industry` parameter in the `main` function represents the specific industry
    for which you want to generate ticker ideas and rank them. It is a string that specifies the
    industry you are interested in. For example, it could be "Technology", "Healthcare", "Finance", etc
    :param years: It seems like the `years` parameter is not being used in the `main` function you
    provided. If you need assistance with how to incorporate the `years` parameter into your code or
    have any other questions, feel free to ask!
    """
    # Generate ticker ideas for the industry
    tickers = await generate_ticker_ideas(industry)
    st.write(f"\n 5 Ticker Ideas for {industry} Industry:     "+ ", ".join(tickers))
    await rank_byLLM(industry, tickers)
     

if __name__ == "__main__":
    st.title("GENAI in Financial analysis")
    industry = st.text_input("Input Industry")

    if st.button("Analyze") and industry:
        years = 1 # int(input("Enter the number of years for analysis: "))

        asyncio.run(main(industry, years))

       

