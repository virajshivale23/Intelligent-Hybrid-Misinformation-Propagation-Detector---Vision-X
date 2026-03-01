import requests
import streamlit as st

API_KEY = "789601692407496dbc647ddb76d76aa4"

@st.cache_data(ttl=300)
def fetch_news(keyword):
    url = f"https://newsapi.org/v2/everything?q={keyword}&language=en&pageSize=5&apiKey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    articles = []

    if data["status"] == "ok":
        for article in data["articles"]:
            articles.append({
                "title": article["title"],
                "description": article["description"],
                "source": article["source"]["name"],
                "published_at": article["publishedAt"]
            })

    return articles