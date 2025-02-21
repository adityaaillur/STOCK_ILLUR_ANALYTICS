import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from loguru import logger
from textblob import TextBlob
import os
from datetime import datetime

class NewsCollector:
    """Collects and analyzes financial news"""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.sources = [
            'bloomberg.com',
            'reuters.com',
            'cnbc.com',
            'wsj.com'
        ]
    
    async def get_market_news(self) -> List[Dict]:
        """Fetch latest market news and analyze sentiment"""
        try:
            url = f"https://newsapi.org/v2/everything"
            news = []
            
            for source in self.sources:
                params = {
                    'apiKey': self.news_api_key,
                    'domains': source,
                    'q': 'stock market OR economy OR federal reserve',
                    'language': 'en',
                    'sortBy': 'publishedAt'
                }
                
                response = requests.get(url, params=params)
                articles = response.json().get('articles', [])
                
                for article in articles:
                    sentiment = TextBlob(article['title']).sentiment.polarity
                    news.append({
                        'title': article['title'],
                        'source': source,
                        'url': article['url'],
                        'published_at': article['publishedAt'],
                        'sentiment': sentiment
                    })
            
            return news
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []
    
    def analyze_market_sentiment(self, news: List[Dict]) -> str:
        """Analyze overall market sentiment from news"""
        if not news:
            return "NEUTRAL"
            
        sentiments = [article['sentiment'] for article in news]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        if avg_sentiment > 0.2:
            return "BULLISH"
        elif avg_sentiment < -0.2:
            return "BEARISH"
        else:
            return "NEUTRAL" 