# tools.py - Stock Analysis Tools
from langchain_core.tools import tool

# Stock analysis agent tools
@tool
def analyze_stock(ticker: str):
    """Analyze a specific stock ticker"""
    return f"Successfully analyzed stock {ticker}. Current analysis shows moderate growth potential with standard volatility."

@tool
def get_market_trends():
    """Get current market trends and indicators"""
    return "Market trends analysis: Overall market showing bullish sentiment with technology sector leading gains. S&P 500 up 2.1% this month."

@tool
def calculate_portfolio_risk(stocks: str):
    """Calculate portfolio risk for given stocks"""
    return f"Portfolio risk analysis for {stocks}: Moderate risk level detected. Diversification score: 7/10. Recommended for balanced investors."

@tool
def recommend_portfolio_allocation(risk_tolerance: str):
    """Recommend portfolio allocation based on risk tolerance"""
    return f"Portfolio allocation recommendation for {risk_tolerance} risk tolerance: 60% equities, 30% bonds, 10% alternatives. Expected annual return: 8-12%."