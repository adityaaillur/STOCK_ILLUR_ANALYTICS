from flask import Flask, jsonify
import os
from config import Config
from modules import stock_preprocessing, fundamental_analysis, quant_models, risk_management, portfolio, reporting
from utils.logger import setup_logger

logger = setup_logger()
app = Flask(__name__)

# Ensure necessary directories exist
os.makedirs(Config.RAW_DATA_PATH, exist_ok=True)
os.makedirs(Config.REPORT_PATH, exist_ok=True)

@app.route("/")
def index():
    return "Ultimate Pre-Market Analytics App is running."

@app.route("/run_analysis", methods=["GET"])
def run_analysis():
    try:
        # PART 1: Data Collection & Pre-Processing
        sentiment, headlines = stock_preprocessing.get_market_sentiment()
        logger.info(f"Market sentiment: {sentiment}")
        
        # Define list of US market S&P 500 tickers (could be expanded)
        ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        clean_data = stock_preprocessing.generate_clean_stock_dataset(ticker_list)
        
        if clean_data.empty:
            return jsonify({"error": "No clean stock data available."}), 500
        
        # Save raw clean data (for debugging)
        clean_data.to_csv(os.path.join(Config.RAW_DATA_PATH, "clean_stock_data.csv"), index=False)
        
        # PART 2: Fundamental Analysis (e.g., screening, valuation)
        screened_data = fundamental_analysis.screen_stocks(clean_data)
        
        # PART 3: Advanced Quantitative Models (DCF, Monte Carlo, etc.)
        valuation_results = quant_models.run_valuation_models(screened_data)
        
        # PART 4: Risk Management & Portfolio Construction
        risk_adjusted_portfolio = portfolio.construct_portfolio(valuation_results)
        
        # PART 5: Generate Report
        report = reporting.generate_report(risk_adjusted_portfolio, market_sentiment=sentiment)
        report_file = os.path.join(Config.REPORT_PATH, "daily_report.md")
        reporting.save_report(report, report_file)
        
        return jsonify({"message": "Analysis complete", "report": report})
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
