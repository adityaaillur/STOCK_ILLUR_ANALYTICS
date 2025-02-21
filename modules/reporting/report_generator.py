import pandas as pd
from utils.logger import setup_logger

logger = setup_logger()

def generate_report(data, market_sentiment, output_format="markdown"):
    report = f"# Daily Pre-Market Stock Analysis Report\n\n"
    report += f"**Market Sentiment:** {market_sentiment}\n\n"
    report += "## Top Stocks:\n\n"
    report += data.to_markdown(index=False)
    
    resources = """
## Resources

- **Morningstar:** [https://www.morningstar.com](https://www.morningstar.com)
- **MarketWatch:** [https://www.marketwatch.com](https://www.marketwatch.com)
- **Seeking Alpha:** [https://seekingalpha.com](https://seekingalpha.com)
- **Finviz:** [https://finviz.com](https://finviz.com)
- **Yahoo Finance:** [https://finance.yahoo.com](https://finance.yahoo.com)
"""
    report += "\n" + resources
    logger.info("Report generated with US market resources.")
    return report

def save_report(report, filename):
    try:
        with open(filename, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {filename}.")
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        raise
