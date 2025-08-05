# tools.py

import pandas as pd
import duckdb
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from langchain_core.tools import tool

@tool
def analyze_highest_grossing_films(url: str, questions: list[str]) -> list:
    """
    Performs a full analysis of the highest-grossing films from a given Wikipedia URL.
    It scrapes the data, cleans it, and then answers a list of questions based on that data.
    It can answer factual questions and generate a scatterplot with a regression line.
    The final output is a list containing the answers in the order of the questions, 
    with the plot's base64 data URI as the last element if requested.
    """
    # === Part 1: Scrape and Clean the Data (from the old scrape tool) ===
    try:
        tables = pd.read_html(url)
        df = tables[0]
        # Data Cleaning
        df['Worldwide gross'] = df['Worldwide gross'].astype(str).str.replace(r'\[\d+\]', '', regex=True).str.replace('$', '').str.replace(',', '').astype(float)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
        df.dropna(subset=['Year', 'Rank', 'Peak', 'Worldwide gross'], inplace=True)
    except Exception as e:
        return [f"Failed to scrape or clean data from {url}: {e}"]

    # === Part 2: Answer Questions and Plot (from the old analysis/plot tools) ===
    answers = []
    for q in questions:
        q_lower = q.lower()
        try:
            if "how many $2 bn movies" in q_lower:
                # This question is ambiguous in the prompt. Assuming it means movies > $2bn.
                answer = df[df['Worldwide gross'] > 2_000_000_000].shape[0]
                answers.append(answer)
            elif "earliest film that grossed over $1.5 bn" in q_lower:
                filtered_df = df[df['Worldwide gross'] > 1_500_000_000]
                answer = filtered_df.sort_values('Year').iloc[0]['Title']
                answers.append(answer)
            elif "correlation between rank and peak" in q_lower:
                answer = df['Rank'].corr(df['Peak'])
                answers.append(answer)
            elif "draw a scatterplot" in q_lower:
                # Plotting logic
                x, y = df['Rank'], df['Peak']
                fig, ax = plt.subplots()
                ax.scatter(x, y, alpha=0.5)
                ax.set_xlabel('Rank')
                ax.set_ylabel('Peak')
                ax.set_title('Scatterplot of Peak vs. Rank')
                ax.grid(True)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line = slope * x + intercept
                ax.plot(x, line, 'r--', label=f'y={slope:.2f}x+{intercept:.2f}')
                ax.legend()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                data_uri = f"data:image/png;base64,{img_base64}"
                if len(data_uri) > 100000:
                    answers.append("Error: Generated image is too large.")
                else:
                    answers.append(data_uri)
        except Exception as e:
            answers.append(f"Error processing question '{q}': {e}")
            
    return answers


# You can keep this tool for the second example question. It's already valid.
@tool
def query_indian_high_court_data(query: str) -> pd.DataFrame:
    """
    Executes a SQL query against the Indian High Court Judgments dataset stored in S3.
    The query should be a valid DuckDB SQL query.
    Returns the result as a pandas DataFrame.
    """
    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute("INSTALL httpfs; LOAD httpfs;")
        result_df = con.execute(query).fetchdf()
        return result_df
    except Exception as e:
        return f"Error executing DuckDB query: {e}"