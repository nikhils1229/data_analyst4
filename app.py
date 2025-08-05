# app.py

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate  # <-- THIS LINE WAS ADDED
from tools import (
    analyze_highest_grossing_films, 
    query_indian_high_court_data
)

# Load environment variables
load_dotenv()

# 1. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Define the tools
tools = [
    analyze_highest_grossing_films,
    query_indian_high_court_data
]

# 3. Create the Agent Prompt
prompt_template = """
You are a powerful data analyst agent. Your goal is to answer the user's request by selecting the single best tool for the job.

1.  Examine the user's request.
2.  If the request is about analyzing Wikipedia film data, use the `analyze_highest_grossing_films` tool. You will need to extract the URL and the specific questions from the user's prompt to use as arguments for the tool.
3.  If the request is about querying the Indian High Court dataset, use the `query_indian_high_court_data` tool with the appropriate SQL query.
4.  Return the final output from the tool directly.

Begin!
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. Create the Agent
agent = create_openai_tools_agent(llm, tools, prompt)

# 5. Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Flask API ---
app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def handle_data_analysis_task():
    """
    API endpoint to handle data analysis tasks.
    It expects a file 'question.txt' in the POST request body.
    """
    if 'question.txt' not in request.files:
        return jsonify({"error": "Missing 'question.txt' file in the request"}), 400

    question_file = request.files['question.txt']
    task_description = question_file.read().decode('utf-8')

    if not task_description:
        return jsonify({"error": "The 'question.txt' file is empty"}), 400

    try:
        # Invoke the agent with the task description
        result = agent_executor.invoke({
            "input": task_description
        })
        
        # The agent's final output from the tool is often a list or dict.
        # jsonify will correctly handle converting it to a JSON response.
        return jsonify(result['output'])
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# This block allows you to test with `python app.py`
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)