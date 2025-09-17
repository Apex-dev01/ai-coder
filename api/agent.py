import os
import subprocess
import re
import requests
from flask import Flask, request, jsonify
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_huggingface import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from github import Github
from datetime import datetime, timedelta

# =========================================================================
# Configuration and Environment Setup
# =========================================================================

app = Flask(__name__)

# Use a specific, free model from the Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # A good, general-purpose model
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Set a much lower token limit for Hugging Face's free tier
API_TOKENS_USED = 0
MAX_API_TOKENS = 50000  # A safe, conservative limit for the free tier

login_attempts = {}
COOLDOWN_PERIOD_MINUTES = 5
MAX_ATTEMPTS = 5

# =========================================================================
# The Agent's Tools
# =========================================================================

def check_api_limit():
    global API_TOKENS_USED
    if API_TOKENS_USED >= MAX_API_TOKENS:
        return True
    return False

@tool
def create_github_repo_and_clone(repo_name: str, description: str = None, private: bool = True):
    """
    Creates a new GitHub repository and clones it to the server's file system.
    Args:
        repo_name (str): The name for the new repository.
        description (str, optional): A description for the repository. Defaults to None.
        private (bool, optional): Whether the repository should be private. Defaults to True.
    Returns:
        str: The path to the new repository on the file system or an error message.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        return "Error: GitHub token not found. Please set the GITHUB_TOKEN environment variable."
    
    try:
        g = Github(github_token)
        user = g.get_user()
        new_repo = user.create_repo(
            repo_name,
            description=description,
            private=private,
            auto_init=True
        )
        repo_path = os.path.join(os.getcwd(), repo_name)
        subprocess.run(["git", "clone", new_repo.clone_url, repo_path], check=True)
        return f"Repository created and cloned to: {repo_path}"
    except Exception as e:
        return f"Error creating and cloning repository: {e}"

@tool
def generate_and_write_code(repo_path: str, goal: str, language_and_stack: str):
    """
    Generates and writes full-stack code into a cloned GitHub repository.
    Args:
        repo_path (str): The local file path to the cloned repository.
        goal (str): The high-level goal for the project.
        language_and_stack (str): The programming language and framework to use.
    Returns:
        str: A message confirming the files were written or an error.
    """
    global API_TOKENS_USED
    
    if check_api_limit():
        return "Warning: API usage limit reached. Stopping project generation to prevent charges."

    try:
        code_prompt = f"""
        You are an expert software developer.
        Generate all the necessary full-stack code for a complete project with the goal: '{goal}'.
        The project should be built using the following technology stack: {language_and_stack}.

        Provide the full content for the main files, including a brief description of the project structure and how to run it.
        Use a clear format for each file, such as:
        
        ### index.html
        ```html
        ```
        
        ### backend/server.js
        ```javascript
        // JavaScript code here
        ```
        
        ### requirements.txt
        ```
        # Dependencies here
        ```
        
        Ensure all files are complete and runnable.
        """
        
        generated_content = llm.invoke(code_prompt).content
        API_TOKENS_USED += len(generated_content.split())
        
        file_regex = r"###\s+([\w\d\./]+\.[\w\d]+)\s+```[\w\d]*\n(.*?)```"
        matches = re.findall(file_regex, generated_content, re.DOTALL)
        
        if not matches:
            return "Error: Could not parse generated code from the model."

        for filename, code_block in matches:
            file_path = os.path.join(repo_path, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(code_block.strip())
        
        return "Frontend and backend code have been generated and written to the repository."

    except Exception as e:
        return f"An error occurred while generating and writing files: {e}"

@tool
def commit_and_push(repo_path: str, message: str):
    """
    Commits changes and pushes them to the remote GitHub repository.
    Args:
        repo_path (str): The local file path to the cloned repository.
        message (str): The commit message.
    Returns:
        str: A message confirming the push or an error.
    """
    try:
        os.chdir(repo_path)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        return "Changes committed and pushed successfully."
    except Exception as e:
        return f"Error committing and pushing changes: {e}"

@tool
def manage_full_stack_project(project_name: str, goal: str):
    """
    Manages a full-stack project from start to finish.
    1. Creates a GitHub repository and clones it.
    2. Determines the best programming language and stack for the goal.
    3. Generates and writes full-stack code in that language.
    4. Commits and pushes the code to GitHub.
    Args:
        project_name (str): The name for the project.
        goal (str): The high-level goal for the project.
    Returns:
        str: A message confirming the project was created or an error.
    """
    global API_TOKENS_USED

    if check_api_limit():
        return "Warning: API usage limit reached. Cannot start new projects."

    try:
        repo_path = create_github_repo_and_clone(project_name, description=f"An AI-generated project with the goal: {goal}")
        
        if "Error" in repo_path:
            return repo_path

        language_prompt = f"Given the project goal: '{goal}', what is the single best programming language and framework or API for a full-stack web application? Be extremely concise and provide only the name, e.g., 'Node.js with Express.js', 'Python with Flask', or 'Web Audio API with vanilla JavaScript'."
        
        language_and_stack_response = llm.invoke(language_prompt)
        API_TOKENS_USED += len(language_and_stack_response.content.split()) + len(language_prompt.split())
        language_and_stack = language_and_stack_response.content.strip()
        
        write_status = generate_and_write_code(repo_path, goal, language_and_stack)
        if "Error" in write_status:
            return write_status

        commit_status = commit_and_push(repo_path, message=f"Initial commit: AI-generated project in {language_and_stack}")
        
        if "Error" in commit_status:
            return commit_status
        
        return f"Full-stack project '{project_name}' successfully created in {language_and_stack}, with code generated and pushed to GitHub. The repository is ready for deployment."

    except Exception as e:
        return f"An error occurred while managing the full-stack project: {e}"

# =========================================================================
# Serverless Function Endpoints
# =========================================================================

@app.route("/api/login", methods=["POST"])
def login_endpoint():
    try:
        data = request.json
        password_attempt = data.get("password")
        
        ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
        
        if ip_address in login_attempts:
            cooldown_end_time = login_attempts[ip_address].get("cooldown_until")
            if cooldown_end_time and datetime.now() < cooldown_end_time:
                return jsonify({"success": False, "message": "Too many failed attempts. Try again later."}), 429
            
        if password_attempt == os.getenv("WEBSITE_PASSWORD"):
            if ip_address in login_attempts:
                del login_attempts[ip_address]
            return jsonify({"success": True}), 200
        else:
            login_attempts.setdefault(ip_address, {"count": 0, "cooldown_until": None})
            login_attempts[ip_address]["count"] += 1
            if login_attempts[ip_address]["count"] >= MAX_ATTEMPTS:
                login_attempts[ip_address]["cooldown_until"] = datetime.now() + timedelta(minutes=COOLDOWN_PERIOD_MINUTES)
            return jsonify({"success": False, "message": "Incorrect password"}), 401

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/agent", methods=["POST"])
def agent_endpoint():
    try:
        data = request.json
        prompt = data.get("prompt")
        mode = data.get("mode")

        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        if mode == "agent":
            if check_api_limit():
                return jsonify({"response": "I'm sorry, I have reached my API usage limit. I cannot create a new project at this time. Please try again later."})
                
            tools = [manage_full_stack_project]
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert autonomous assistant. Your only tool is to manage a full-stack project. The user will provide a project name and a high-level goal. Your job is to use your tool to achieve that goal. Analyze the user's project goal and infer the best programming language, framework, and technologies to use. Fill in any missing details or make smart assumptions to create a complete, functioning project. Your ultimate goal is to deliver a fully functional project ready for hosting. If the user's goal is to create a web-based app for creating music, use the Web Audio API with vanilla JavaScript."),
                ("user", "{prompt}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            agent = create_tool_calling_agent(llm, tools, agent_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            response = agent_executor.invoke({"input": prompt})
            return jsonify({"response": response.get("output")})

        elif mode == "chat":
            chat_response = llm.invoke(prompt)
            return jsonify({"response": chat_response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)