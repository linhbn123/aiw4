import os
import subprocess
import git
from github import Github
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.tools.shell.tool import ShellTool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langsmith import traceable

# Initialize local repository path
LOCAL_REPO_PATH = "/tmp/aiw4"

@tool
@traceable
def clone_repo():
    """
    Clone the repository to the local repo path if not already cloned.
    """
    repo_url = f"https://{os.getenv('GITHUB_TOKEN')}@github.com/{os.getenv('REPO_PATH')}.git"

    if not os.path.exists(LOCAL_REPO_PATH):
        git.Repo.clone_from(repo_url, LOCAL_REPO_PATH)
        print(f"Cloned repo from {repo_url} to {LOCAL_REPO_PATH}")
    else:
        print(f"Repo already exists at {LOCAL_REPO_PATH}")

@tool
@traceable
def switch_to_local_repo_path():
    """
    Switch to the local repo path.
    """
    try:
        # Change the current working directory to the specified directory
        os.chdir(LOCAL_REPO_PATH)
        print(f"Switched to directory: {LOCAL_REPO_PATH}")
    except Exception as e:
        print(f"Failed to switch to directory: {e}")

@tool
@traceable
def checkout_source_branch():
    """
    Checkout the main branch, pull latest changes, and checkout the specific branch.
    """
    # Initialize the repository object
    repo = git.Repo(os.getcwd())

    # 1. Checkout the main branch
    repo.git.checkout('main')

    # 2. Pull the latest changes
    repo.git.pull()

    # 3. Checkout the specific branch
    repo.git.checkout(os.getenv('SOURCE_BRANCH'))
    print(f"Active branch: {repo.active_branch.name}")

@tool
@traceable
def get_files_from_pull_request():
    """
    Retrieve the list of files changed in a pull request.

    Returns:
        list: A list of filenames that have been changed in the pull request.
    """
    # Initialize GitHub API with token
    g = Github(os.getenv('GITHUB_TOKEN'))

    # Get the repo path and PR number from the environment variables
    repo_path = os.getenv('REPO_PATH')
    
    # Get the repo object
    repo = g.get_repo(repo_path)

    # Fetch pull request by number
    pull_request_number = int(os.getenv('PR_NUMBER'))
    pull_request = repo.get_pull(pull_request_number)

    # Get the diffs of the pull request
    return [file.filename for file in pull_request.get_files()]

@tool
@traceable
def run_autopep8(files):
    """
    Run autopep8 on a list of files.

    Args:
        files (list): A list of file paths to be formatted.
    """
    for file in files:
        subprocess.run(['autopep8', '--in-place', file])

@tool
@traceable
def has_changes():
    """
    Check if the repository has any changes.

    Returns:
        bool: True if the repository has changes, False otherwise.
    """
    repo = git.Repo(os.getcwd())
    print(f"Repo has changes: {repo.is_dirty()}")
    return repo.is_dirty()

@tool
@traceable
def commit_and_push(commit_message):
    """
    Commit the changes and push to the remote branch.

    Args:
        commit_message (str): The commit message.
    """

    branch_name = os.getenv('SOURCE_BRANCH')
    repo = git.Repo(os.getcwd())
    print(f"Active branch: {repo.active_branch.name}")
    repo.git.add(update=True)
    repo.index.commit(commit_message)
    origin = repo.remote(name='origin')
    print(f"Pushing changes to remote branch '{branch_name}'")
    push_result = origin.push(refspec=f'{branch_name}:{branch_name}')
    print("Push result:")
    for push_info in push_result:
        print(f"  - Summary: {push_info.summary}")
        print(f"    Remote ref: {push_info.remote_ref}")
        print(f"    Local ref: {push_info.local_ref}")
        print(f"    Flags: {push_info.flags}")

# List of tools to use
tools = [
    ShellTool(ask_human_input=True),
    clone_repo,
    switch_to_local_repo_path,
    checkout_source_branch,
    get_files_from_pull_request,
    run_autopep8,
    has_changes,
    commit_and_push
]

# Configure the language model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Python expert. 
            You always write code following code convention and best practices in PEP8. 
            Your audience are experienced engineers and managers.
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

user_prompt = """
Clone the repository to a local directory.
Then in that local directory, 
checkout the source branch of the pull request,
beautify all the code in the .py source code files 
changed by the pull request as per pep8 standards, 
commit the change and push to the remote repository.
"""
list(agent_executor.stream({"input": user_prompt}))
