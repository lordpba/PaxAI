from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.llms import OpenAI, Ollama
from crewai_tools import SerperDevTool, ScrapeElementFromWebsiteTool, ScrapeWebsiteTool
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

groq = ChatGroq(temperature=0.3, model_name="llama3-8b-8192") # mixtral-8x7b-32768 - llama3-70b-8192 - gemma-7b-it - llama3-8b-8192
gpt35_turbo = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
llm = gpt35_turbo

search_tool = SerperDevTool()
#question = input("What is your question? ")
question  = "How to solve the Israel - Palestin war?"

# Define the agents with their specific roles and goals
Scientist = Agent(
    role='Scientist',
    goal='Conduct technical analysis and provide logical conclusions.',
    backstory='Scientist is a renowned scientist with expertise in data analysis and interpretation in war zones\
        he analyzes the history of the conflicts and both parties involved.',
    verbose=True,
    allow_delegation=False,
    llm = llm,
    tools = [search_tool]
)

Strategist = Agent(
    role='Strategist',
    goal='Develop defense strategies and oversee tactical operations.',
    backstory='Strategist is a seasoned strategist with a deep understanding of military tactics and security in conflict zone\
        he analyzes the situation and finds a peacful solution.',
    verbose=True,
    allow_delegation=False,
    llm = llm,
    tools = [search_tool],
)

Diplomat = Agent(
    role='Diplomat',
    goal='Evaluate ethical implications and make balanced decisions.',
    backstory='Diplomat is a skilled diplomat with a keen sense of ethics and morality, he knows very well the history of the parts involved\
        he seeks always a peaceful solution.',
    verbose=True,
    allow_delegation=False,
    tools = [search_tool],
    llm = llm
)

Reporter = Agent(
    role='Reporter',
    goal='gather the information from the previous analysis and make an action plan in Italian language',
    backstory='Reporter is able to gather all the information from previuos agents and propose an internationl threat that can be accepted by all the parts in conflict',
    verbose=True,
    allow_delegation=False,
    llm = llm
)

# Define tasks that might be typical for each role
# (These will need to be fleshed out based on specific requirements)

scientific_analysis_task = Task(
    description=f'Analyze the technical data provided and extract conclusions about {question}.',
    expected_output='Detailed report with conclusions based on the data analysis.',
    agent = Scientist,
    
)

strategy_task = Task(
    description='Formulate a strategic plan based on current threats.',
    expected_output='A comprehensive peace strategy.',
    agent = Strategist
)

diplomacy_task = Task(
    description='Assess the ethical implications of a proposed action.',
    expected_output='A reasoned judgment on the ethical acceptability of the action.',
    agent = Diplomat,
    human_input=True
)

action_plan_task = Task(
    description='Gather the information from the previous analysis and make an action plan in Italian',
    expected_output='A detailed action plan based on the analysis and conclusions all written in Italian.',
    agent = Reporter
)
# Form the crew
paxai_system = Crew(
    agents=[Scientist, Strategist, Diplomat, Reporter],
    tasks=[scientific_analysis_task, strategy_task, diplomacy_task, action_plan_task],
    memory=True,
    cache=True,
    verbose=1,
    process=Process.sequential  # assuming a consensus process is needed for decision
)

result = paxai_system.kickoff()
print("######################")
print(result)
# genrate a text file with the question and the result
with open(f"PaxAI_response.txt", "w") as f:
    f.write(f"Question: {question}\n")
    f.write(f"Answer: {result}")

print(paxai_system.usage_metrics)