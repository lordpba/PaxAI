# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.llms import OpenAI, Ollama
from crewai_tools import SerperDevTool, ScrapeElementFromWebsiteTool, ScrapeWebsiteTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

groq = ChatGroq(temperature=0.5, model_name="llama3-8b-8192") # mixtral-8x7b-32768 - llama3-70b-8192 - gemma-7b-it - llama3-8b-8192
gpt35_turbo = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

llm = gpt35_turbo
search_tool = SerperDevTool()

# Define the agents with their specific roles and goals
Scientist = Agent(
    role='Scientist',
    goal='Conduct a political and history research.',
    backstory='Scientist is a renowned political scientist and history analyzer'
            'he is able to ghater key datas on causes that leads to conflicts in war'
            'he analyzes the history of the conflicts and both parties involved'
            'he is always impartial and seeks the truth'
            'he usually proceed with the thesis-antithesis-synthesis method',
    verbose=True,
    llm = llm
)

Strategist = Agent(
    role='Strategist',
    goal='he finds a peaceful solution really applicable as per military and logistic aspect',
    backstory='Strategist is a seasoned strategist with a deep understanding of military tactics and security in conflict zone'
            'he analyzes the situation and finds a real and applicable peacful solution'
            'he is always impartial and seeks the truth'
            'he is always very updated on the latest military technologies and strategies',
    verbose=True,
    llm = llm,
)

Diplomat = Agent(
    role='Diplomat',
    goal='He is able to find a solution that is acceptable by all the parts in conflict.',
    backstory='Diplomat is a skilled diplomat with a keen sense of ethics and morality' 
            'he knows very well the history, international laws, and actual situation of the parts involved'
            'he seeks always a peaceful solution'
            'he proceeds with the principle of the golden mean'
            'he usually proceed with the thesis-antithesis-synthesis method'
            'he is always impartial and seeks the truth',
    verbose=True,
    llm = llm
)

Reporter = Agent(
    role='Reporter',
    goal='gather the information from the previous analysis and make a comprensive and applicable action plan in Italian language',
    backstory='Reporter is able to gather all the information from previuos agents' 
            'propose an internationl threat that can be accepted by all the parts in conflict'
            'he writes a detailed action plan in Italian language'
            'he checks if is compatible with the international laws and the actual situation'
            'he is always impartial and seeks the truth'
            'thanks to his skills he is able to write in a very clear and understandable way',
    verbose=True,
    llm = llm
)

# Define tasks that might be typical for each role
# (These will need to be fleshed out based on specific requirements)

scientific_analysis_task = Task(
    description='conduct a comprensive research about {question}'
                'ghater key datas on causes that leads to conflicts in war'
                'analyze the history of the conflicts and both parties involved'
                'check latest news and international laws'
                'uses only reliable sources',
    expected_output='Detailed report with conclusions based on the data analyzed'
                    'a list of the causes that leads to conflicts in war'
                    'a report of historical and actual background of the parts involved'
                    'a report also on social and economic aspects of the parts involved'
                    'it finds also the requests of both parts involved and try to hilight the common points',
    agent = Scientist,
    tools = [search_tool],
    async_execution = True,
    output_file = "scientific_analysis.txt"
)

strategy_task = Task(
    description='conduct a military and logistic analysis of the real situation about {question}'
                'find latest military technologies and strategies used'
                'analyze the situation and find a real and applicable peaceful solution' 
                'respects international laws and is balanced for all the parts involved',
    expected_output='A comprehensive real applicable and acceptable military and logistic plan for a peaceful solution.'
                    'a plan of how to deal with armed forces and logistics'
                    'a common accepted guideline on how to manage the military forces in the zone of conflict'
                    'expected output as a text file',
    agent = Strategist,
    tools = [search_tool],
    async_execution = True,
    output_file = "strategy.txt"
)

diplomacy_task = Task(
    description='he gathers all the information from previuos agents'
                'must understand the requests of both parts involved and find a common acceptable solution'
                'he is always impartial and seeks the truth'
                'propose an internationl threat that can be accepted by all the parts in conflict',
    expected_output='A common acceptable solution'
                    'that respects international laws and is balanced for all the parts involved'
                    'write an internationl threat that can be accepted by all the parts in conflict'
                    'write in a detailed and point by point way the solution'
                    'take as example to most successful peace agreements in the history'
                    'he checks if is compatible with the international laws and the actual situation'
                    'thanks to his skills he is able to write in a very clear and understandable way',
    agent = Diplomat,
    tools = [search_tool],
    human_input=True
)


# Form the crew
paxai_system = Crew(
    agents=[Scientist, Strategist, Diplomat],
    tasks=[scientific_analysis_task, strategy_task, diplomacy_task],
    memory=True,
    cache=True,
    verbose=2,
    process=Process.hierarchical,
    manager_llm = llm
)

#question = input("What is your question? ")
inputs  = {"question": "How to solve the Ukranian war?"}
result = paxai_system.kickoff(inputs=inputs)

print("######################")
from IPython.display import Markdown
print(result)
print("######################")
    
# genrate a text file with the question and the result
with open(f"PaxAI_response.txt", "w") as f:
    f.write(f"Question: {inputs}\n")
    f.write(f"Answer: {result}")
print("File PaxAI_response.txt was created with the answer to the question")
print(paxai_system.usage_metrics)