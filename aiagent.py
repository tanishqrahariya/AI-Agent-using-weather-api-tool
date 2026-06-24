from dotenv import load_dotenv
import requests
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)

# Tool 1
search_tool = DuckDuckGoSearchRun()

# Tool 2
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather data for a given city.
    """
    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key=a63f8c9774768028cf9386cb19378c09&query={city}"
    )

    data = requests.get(url).json()

    if "current" not in data:
        return f"Weather data not found: {data}"

    return (
        f"Temperature: {data['current']['temperature']}°C, "
        f"Weather: {data['current']['weather_descriptions'][0]}"
    )

# Create Agent
agent = create_agent(
    model=model,
    tools=[search_tool, get_weather_data],
    system_prompt="""
    You are a helpful assistant.
    Use tools whenever needed to answer questions.
    """
)

# Invoke
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Find the capital of Madhya Pradesh, then find its current weather condition")
        ]
    }
)
print(response["messages"][-1].content)