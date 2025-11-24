# pip install -qU "langchain[anthropic]" to call the model
import json
import os
from dotenv import load_dotenv
import requests
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
weather_api = os.getenv("QA_AGENT_WEATHER_API_KEY")

@tool("get_weather", description="Get the weather for a given city", return_direct=False)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    BASE_URL = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": weather_api,
        "q": city,
    }
    resp = requests.get(BASE_URL, params=params)
    return resp.json()

llm = ChatOllama(
    model="gpt-oss:20b",
    validate_model_on_init=True,
    temperature=0.3,
    base_url="http://localhost:11434"
 )#.bind_tools([get_weather])

# result = llm.invoke("What is the weather in London?")
# # 检查是否有工具调用
# if isinstance(result, AIMessage) and result.tool_calls:
#     print(result.tool_calls)
# if result.tool_calls:
#     call = result.tool_calls[0]
#     name = call["name"]
#     args = call["args"]

#     # Step 2: run the tool
#     result = get_weather.run(args)
#     print(result)
#     # Step 3: send response back to LLM
#     final = llm.invoke([
#         {"role": "user", "content": "What is the weather in London?"},
#         {
#             "role": "tool",
#             "content": json.dumps(result),   # 工具结果一定要转成 string
#             "tool_call_id": call["id"]
#         }
#     ])

#     print("Final answer:", final.content)
    


agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant"
)

# Run the agent

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in wuxi, is is suitable for outing?"}]}
)
print(response["messages"][-1].content)