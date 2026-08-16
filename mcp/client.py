from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http"
            }
        },
    )
    tools = await client.get_tools()
    model = ChatGroq(model="llama-3.3-70b-versatile")
    agent = create_react_agent(model, tools)
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user",
                       "content": "First use the add tool to calculate 3 + 5. After you get the result, use the multiply tool to multiply that result by 123."
                       }]}
    )
    weather_response = await agent.ainvoke({"messages": [{"role": "user",
                                                         "content": "Use the get_weather tool to get the weather for California. Do not answer using your own knowledge."
                                                         }]})

    print("Math response: ", math_response["messages"][-1].content)
    print("Weather response : ", weather_response["messages"][-1].content)


asyncio.run(main())
