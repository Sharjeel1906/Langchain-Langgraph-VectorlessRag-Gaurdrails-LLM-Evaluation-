from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async  def get_weather(location:str)->str:
    """Get the current weather for a given location."""
    return "It is always raining in california!"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")