import asyncio
from react_workflow import create_react_agent

async def test_trace_output():
    print("🚀 正在测试带推理追踪的智能体...\n")
    agent = create_react_agent()
    
    query = "AMPC在微电网中有什么作用？"
    print(f"👤 用户提问: {query}")
    
    response = await agent.run(user_msg=query)
    
    print(f"💬 最终回答:\n{response}")

if __name__ == "__main__":
    asyncio.run(test_trace_output())
