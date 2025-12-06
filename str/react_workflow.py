import os
import sys
import asyncio
from dotenv import load_dotenv

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer

# 导入 Instrumentation
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.core.instrumentation.events.base import BaseEvent

# 导入上一部分构建的智能检索工具
try:
    from search_tool import search_tool
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from search_tool import search_tool

# 加载环境变量
load_dotenv()

# 定义事件处理器
class ReActTraceHandler(BaseEventHandler):
    step_count: int = 0

    @classmethod
    def class_name(cls) -> str:
        return "ReActTraceHandler"

    def handle(self, event: BaseEvent, **kwargs):
        if isinstance(event, AgentToolCallEvent):
            self.step_count += 1
            print(f"\n[Step {self.step_count}]")
            print(f"⚡ Action    (行动): 调用工具 {event.tool.name}")
            print(f"   参数: {event.arguments}")
        elif isinstance(event, LLMChatEndEvent):
            # 捕获 LLM 的输出，通常包含 Reasoning
            response = event.response
            content = response.message.content
            if content:
                # 简单的过滤，避免重复打印最终答案
                if "Final Answer" not in content:
                    print(f"\n🤔 Reasoning (思考): \n{content}")

# 注册事件处理器
trace_handler = ReActTraceHandler()
dispatcher = instrument.get_dispatcher(__name__)
dispatcher.add_event_handler(trace_handler)
# 也要注册到根 dispatcher 以确保捕获
root_dispatcher = instrument.get_dispatcher("llama_index")
root_dispatcher.add_event_handler(trace_handler)

def build_llm() -> LlamaOpenAI:
    base_url = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")
    api_key = os.environ.get("YUNWU_API_KEY")
    model = os.environ.get("YUNWU_MODEL", "gpt-4o-mini")
    
    if not api_key:
        raise ValueError("Missing YUNWU_API_KEY")
        
    return LlamaOpenAI(
        model=model,
        api_key=api_key,
        api_base=base_url,
        temperature=0.1,
    )

def create_react_agent():
    llm = build_llm()
    Settings.llm = llm
    tools = [search_tool]
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    
    system_prompt = """
    你是一位专业的学术论文助手。
    
    IMPORTANT: 你必须展示你的思考过程 (ReAct Trace)。
    
    在回答之前，请先输出以下部分：
    
    ### Reasoning
    (在这里描述你为什么需要调用工具，或者如何分析问题)
    
    ### Action
    (如果你调用了工具，请在这里说明工具名称和目的)
    
    ### Observation
    (在这里总结工具返回的关键信息)
    
    ### Final Answer
    (这里才是给用户的最终回答)
    
    请务必严格遵守此格式。
    """
    
    agent = ReActAgent(
        tools=tools,
        llm=llm,
        verbose=True, 
        system_prompt=system_prompt,
        name="GraphRAGAgent",
        description="A ReAct agent.",
        memory=memory
    )
    
    return agent

def main():
    print("🤖 初始化 ReAct 智能体...")
    try:
        agent = create_react_agent()
        print("✅ 智能体就绪！(输入 'exit' 退出)")
        print("-" * 50)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    async def run_chat_loop():
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, input, "\n👤 用户: ")
                user_input = user_input.strip()
                if not user_input: continue
                if user_input.lower() in ["exit", "quit", "退出"]: break
                
                print("\n🤖 智能体正在思考...")
                # 重置步骤计数
                trace_handler.step_count = 0
                
                response = await agent.run(user_msg=user_input)
                
                print(f"\n💬 回答:\n{response}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")

    asyncio.run(run_chat_loop())

if __name__ == "__main__":
    main()
