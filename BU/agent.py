import asyncio
import os
from typing import Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from browser_use import Agent
from browser_use.browser import BrowserSession, BrowserProfile
from pydantic import Field

load_dotenv()


class PatchedChatOllama(ChatOllama):
    """
    Pydantic v2 fix:
    - 'provider' field required by browser-use to identify the LLM backend.
    - 'model_name' property added as an alias for 'model', which browser-use's
      cloud event dispatcher (cloud_events.py line 217) accesses directly.
    - __setattr__ override allows browser-use's token tracker to monkey-patch
      ainvoke/invoke without Pydantic raising 'object has no field' errors.
    """
    provider: str = Field(default="ollama")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def model_name(self) -> str:
        """Alias expected by browser-use's CreateAgentTaskEvent.from_agent()"""
        return self.model

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("ainvoke", "invoke", "stream", "astream"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


async def main():
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe'
        )
    )

    # Local Worker (Qwen 2.5 3B) - handles actual browser actions
    local_llm = PatchedChatOllama(
        model="qwen2.5:3b",
        num_ctx=16384
    )

    # Cloud Planner (Gemini Flash Lite) - handles high-level strategy
    planner_llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')

    initial_actions = [
        {'navigate': {'url': 'https://www.youtube.com/', 'new_tab': False}}
    ]

    agent = Agent(
        task=(
            "Investigate the latest 5 videos on youtube page "
            "and select any one title to play and tell which video is played."
        ),
        llm=local_llm,
        planner_llm=planner_llm,
        browser_session=browser_session,
        initial_actions=initial_actions,
        use_vision=False,
        save_conversation_path='logs/conversation.json',
        planner_interval=4,
    )

    try:
        history = await agent.run()
        result = history.final_result()
        if result:
            print(f"\nExtracted Results:\n{result}")
        else:
            print("\nNo videos were found.")
    finally:
        await browser_session.stop()


if __name__ == "__main__":
    asyncio.run(main())