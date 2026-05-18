"""
Browser-Use Agent — Local FastAPI Backend
Accepts a task from the frontend, runs it through browser-use agent
(Ollama local worker + Gemini planner), and streams events via SSE.
Serves the frontend HTML directly.
"""

import asyncio
import json
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, TypeVar

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from browser_use import Agent, BrowserSession, BrowserProfile
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
os.chdir(BASE_DIR)  # ensure relative paths in agent work correctly

T = TypeVar('T', bound=BaseModel)

class MultiFallbackLLM(BaseChatModel):
    _verified_api_keys: bool = False

    def __init__(self, models: list[BaseChatModel]):
        self.models = models
        self.model = models[0].model

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def provider(self) -> str:
        return self.models[0].provider

    @property
    def name(self) -> str:
        return f"MultiFallback({[m.name for m in self.models]})"

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        last_exception = None
        for i, model in enumerate(self.models):
            try:
                return await model.ainvoke(messages, output_format, **kwargs)  # type: ignore
            except Exception as e:
                last_exception = e
                print(f"Fallback model {i} ({model.name}) failed: {e}")
                if i == len(self.models) - 1:
                    raise last_exception
        raise last_exception or Exception("All fallbacks failed")


CONDENSED_SYSTEM_PROMPT = """
You are a browser-use agent. Automate tasks by outputting structured JSON.
ALWAYS respond with valid JSON using this EXACT structure:
{
  "thinking": "Brief analysis of state and plan.",
  "evaluation_previous_goal": "Success/Failure of last action.",
  "memory": "1-2 sentences of progress.",
  "next_goal": "Next immediate step.",
  "action": [
    { "click": { "index": 5 } },
    { "input": { "index": 10, "text": "hello" } }
  ]
}
RULES:
1. Only interact with indexed elements [index].
2. POPUPS: Always REJECT or close cookie/privacy/notification popups immediately.
3. Navigation: Use `navigate` with `url`.
4. Visibility: If no relevant content is visible, SCROLL DOWN.
5. If stuck, change strategy or scroll.
6. TOOL NAMES (STRICT): 
   - `click`: {"index": 0}
   - `input`: {"index": 0, "text": "...", "clear": true}
   - `navigate`: {"url": "..."}
   - `search`: {"query": "...", "engine": "google"}
   - `scroll`: {"down": true, "pages": 1.0}
   - `done`: {"text": "...", "success": true}
7. VERIFICATION: Before each action, check if the current page already matches your final goal (e.g., if a video is playing). If so, call `done` immediately.
8. STABILITY: Do not click the same element more than twice if the page doesn't change. If stuck, try scrolling or slightly different search terms.
9. NO MARKDOWN: Never use ```json or any markdown blocks. Return the raw JSON object only.
10. Tool keys must match exactly: click, input, navigate, search, scroll, done.
"""

# ── Global State ────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.agent_task: asyncio.Task | None = None
        self.event_queues: list[asyncio.Queue] = []
        self.current_status = "idle"
        self.last_result: str | None = None
        self.last_error: str | None = None
        self.step_count = 0
        self.max_steps = 50

state = AppState()

# ── Lifespan ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Browser-Use Agent server started")
    yield
    print("Server shutting down")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Broadcast helpers ──────────────────────────────────────────────────────────
async def broadcast_event(event: dict):
    data = json.dumps(event)
    dead = []
    for q in state.event_queues:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        state.event_queues.remove(q)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "agent_status": state.current_status,
        "step_count": state.step_count,
        "last_result": state.last_result,
        "last_error": state.last_error,
    }

@app.post("/api/run")
async def run_task(body: dict):
    if state.current_status == "running":
        return {"error": "Agent is already running a task"}

    task = body.get("task", "").strip()
    if not task:
        return {"error": "No task provided"}

    max_steps = body.get("max_steps", 50)

    async def agent_runner():
        load_dotenv(BASE_DIR / ".env", override=True)
        
        state.current_status = "running"
        state.step_count = 0
        state.max_steps = max_steps
        state.last_result = None
        state.last_error = None

        await broadcast_event({
            "type": "status",
            "status": "running",
            "task": task,
            "max_steps": max_steps,
            "ts": time.time(),
        })

        browser_session = None
        try:
            browser_session = BrowserSession(
                browser_profile=BrowserProfile(
                    wait_before_action=3.0,
                    dom_highlight_elements=True,
                    highlight_elements=False,
                )
            )

            main_llm = ChatOpenAI(
                model='meta/llama-3.3-70b-instruct',
                base_url='https://integrate.api.nvidia.com/v1',
                api_key=os.getenv('NVIDIA_API_KEY', ''),
                temperature=0.0,
                dont_force_structured_output=True,
                timeout=30.0,
            )
            
            fallback_llm = ChatOpenAI(
                model='llama-3.3-70b-versatile',
                base_url='https://api.groq.com/openai/v1',
                api_key=os.getenv('GROQ_API_KEY', ''),
                temperature=0.0,
                dont_force_structured_output=True,
                timeout=25.0,
            )

            fallback_cerebras = ChatOpenAI(
                model="llama3.1-8b",
                base_url="https://api.cerebras.ai/v1",
                api_key=os.getenv("CEREBRAS_API_KEY", ""),
                temperature=0.0,
                dont_force_structured_output=True,
            )

            # Combine fallbacks into our custom multi-fallback handler
            fallback_chain = MultiFallbackLLM([fallback_llm, fallback_cerebras])

            async def on_step(browser_state, agent_output, step_number):
                state.step_count = step_number
                action_info = None
                if agent_output and agent_output.action:
                    try:
                        action_info = agent_output.action[0].model_dump() if agent_output.action else None
                    except Exception:
                        action_info = str(agent_output.action)

                await broadcast_event({
                    "type": "step",
                    "step": step_number,
                    "action": action_info,
                    "url": browser_state.url if browser_state else None,
                    "ts": time.time(),
                })
                await asyncio.sleep(0.5)

            agent = Agent(
                task=task,
                llm=main_llm,
                fallback_llm=fallback_chain,
                browser_session=browser_session,
                use_vision=False,
                save_conversation_path='logs/conversation.json',
                register_new_step_callback=on_step,
                override_system_message=CONDENSED_SYSTEM_PROMPT,
                enable_planning=False,
                include_attributes=["title", "aria-label", "name", "role", "type", "value", "placeholder", "id"],
                max_clickable_elements_length=8000,
            )

            history = await agent.run(max_steps=max_steps)
            result = history.final_result()

            state.current_status = "done"
            state.last_result = str(result) if result else "No result returned"

            await broadcast_event({
                "type": "status",
                "status": "done",
                "result": state.last_result[:4000],
                "ts": time.time(),
            })

        except asyncio.CancelledError:
            state.current_status = "stopped"
            await broadcast_event({"type": "status", "status": "stopped", "ts": time.time()})
        except Exception as e:
            state.current_status = "error"
            state.last_error = str(e)
            await broadcast_event({
                "type": "status",
                "status": "error",
                "error": str(e),
                "trace": traceback.format_exc()[-1500:],
                "ts": time.time(),
            })
        finally:
            if browser_session:
                try:
                    await browser_session.stop()
                except Exception:
                    pass

    state.agent_task = asyncio.create_task(agent_runner())
    return {"ok": True, "task": task}

@app.post("/api/stop")
async def stop_task():
    if state.agent_task and not state.agent_task.done():
        state.agent_task.cancel()
        return {"ok": True}
    return {"ok": False, "reason": "No active task"}

@app.get("/api/stream")
async def sse_stream():
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    state.event_queues.append(q)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'connected', 'agent_status': state.current_status})}\n\n"
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=25)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in state.event_queues:
                state.event_queues.remove(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/")
async def serve_frontend():
    return FileResponse(BASE_DIR / "frontend.html")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
