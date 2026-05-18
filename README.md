# 🔮 Autonomous AI Browser Agent

An advanced, autonomous web-automation suite that empowers AI models to control, browse, and extract data from the web in real time. It features a dual-interface architecture tailored for **Local Premium Desktop environments** and **Remote Cloud Sandbox environments** (like Kaggle or custom servers), orchestrated by a high-resiliency multi-fallback LLM chain.

---

## 🚀 Key Features

*   **Multi-Fallback LLM Resiliency Chain:** Features a dynamic cascading router that automatically fails over if an API goes down:
    $$\text{NVIDIA Llama 3.3 (70B)} \longrightarrow \text{Groq Llama 3.3 (70B)} \longrightarrow \text{Cerebras Llama 3.1 (8B)}$$
*   **Dual-Frontend Client Systems:**
    *   **Local Dashboard (`frontend.html`):** A premium dark-slate workspace interface featuring a futuristic Siri-inspired glowing voice-wave visualizer (powered by Lottie), progress tracking, and interactive action feeds.
    *   **Remote Sandbox Browser (`index.html`):** Formatted for remote server instances (like Kaggle notebooks) with a low-latency WebSockets-based live browser viewport stream and real-time step monitoring.
*   **Intelligent System Prompts:** Robust popup rejection filters, smart scroll behaviors, strict JSON constraints, and selector highlighting built directly into the agent's core instruction set.
*   **Multi-Agent Coordination (`agent.py`):** Standalone hybrid setup pairing a local execution model (**Ollama + Qwen 2.5 3B**) for actions and a cloud model (**Gemini 2.0 Flash Lite**) for high-level strategy planning.

---

## 📁 Project Architecture

The workspace consists of the following key files:

| File Name | Role | Primary Environment | Interface |
| :--- | :--- | :--- | :--- |
| [**`server.py`**](file:///c:/Users/dilee/Downloads/Projects/BU/BU/server.py) | Local FastAPI backend containing SSE streams, multi-fallback routers, and HTML server endpoints. | Local Desktop | Serves `frontend.html` |
| [**`frontend.html`**](file:///c:/Users/dilee/Downloads/Projects/BU/BU/frontend.html) | Premium dark-themed dashboard UI featuring animated Siri-like visualizers and step metrics. | Local Desktop | Renders at `/` |
| [**`agent.py`**](file:///c:/Users/dilee/Downloads/Projects/BU/BU/agent.py) | Standalone local CLI runner using Ollama (Qwen) + Gemini Flash planning. | Command Line | Console logs |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Node.js/npm** (optional, only if using certain custom frontends) installed.

### 2. Set Up a Virtual Environment
Navigate to the project root and create a virtual environment:
```bash
# Create the environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Python Dependencies
Install the required packages using the curated `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Install Playwright Browsers
Install the headless Chromium browser instances required by `browser-use`:
```bash
playwright install chromium
```

### 5. Setup Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
# Required for server.py multi-fallback system
NVIDIA_API_KEY=nvapi-your-key-here
GROQ_API_KEY=gsk_your-key-here
CEREBRAS_API_KEY=csk-your-key-here

# Required for agent.py planner
GOOGLE_API_KEY=AIzaSyYourKeyHere
```

---

## 🕹️ How to Run the Applications

### Option A: Premium Local Server (Recommended)
This starts the local FastAPI server serving the modern animated frontend interface.

1. Start the server:
   ```bash
   python server.py
   ```
2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```
3. Type in your task (e.g., *"Search for the best open-source AI projects on GitHub"*), adjust the step limit, and click **▶ Run**. Watch the Siri visualizer glow while the agent completes your task in the background!

---


### Option B: Standalone Command-Line Runner
For direct terminal execution without launching a web server:

1. Ensure **Ollama** is running locally and has the `qwen2.5:3b` model installed:
   ```bash
   ollama run qwen2.5:3b
   ```
2. Run the standalone script:
   ```bash
   python agent.py
   ```
3. The script will initialize an Ollama instance to control the browser, backed by Gemini Flash Lite to design the strategic steps, printing logs directly to your shell.

---

## 🧩 How it Works Under the Hood

1. **Orchestration:** The user enters a task, which is received by FastAPI.
2. **Dynamic DOM Parsing:** `browser-use` launches a Playwright instance and parses the visible page DOM, highlighting interactive elements with unique numerical indices (e.g., `[index=12]`).
3. **Structured Thinking:** The active LLM (guided by the strict custom system prompt) determines the next immediate step and outputs structured JSON containing its `thinking` process and an `action` array (e.g., `[{"click": {"index": 5}}]`).
4. **Resiliency Failover:** If the primary LLM API hits rate limits or throws an error, the `MultiFallbackLLM` immediately captures the exception and retries the exact action using the fallback providers in sequence.
5. **Streaming Visuals:** In local mode, the state and log stream via SSE. In remote mode, the system runs an asynchronous loop capturing the viewport, shrinking and converting it to high-quality JPEGs, and transmitting it as base64 over WebSockets to your browser.
