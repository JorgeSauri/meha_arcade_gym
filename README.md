# Arcade Gym: A Benchmark Suite for Spatial Grounding and Active Exploration

Welcome to **Arcade Gym**, the official open-source benchmark suite introduced in the BICA 2026 paper:
**"MEHA: Interleaved Wake-Sleep Consolidation for Spatial Grounding in Small LLM-Based Agents"**.

Arcade Gym is designed to evaluate spatial reasoning, motor passivity, and active exploration in situated agents. It features **8 diverse 2D mini-games** with discrete state-action spaces, synthetic sound effects, and dense causal feedback.

This repository is fully self-contained and open-source, allowing third-party researchers to reproduce our experiments, evaluate their own models, and compare performance horizontally against contemporary methods.

---

## 🚀 Key Features

* **8 Diverse Games:** Includes physics-based puzzles, memory games, and hand-eye coordination challenges.
* **OpenAI-Compatible API Integration:** Easily connect any frontier-scale model (e.g., GPT-4o, GPT-4o-mini) or local LLM (via Ollama, LM Studio, vLLM, etc.) directly through the GUI.
* **Vision & Text Modes:** Evaluate models using direct visual board inputs (PNG frames) or flat text-based grid arrays.
* **Interactive Pygame GUI:** A beautiful, dark-themed control panel with live performance metrics, console logs, a dedicated model reasoning panel, and manual play support.
* **Live Metrics Dashboard:** Tracks **Motor Passivity (No-Op Rate)**, **Causal Coverage (Exploration)**, **Cumulative Reward**, and **Task Completion (Solved Rate)** in real-time.
* **Customizable Episode Length:** Choose exactly how many steps a model or player can take per episode with the **Max Steps** selector.
* **Dedicated Reasoning Panel:** Watch your LLM's raw thoughts, observations, and step-by-step reasoning unfold in real-time in a beautifully formatted column.

---

## 📦 Installation

To run Arcade Gym, you need Python 3.8+ and a few standard libraries.

1. Clone or copy this directory into your workspace.
2. Install the required dependencies:

```bash
pip install -r arcade_gym/requirements.txt
```

---

## 🎮 Running the Interactive Evaluator GUI

To launch the Pygame GUI, simply run:

```bash
python arcade_gym/play_model.py
```

### 🕹️ Game Selection & Episode Length
The GUI features dedicated selectors that allow you to customize the evaluation:
* **Game Selector:** Choose exactly which game to play or evaluate.
  * **`random` (Default):** The environment automatically selects a random game from the suite upon initialization or reset.
  * **Specific Games:** Use the `<` and `>` arrow buttons on the selector to choose any of the 8 custom games by their exact paper code names (e.g., `p61_sort`, `nibbles`, `platformer`, `fabulous_fred`).
  * **Next Random Game:** Click this button to instantly choose and load another random game from the suite.
* **Max Steps per Episode:** Type any integer (e.g., `100`, `200`, `500`) to define the maximum steps allowed per episode. The environment will dynamically read and apply this limit upon reset.

### 🕹️ Manual Play Mode
Want to play the games yourself to understand their dynamics?
1. Click the **"Manual Play Mode"** button.
2. Use the following controls:
   * **Arrow Keys:** Move Up, Down, Left, Right (Actions `A1`-`A4`).
   * **Spacebar:** Primary Action (Action `A5`).
   * **Q Key:** Localized Action (Action `A6`).
   * **W Key:** Secondary Action (Action `A7`).
   * **E Key:** Wait / No-Op (Action `A8`).

---

## 🤖 Connecting Your Own Model

The GUI makes it incredibly simple to plug in and evaluate your own models:

1. **OpenAI Models:**
   * Paste your **OpenAI API Key** into the text field (you can paste with `Ctrl+V`).
   * Keep the default API Base URL (`https://api.openai.com/v1`).
   * Specify the model name (e.g., `gpt-4o-mini`, `gpt-4o`).
   * Click **"Start Model Play"**.

2. **Local Models (Ollama, LM Studio, vLLM, etc.):**
   * Run your local model server (e.g., `ollama serve`).
   * In the GUI, change the **API Base URL** to your local endpoint (e.g., `http://localhost:11434/v1` for Ollama, or `http://localhost:1234/v1` for LM Studio).
   * Enter your local model name (e.g., `llama3`, `qwen2.5`, `mistral`).
   * Toggle **"Vision Enabled"** off if your local model does not support vision (it will automatically send the grid as a flat text array).
   * Click **"Start Model Play"**.

---

## 📊 Live Performance Metrics

The GUI tracks and displays the following key metrics defined in the MEHA paper:

1. **Motor Passivity (No-Op Rate %):** The percentage of actions that resulted in no change to the environment state. Lower values indicate more effective, situated motor execution.
2. **Causal Coverage (%):** The percentage of distinct actions (out of 8) that successfully altered the game state. Higher values indicate more thorough active exploration of the action-effect space.
3. **Solved Rate (%):** The proportion of games successfully completed/solved (achieving maximum positive reward).
4. **Cumulative Reward:** The total extrinsic reward accumulated during the current episode.

---

## 🎮 Included Games and Rules

1. **`p61_sort` (Sorting):** Push colored blocks into their matching color containers.
2. **`nibbles` (Snake):** Eat green, blue, and yellow food for positive rewards; avoid red poison food and wall collisions.
3. **`angry_blocks` (Slingshot):** Launch a square projectile in a gravity arc to hit and collapse block towers.
4. **`platformer` (Jump & Run):** Jump over hazards, collect pickups, and reach the goal on the far right.
5. **`phoenix_duel` (Space Shooter):** Move and fire lasers to shoot down moving geometric enemies while dodging their bullets.
6. **`sky_catch` (Catcher):** Move a basket left and right to catch falling reward objects while dodging deadly hazards.
7. **`dr_capsule` (Dr. Mario Puzzle):** Rotate and drop bicolor capsules to form same-color matches and clear viruses.
8. **`fabulous_fred` (Simon Memory):** Watch the machine's color/sound sequence and repeat it in the correct order.
