# Local AI Meeting Assistant

I got tired of seeing companies sell $100+ hardware that simply contains a few microphones and a battery and outsourcing the AI usage to some other company so I made this repo. It is a privacy-first, fully local macOS application designed to transcribe meeting audio and automatically generate structured action items and summaries. Built specifically for Apple Silicon (M-series), this app leverages hardware acceleration for transcription and hooks into local LLMs to keep your sensitive corporate data entirely off the cloud. Record the audio using the device of your choice, whether it is your Apple Watch, a third-party microphone, or even native your laptop speakers! This keeps battery usage to a minimum and assures you have small file sizes for when meetings push a couple hours. Then drop the audio file into the app and begin your process of transcribing, and even generating Markdown notes based on a template to import into your favorite note taking app!

## Key Features

* **Zero Cloud Dependency:** Everything runs directly on your Mac. No data is sent to OpenAI, Google, or third-party servers.
* **Apple Silicon Accelerated:** Uses Apple's MLX framework and the `Whisper Large v3 Turbo` model to transcribe hours of audio via your Mac's GPU in minutes.
* **Smart Audio Cleanup:** Automatically detects and mutes background noise and dead air before transcription to prevent AI hallucinations.
* **Memory Juggling:** Safely manages VRAM and System RAM. It isolates the Whisper transcription process and automatically evicts LLMs from memory when finished to prevent system crashes.
* **Built-in System Telemetry:** Live tracking of System RAM and Battery.
* **Verkada Optimized:** Includes a pre-configured LLM grounding prompt to automatically fix phonetic misspellings of company's specific industry terms (e.g., correcting "Ricotta" to "Verkada") when creating notes.

> [!WARNING] LLM CATCH
> The output is only as good as the LLM that you are running on. Small, non-multi-modal LLM = poor notes.

## Prerequisites

To run and build this application, you need:

1. **Apple Silicon Mac** (M1, M2, M3, M4).
2. **LM Studio** installed in your `/Applications` folder.
3. **Python 3.10+**.

## Setup & Installation

### 1. Configure LM Studio

This application relies on LM Studio to process the transcripts and generate the notes.

1. Download and install [LM Studio](https://lmstudio.ai/).
2. Download your preferred Large Language Model (e.g., Qwen 3.5, DeepSeek-R1, Llama 3).
3. Navigate to the **Developer Tab** (`<->` icon).
4. Start the **Local Inference Server** on port `1234`.

### 2. Set Up the Python Environment

Open your terminal and configure a virtual environment:

```bash
# Clone or navigate to the project directory
cd ai-audio-meeting-assistant

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
