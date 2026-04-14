import sys
import os
import json
import requests
import subprocess
import time
from datetime import datetime
import mlx.core as mx 
import psutil 
import multiprocessing as mp
from queue import Empty
import shutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QProgressBar, QFileDialog, QMessageBox, QSplitter,
                             QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# --- Configuration ---
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

TODAY = datetime.now().strftime("%Y-%m-%d")
TODAY_READABLE = datetime.now().strftime("%B %d, %Y")

MEETING_TEMPLATE = f"""---
date: {TODAY}
type: meeting
context: "[[Work]]"
attendees: []
status: draft
tags: [meeting]
---

# Meeting — {TODAY_READABLE}

## Context
- **Related contact(s):** - **Meeting purpose:** ## Quick Summary
_

## Key Points Discussed
- 

## Commitments Made
| Person | Commitment | Due |
|--------|-----------|-----|
|  |  |  |

## My Action Items
- [ ] 

## Open Questions / Unresolved
- 

## Next Meeting
- **Date:** """

# --- Isolated Worker Process ---
def whisper_worker(audio_path, model_repo, conn):
    """Isolated process to ensure absolute VRAM reclamation upon termination."""
    try:
        import os
        # Prevent librosa/joblib from spawning 'loky' background processes that leak
        os.environ["LOKY_MAX_CPU_COUNT"] = "1"
        os.environ["JOBLIB_MULTIPROCESSING"] = "0"
        
        import warnings
        # Gag librosa's harmless fallback and deprecation warnings
        warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        import librosa
        import numpy as np
        import mlx_whisper
        
        conn.send({"status": "Cleaning audio (muting background noise)..."})
        y, sr = librosa.load(audio_path, sr=16000)
        
        # ZERO-OUT SILENCE: Preserves exact audio length to prevent jump-cuts, 
        # but completely mutes background crowd noise to prevent hallucinations.
        intervals = librosa.effects.split(y, top_db=30)
        y_clean = np.zeros_like(y)
        for start, end in intervals:
            y_clean[start:end] = y[start:end]
            
        conn.send({"status": "Transcribing on Mac GPU... (This may take a few minutes)"})
        
        # UN-CHUNKED: Processing the entire continuous array at once to prevent cut-off words
        result = mlx_whisper.transcribe(
            y_clean, 
            path_or_hf_repo=model_repo,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.4
        )
            
        conn.send({"status": "Unloading Whisper from GPU..."})
        conn.send({"done": result["text"].strip()})
        
    except Exception as e:
        conn.send({"error": str(e)})
    finally:
        conn.close()


# --- Background Threads ---
class SystemMonitorThread(QThread):
    stats_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        while self._is_running:
            try:
                ram = psutil.virtual_memory().percent
                
                battery = psutil.sensors_battery()
                batt_str = f"{int(battery.percent)}%" if battery else "N/A"
                if battery and battery.power_plugged:
                    batt_str += " (Charging)"

                stats_text = f"RAM: {ram}% | Battery: {batt_str}"
                self.stats_signal.emit(stats_text)
                
                time.sleep(1.5)
            except Exception:
                pass

    def stop(self):
        self._is_running = False


class TranscribeManager(QThread):
    progress_signal = pyqtSignal(int, str)
    text_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file
        self.process = None
        self.parent_conn = None
        self._is_running = True

    def run(self):
        ctx = mp.get_context('spawn')
        # Using a Pipe entirely bypasses Python's semaphore tracking
        self.parent_conn, child_conn = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=whisper_worker, args=(self.audio_file, WHISPER_MODEL, child_conn))
        self.process.start()
        
        # Close the child end in the parent process
        child_conn.close()
        
        while self._is_running and self.process.is_alive():
            try:
                # Poll waits for up to 0.5 seconds for data
                if self.parent_conn.poll(0.5):
                    msg = self.parent_conn.recv()
                    if "status" in msg:
                        self.progress_signal.emit(0, msg["status"]) 
                    elif "error" in msg:
                        self.error_signal.emit(msg["error"])
                        break
                    elif "done" in msg:
                        self.progress_signal.emit(100, "Transcription Finished.")
                        self.finished_signal.emit(msg["done"])
                        break
            except EOFError:
                # Catch if the OS forcibly severs the pipe
                break
                
        self._cleanup_process()

    def stop(self):
        self._is_running = False

    def _cleanup_process(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                
        if self.parent_conn:
            try:
                self.parent_conn.close()
            except Exception:
                pass
                
        try:
            mx.clear_cache()
        except Exception:
            pass


class LMStudioThread(QThread):
    text_signal = pyqtSignal(str)
    reasoning_signal = pyqtSignal(str) 
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, transcript, model_id):
        super().__init__()
        self.transcript = transcript
        self.model_id = model_id
        self._is_running = True

    def unload_model(self):
        lms_paths = [
            shutil.which("lms"),
            "/usr/local/bin/lms",
            "/opt/homebrew/bin/lms",
            os.path.expanduser("~/.cache/lm-studio/bin/lms"),
            os.path.expanduser("~/.lmstudio/bin/lms")
        ]
        lms_exe = next((p for p in lms_paths if p and os.path.exists(p)), None)
        
        if lms_exe:
            try:
                subprocess.run([lms_exe, "unload", "--all"], check=False, capture_output=True)
            except Exception:
                pass

    def run(self):
        self.status_signal.emit(f"Booting {self.model_id} into RAM... (Takes ~15-30s)")
        
        prompt = (
            "You are an expert executive assistant. Read the transcript and fill out the Markdown template "
            "based ONLY on the transcript. \n\n"
            "CRITICAL CONTEXT: The company we work for is 'Verkada'. The speech-to-text transcription may have "
            "hallucinated or misspelled this as 'Ricotta', 'Mercado', 'Riccardo', 'Brocade', 'Canada', or other similar "
            "sounding phonetic variations. Please actively correct these misspellings to 'Verkada' in the notes.\n\n"
            "Return ONLY the fully populated Markdown template without any conversational wrapper.\n\n"
            f"--- TRANSCRIPT ---\n{self.transcript}\n\n--- TEMPLATE ---\n{MEETING_TEMPLATE}"
        )
        
        payload = {
            "model": self.model_id, 
            "messages": [
                {"role": "system", "content": "You are a precise assistant that outputs raw, formatted markdown."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "stream": True 
        }
        
        try:
            response = requests.post(LM_STUDIO_URL, json=payload, stream=True)
            response.raise_for_status()
            
            self.status_signal.emit("Generating Notes...")
            
            notes_text = ""
            reasoning_text = ""
            is_thinking = False

            for line in response.iter_lines():
                if not self._is_running:
                    response.close()
                    break

                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                
                                reasoning = delta.get('reasoning_content', '')
                                content = delta.get('content', '')
                                
                                if reasoning:
                                    reasoning_text += reasoning
                                    # CAP RAM: Only keep the rolling tail end
                                    if len(reasoning_text) > 500:
                                        reasoning_text = "..." + reasoning_text[-347:]
                                    self.reasoning_signal.emit(reasoning_text)
                                
                                if content:
                                    if "<think>" in content:
                                        is_thinking = True
                                        content = content.replace("<think>", "")
                                    
                                    if "</think>" in content:
                                        is_thinking = False
                                        parts = content.split("</think>")
                                        reasoning_text += parts[0]
                                        if len(reasoning_text) > 350:
                                            reasoning_text = "..." + reasoning_text[-347:]
                                        self.reasoning_signal.emit(reasoning_text)
                                        
                                        if len(parts) > 1:
                                            notes_text += parts[1]
                                            self.text_signal.emit(notes_text)
                                        continue

                                    if is_thinking:
                                        reasoning_text += content
                                        # CAP RAM
                                        if len(reasoning_text) > 350:
                                            reasoning_text = "..." + reasoning_text[-347:]
                                        self.reasoning_signal.emit(reasoning_text)
                                    else:
                                        notes_text += content
                                        self.text_signal.emit(notes_text)
                                        
                        except json.JSONDecodeError:
                            continue
                            
            self.status_signal.emit("Unloading model...")
            self.unload_model()

            if not self._is_running:
                self.status_signal.emit("Generation aborted. Memory Freed.")
            else:
                self.status_signal.emit("Notes Generation Complete.")
                # When done, clear the ghosty rolling text to signal completion
                self.reasoning_signal.emit("Thought process finished.")
                self.finished_signal.emit(notes_text)
            
        except requests.exceptions.ConnectionError:
            self.unload_model()
            self.error_signal.emit("Could not connect to LM Studio. Ensure the Local Server is running on port 1234.")
        except Exception as e:
            self.unload_model()
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local AI Meeting Assistant")
        self.setGeometry(100, 100, 1250, 850)
        self.audio_path = None
        self.transcribe_thread = None
        self.lm_thread = None
        
        self.setAcceptDrops(True)
        self.setup_ui()
        self.fetch_models() 
        
        self.monitor_thread = SystemMonitorThread()
        self.monitor_thread.stats_signal.connect(self.update_telemetry)
        self.monitor_thread.start()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        header_layout = QHBoxLayout()
        header = QLabel("AI Meeting Assistant")
        
        header_font = QFont()
        header_font.setPointSize(24)
        header_font.setBold(True)
        header.setFont(header_font)
        
        header_layout.addWidget(header)
        
        self.telemetry_lbl = QLabel("Initializing sensors...")
        self.telemetry_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.telemetry_lbl)
        
        layout.addLayout(header_layout)
        
        self.status_lbl = QLabel("Ready. Drag and drop an audio file anywhere.")
        layout.addWidget(self.status_lbl)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100) 
        layout.addWidget(self.progress_bar)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        self.btn_select_file = QPushButton("Browse Audio")
        self.btn_select_file.clicked.connect(self.select_file)
        
        self.btn_transcribe = QPushButton("Transcribe Audio")
        self.btn_transcribe.clicked.connect(self.start_transcription)
        self.btn_transcribe.setEnabled(False)
        
        self.btn_stop = QPushButton("Stop Process")
        self.btn_stop.clicked.connect(self.stop_action)
        self.btn_stop.setEnabled(False)

        controls_layout.addWidget(self.btn_select_file)
        controls_layout.addWidget(self.btn_transcribe)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addStretch() 
        layout.addLayout(controls_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)
        
        lbl_transcript = QLabel("Raw Transcript")
        font_bold = QFont()
        font_bold.setBold(True)
        lbl_transcript.setFont(font_bold)
        left_layout.addWidget(lbl_transcript)
        
        self.transcript_box = QTextEdit()
        self.transcript_box.setReadOnly(True)
        left_layout.addWidget(self.transcript_box)
        
        self.btn_save_transcript = QPushButton("Save Transcript")
        self.btn_save_transcript.clicked.connect(lambda: self.save_file(self.transcript_box.toPlainText(), "transcript.txt"))
        left_layout.addWidget(self.btn_save_transcript)
        
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)
        
        model_layout = QHBoxLayout()
        lbl_model = QLabel("Target LLM:")
        lbl_model.setFont(font_bold)
        model_layout.addWidget(lbl_model)
        
        self.model_dropdown = QComboBox()
        self.model_dropdown.setMinimumWidth(200)
        model_layout.addWidget(self.model_dropdown)
        
        self.btn_refresh_models = QPushButton("Refresh")
        self.btn_refresh_models.clicked.connect(self.fetch_models)
        model_layout.addWidget(self.btn_refresh_models)
        
        self.btn_generate = QPushButton("Generate Notes")
        self.btn_generate.clicked.connect(self.start_generation)
        self.btn_generate.setEnabled(False)
        model_layout.addWidget(self.btn_generate)
        
        right_layout.addLayout(model_layout)
        
        right_vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        
        reasoning_widget = QWidget()
        reasoning_layout = QVBoxLayout(reasoning_widget)
        reasoning_layout.setContentsMargins(0, 5, 0, 0)
        lbl_reasoning = QLabel("AI Thought Process")
        lbl_reasoning.setFont(font_bold)
        reasoning_layout.addWidget(lbl_reasoning)
        
        # UI TWEAK: The "Ghosty" Rolling Text Box
        # UI TWEAK: The "Ghosty" Rolling Text Box
        self.reasoning_box = QTextEdit()
        self.reasoning_box.setReadOnly(True)
        self.reasoning_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) 
        # By only styling the text, it inherits the perfect native macOS background and border
        self.reasoning_box.setStyleSheet("""
            QTextEdit {
                color: #8E8E93; 
                font-style: italic; 
            }
        """)
        reasoning_layout.addWidget(self.reasoning_box)
        right_vertical_splitter.addWidget(reasoning_widget)

        notes_widget = QWidget()
        notes_layout = QVBoxLayout(notes_widget)
        notes_layout.setContentsMargins(0, 5, 0, 0)
        lbl_notes = QLabel("Formatted Notes")
        lbl_notes.setFont(font_bold)
        notes_layout.addWidget(lbl_notes)
        self.notes_box = QTextEdit()
        notes_layout.addWidget(self.notes_box)
        
        self.btn_save_notes = QPushButton("Save Notes")
        self.btn_save_notes.clicked.connect(lambda: self.save_file(self.notes_box.toPlainText(), f"Meeting_Notes_{TODAY}.md"))
        notes_layout.addWidget(self.btn_save_notes)
        right_vertical_splitter.addWidget(notes_widget)
        
        right_vertical_splitter.setSizes([100, 500]) # Made reasoning box smaller to save space
        right_layout.addWidget(right_vertical_splitter)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])
        layout.addWidget(splitter)

    # --- Methods ---
    def update_telemetry(self, text):
        self.telemetry_lbl.setText(text)

    def fetch_models(self):
        self.model_dropdown.clear()
        self.model_dropdown.addItem("Fetching...")
        try:
            base_url = LM_STUDIO_URL.replace("/chat/completions", "")
            response = requests.get(f"{base_url}/models", timeout=3)
            
            if response.status_code == 200:
                models = [m["id"] for m in response.json().get("data", [])]
                self.model_dropdown.clear()
                if models:
                    self.model_dropdown.addItems(models)
                else:
                    self.model_dropdown.addItem("No models found")
            else:
                self.model_dropdown.clear()
                self.model_dropdown.addItem("Error: Check LM Studio")
        except Exception:
            self.model_dropdown.clear()
            self.model_dropdown.addItem("Offline / Server not running")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile().lower()
                if file_path.endswith(('.m4a', '.mp3', '.wav', '.ogg', '.flac')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            self.audio_path = file_path
            self.status_lbl.setText(f"Loaded: {file_path}")
            self.btn_transcribe.setEnabled(True)
            event.acceptProposedAction()

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.m4a *.mp3 *.wav *.ogg *.flac)")
        if file_name:
            self.audio_path = file_name
            self.status_lbl.setText(f"Loaded: {file_name}")
            self.btn_transcribe.setEnabled(True)

    def start_transcription(self):
        if not self.audio_path: return
        self.transcript_box.clear()
        self.btn_transcribe.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_select_file.setEnabled(False)
        
        self.progress_bar.setRange(0, 0)
        
        self.transcribe_thread = TranscribeManager(self.audio_path)
        self.transcribe_thread.progress_signal.connect(self.update_progress)
        self.transcribe_thread.text_signal.connect(self.update_transcript)
        self.transcribe_thread.finished_signal.connect(self.transcription_finished)
        self.transcribe_thread.error_signal.connect(self.show_error)
        self.transcribe_thread.start()

    def stop_action(self):
        if self.transcribe_thread and self.transcribe_thread.isRunning():
            self.transcribe_thread.stop()
        
        if self.lm_thread and self.lm_thread.isRunning():
            self.lm_thread.stop()
            
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.btn_stop.setEnabled(False)
        self.btn_transcribe.setEnabled(True)
        self.btn_select_file.setEnabled(True)
        self.status_lbl.setText("Process stopped by user.")
        
        if self.transcript_box.toPlainText().strip():
            self.btn_generate.setEnabled(True)

    def update_progress(self, percent, message):
        if percent == 100:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
        self.status_lbl.setText(message)

    def update_transcript(self, text):
        self.transcript_box.setPlainText(text)
        scrollbar = self.transcript_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def transcription_finished(self, text):
        self.btn_stop.setEnabled(False)
        self.btn_transcribe.setEnabled(True)
        self.btn_select_file.setEnabled(True)
        
        if text.strip():
            self.transcript_box.setPlainText(text)
            self.btn_generate.setEnabled(True)

    def start_generation(self):
        transcript = self.transcript_box.toPlainText()
        selected_model = self.model_dropdown.currentText()
        
        if not transcript: return
        if "Error" in selected_model or "Offline" in selected_model or "No models" in selected_model or "Fetching" in selected_model:
            self.show_error("Please start LM Studio server and select a valid model.")
            return
            
        self.notes_box.clear()
        self.reasoning_box.clear()
        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.lm_thread = LMStudioThread(transcript, selected_model)
        self.lm_thread.status_signal.connect(lambda msg: self.status_lbl.setText(msg))
        self.lm_thread.text_signal.connect(self.update_notes)
        self.lm_thread.reasoning_signal.connect(self.update_reasoning)
        self.lm_thread.finished_signal.connect(self.generation_finished)
        self.lm_thread.error_signal.connect(self.show_error)
        self.lm_thread.start()

    def generation_finished(self, text):
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_notes(self, text):
        self.notes_box.setPlainText(text)
        scrollbar = self.notes_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_reasoning(self, text):
        self.reasoning_box.setPlainText(text)
        scrollbar = self.reasoning_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_error(self, err_msg):
        self.status_lbl.setText(f"Error: {err_msg}")
        QMessageBox.critical(self, "Error", err_msg)
        
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.btn_transcribe.setEnabled(True)
        self.btn_select_file.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.transcript_box.toPlainText().strip():
            self.btn_generate.setEnabled(True)

    def save_file(self, content, default_name):
        if not content.strip(): return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", default_name, "Text Files (*.txt *.md)")
        if file_name:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)
            self.status_lbl.setText(f"Saved: {file_name}")

    def closeEvent(self, event):
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.stop()
            self.monitor_thread.wait()
        event.accept()

if __name__ == "__main__":
    mp.freeze_support() 
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())