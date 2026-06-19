#!/usr/bin/env encoding=utf-8
from __future__ import annotations

import os
import sys
import json
import math
import time
import base64
import random
import requests
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pygame

# Añadir el directorio raíz al path para poder importar arcade_gym si se ejecuta desde fuera
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arcade_gym.arcade_games import ArcadeSuiteEnv, GAME_CLASSES, ACTION_LABELS
from arcade_gym.arcade_audio import make_audio_event
from arcade_gym.affect import affect_for_transition
from arcade_gym.language import language_for_transition

# Colores del tema oscuro (Cursor/VS Code Style)
COLOR_BG = (30, 30, 30)
COLOR_PANEL = (37, 37, 38)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_MUTED = (130, 130, 130)
COLOR_ACCENT = (0, 122, 204)  # Azul VS Code
COLOR_ACCENT_HOVER = (20, 142, 224)
COLOR_SUCCESS = (76, 175, 80)
COLOR_DANGER = (244, 67, 54)
COLOR_BORDER = (51, 51, 51)
COLOR_INPUT_BG = (45, 45, 45)

# Paleta de colores para el grid de juegos (P6.1/ARC)
PALETTE_RGB = [
    (0, 0, 0),        # 0: Negro
    (0, 116, 217),    # 1: Azul
    (255, 65, 54),    # 2: Rojo
    (46, 204, 64),    # 3: Verde
    (255, 220, 0),    # 4: Amarillo
    (170, 170, 170),  # 5: Gris
    (240, 18, 190),   # 6: Magenta
    (255, 133, 27),   # 7: Naranja
    (127, 219, 255),  # 8: Celeste
    (135, 12, 37),    # 9: Guinda
    (1, 255, 112),    # 10: Lima
    (255, 255, 255),  # 11: Blanco
    (92, 52, 0),      # 12: Café oscuro
    (191, 191, 191),  # 13: Plateado
    (64, 224, 208),   # 14: Turquesa
    (128, 0, 128),    # 15: Púrpura
]

def get_clipboard_text() -> str:
    # 1. Intentar con tkinter (estándar en la mayoría de sistemas)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        return str(text)
    except Exception:
        pass

    # 2. Intentar con xclip en Linux (muy común)
    try:
        import subprocess
        p = subprocess.Popen(['xclip', '-selection', 'clipboard', '-o'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode == 0:
            return out.decode('utf-8')
    except Exception:
        pass

    # 3. Intentar con xsel en Linux
    try:
        import subprocess
        p = subprocess.Popen(['xsel', '-clipboard', '-o'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode == 0:
            return out.decode('utf-8')
    except Exception:
        pass

    # 4. Intentar con pyperclip si está instalado
    try:
        import pyperclip
        return str(pyperclip.paste())
    except Exception:
        pass

    return ""

class UIElement:
    def draw(self, screen: pygame.Surface) -> None:
        pass
    def handle_event(self, event: pygame.event.Event) -> bool:
        return False

class Button(UIElement):
    def __init__(self, x: int, y: int, w: int, h: int, text: str, callback: callable, color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER, self.rect, width=1, border_radius=4)
        
        txt_surf = font.render(self.text, True, (255, 255, 255))
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

class Selector(UIElement):
    def __init__(self, x: int, y: int, w: int, h: int, label: str, options: list[str], default_idx: int = 0, callback: callable = None) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.options = options
        self.idx = default_idx
        self.callback = callback
        self.is_hovered_left = False
        self.is_hovered_right = False
        self.btn_left = pygame.Rect(x, y, 30, h)
        self.btn_right = pygame.Rect(x + w - 30, y, 30, h)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        # Dibujar etiqueta arriba
        lbl_surf = font.render(self.label, True, COLOR_TEXT_MUTED)
        screen.blit(lbl_surf, (self.rect.x, self.rect.y - 18))

        # Dibujar caja principal
        pygame.draw.rect(screen, COLOR_INPUT_BG, self.rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER, self.rect, width=1, border_radius=4)

        # Dibujar botón izquierdo
        color_left = COLOR_ACCENT if self.is_hovered_left else COLOR_INPUT_BG
        pygame.draw.rect(screen, color_left, self.btn_left, border_radius=4)
        txt_left = font.render("<", True, COLOR_TEXT)
        screen.blit(txt_left, txt_left.get_rect(center=self.btn_left.center))

        # Dibujar botón derecho
        color_right = COLOR_ACCENT if self.is_hovered_right else COLOR_INPUT_BG
        pygame.draw.rect(screen, color_right, self.btn_right, border_radius=4)
        txt_right = font.render(">", True, COLOR_TEXT)
        screen.blit(txt_right, txt_right.get_rect(center=self.btn_right.center))

        # Dibujar opción actual
        opt_text = self.options[self.idx]
        txt_opt = font.render(opt_text, True, COLOR_TEXT)
        screen.blit(txt_opt, txt_opt.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2)))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered_left = self.btn_left.collidepoint(event.pos)
            self.is_hovered_right = self.btn_right.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.btn_left.collidepoint(event.pos):
                    self.idx = (self.idx - 1) % len(self.options)
                    if self.callback:
                        self.callback(self.options[self.idx])
                    return True
                elif self.btn_right.collidepoint(event.pos):
                    self.idx = (self.idx + 1) % len(self.options)
                    if self.callback:
                        self.callback(self.options[self.idx])
                    return True
        return False

class Checkbox(UIElement):
    def __init__(self, x: int, y: int, label: str, checked: bool = False) -> None:
        self.rect = pygame.Rect(x, y, 18, 18)
        self.label = label
        self.checked = checked
        self.is_hovered = False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        # Dibujar caja
        pygame.draw.rect(screen, COLOR_INPUT_BG, self.rect, border_radius=3)
        border_color = COLOR_ACCENT if self.is_hovered else COLOR_BORDER
        pygame.draw.rect(screen, border_color, self.rect, width=1, border_radius=3)
        
        if self.checked:
            # Dibujar marca de check
            pygame.draw.line(screen, COLOR_SUCCESS, (self.rect.x + 4, self.rect.y + 9), (self.rect.x + 8, self.rect.y + 13), width=2)
            pygame.draw.line(screen, COLOR_SUCCESS, (self.rect.x + 8, self.rect.y + 13), (self.rect.x + 14, self.rect.y + 4), width=2)
            
        # Dibujar etiqueta
        lbl_surf = font.render(self.label, True, COLOR_TEXT)
        screen.blit(lbl_surf, (self.rect.right + 8, self.rect.y + 1))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and (self.rect.collidepoint(event.pos) or pygame.Rect(self.rect.x, self.rect.y, 150, 18).collidepoint(event.pos)):
                self.checked = not self.checked
                return True
        return False

class InputBox(UIElement):
    def __init__(self, x: int, y: int, w: int, h: int, label: str, default_text: str = "", is_password: bool = False) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.text = default_text
        self.is_password = is_password
        self.active = False
        self.is_hovered = False
        self.cursor_visible = True
        self.last_cursor_toggle = time.time()

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        # Dibujar etiqueta arriba
        lbl_surf = font.render(self.label, True, COLOR_TEXT_MUTED)
        screen.blit(lbl_surf, (self.rect.x, self.rect.y - 18))
        
        # Dibujar caja de fondo
        pygame.draw.rect(screen, COLOR_INPUT_BG, self.rect, border_radius=4)
        border_color = COLOR_ACCENT if self.active else (COLOR_ACCENT_HOVER if self.is_hovered else COLOR_BORDER)
        pygame.draw.rect(screen, border_color, self.rect, width=1, border_radius=4)
        
        # Renderizar texto
        display_text = "*" * len(self.text) if (self.is_password and not self.active) else self.text
        # Si el texto es muy largo, recortarlo para que quepa
        max_w = self.rect.width - 16
        txt_surf = font.render(display_text, True, COLOR_TEXT)
        while txt_surf.get_width() > max_w and len(display_text) > 0:
            display_text = display_text[1:]
            txt_surf = font.render(display_text, True, COLOR_TEXT)
            
        screen.blit(txt_surf, (self.rect.x + 8, self.rect.y + (self.rect.height - txt_surf.get_height()) // 2))
        
        # Dibujar cursor parpadeante
        if self.active:
            now = time.time()
            if now - self.last_cursor_toggle > 0.5:
                self.cursor_visible = not self.cursor_visible
                self.last_cursor_toggle = now
            if self.cursor_visible:
                cursor_x = self.rect.x + 8 + txt_surf.get_width()
                cursor_y_start = self.rect.y + 6
                cursor_y_end = self.rect.bottom - 6
                pygame.draw.line(screen, COLOR_TEXT, (cursor_x, cursor_y_start), (cursor_x, cursor_y_end), width=1)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.active = self.rect.collidepoint(event.pos)
                return self.active
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                # Pegar desde el portapapeles de forma robusta
                self.text += get_clipboard_text()
            elif event.unicode.isprintable() and len(event.unicode) > 0:
                self.text += event.unicode
            return True
        return False

class PlayModelApp:
    def __init__(self) -> None:
        pygame.init()
        # Inicializar el mixer de pygame con la frecuencia exacta de Arcade Gym (16000 Hz, mono, 16-bit)
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=16000, size=-16, channels=1)
        except Exception:
            pygame.mixer.init()
        pygame.display.set_caption("ARC-AGI / Arcade Gym Model Evaluator GUI")
        
        # Dimensiones de la pantalla
        self.grid_size = 64
        self.cell_scale = 6
        self.grid_px = self.grid_size * self.cell_scale  # 384 px
        
        self.sidebar_width = 896
        self.width = self.grid_px + self.sidebar_width  # 1280 px
        self.height = 840  # Incrementado a 840 para evitar superposiciones
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Fuentes
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_bold = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 22, bold=True)
        
        # Estado de la aplicación
        self.running = True
        self.mode = "idle"  # "idle", "model_playing", "manual_playing"
        self.selected_game_id = "random"  # "random" o el ID del juego seleccionado
        self.current_game_id = "p61_sort"
        self.env: ArcadeSuiteEnv | None = None
        self.obs: dict[str, Any] = {}
        self.info: dict[str, Any] = {}
        self.step_idx = 0
        self.max_steps = 200
        
        # Historial de conversación y memoria
        self.conversation_history = []
        self.last_executed_action_str = "None (Initial State)"
        
        # Razonamiento del modelo
        self.last_model_reasoning = "Ninguna consulta realizada aún. El razonamiento del modelo se mostrará aquí en tiempo real."
        
        # Métricas en vivo
        self.cum_reward = 0.0
        self.no_op_count = 0
        self.total_actions = 0
        self.causal_actions = set()  # Acciones que cambiaron el estado
        self.solved_episodes = 0
        self.total_episodes = 0
        
        # Consola de logs
        self.console_logs: list[str] = ["Iniciado ARC-AGI Model Evaluator GUI.", "Listo para conectar modelo."]
        
        # Elementos de la interfaz de usuario (Sidebar)
        start_y = 40
        self.input_api_key = InputBox(self.grid_px + 20, start_y + 20, 420, 32, "API Key (Pega con Ctrl+V)", "", is_password=True)
        self.input_api_url = InputBox(self.grid_px + 20, start_y + 80, 420, 32, "API Base URL (Ollama, HuggingFace, etc.)", "https://api.openai.com/v1")
        self.input_model = InputBox(self.grid_px + 20, start_y + 140, 420, 32, "Model Name", "gpt-4o-mini")
        
        # Botones de presets rápidos para cambiar de proveedor
        self.btn_preset_openai = Button(self.grid_px + 20, start_y + 180, 95, 24, "OpenAI", self.set_preset_openai, color=COLOR_PANEL)
        self.btn_preset_hf = Button(self.grid_px + 125, start_y + 180, 95, 24, "HuggingFace", self.set_preset_hf, color=COLOR_PANEL)
        self.btn_preset_ollama = Button(self.grid_px + 230, start_y + 180, 95, 24, "Ollama", self.set_preset_ollama, color=COLOR_PANEL)
        self.btn_preset_lmstudio = Button(self.grid_px + 335, start_y + 180, 105, 24, "LM Studio", self.set_preset_lmstudio, color=COLOR_PANEL)

        # Selector de Juegos (Nombres código exactos del paper + random)
        game_options = ["random"] + list(GAME_CLASSES.keys())
        self.selector_game = Selector(self.grid_px + 20, start_y + 235, 420, 32, "Select Game (ARC-AGI / Arcade Gym)", game_options, default_idx=0, callback=self.on_game_selected)

        # Inputs de Max Steps y Conversation History Length (lado a lado)
        self.input_max_steps = InputBox(self.grid_px + 20, start_y + 295, 200, 32, "Max Steps per Episode", "200")
        self.input_history_len = InputBox(self.grid_px + 240, start_y + 295, 200, 32, "Conv. History Length", "5")

        # Checkboxes colocados verticalmente para evitar superposiciones (desplazados hacia abajo para dar espacio al selector y max steps)
        self.chk_vision = Checkbox(self.grid_px + 20, start_y + 345, "Vision Enabled (Sends board image)", checked=True)
        self.chk_sound = Checkbox(self.grid_px + 20, start_y + 370, "Sound Enabled", checked=True)
        
        # Botones de control con más espacio vertical
        self.btn_play = Button(self.grid_px + 20, start_y + 405, 200, 36, "Start Model Play", self.toggle_model_play, color=COLOR_SUCCESS)
        self.btn_manual = Button(self.grid_px + 240, start_y + 405, 200, 36, "Manual Play Mode", self.toggle_manual_play, color=COLOR_ACCENT)
        self.btn_reset = Button(self.grid_px + 20, start_y + 455, 200, 36, "Reset Game", self.reset_env, color=COLOR_DANGER)
        self.btn_next_game = Button(self.grid_px + 240, start_y + 455, 200, 36, "Next Random Game", self.next_game, color=COLOR_PANEL)
        
        self.ui_elements = [
            self.input_api_key, self.input_api_url, self.input_model,
            self.btn_preset_openai, self.btn_preset_hf, self.btn_preset_ollama, self.btn_preset_lmstudio,
            self.selector_game, self.input_max_steps, self.input_history_len,
            self.chk_vision, self.chk_sound,
            self.btn_play, self.btn_manual, self.btn_reset, self.btn_next_game
        ]
        
        # Inicializar el entorno
        self.reset_env()

    def log(self, message: str) -> None:
        self.console_logs.append(message)
        if len(self.console_logs) > 18:
            self.console_logs.pop(0)
        print(f"[GUI Log] {message}")

    def reset_env(self) -> None:
        # Actualizar max_steps dinámicamente desde la interfaz de usuario
        try:
            self.max_steps = int(self.input_max_steps.text.strip())
        except ValueError:
            self.max_steps = 200  # Valor por defecto
            
        # Reiniciar el historial de conversación y memoria para el nuevo episodio
        self.conversation_history = []
        self.last_executed_action_str = "None (Initial State)"
            
        target_game = None if self.selected_game_id == "random" else self.selected_game_id
        self.log(f"Inicializando juego: {self.selected_game_id}...")
        seed = random.randint(0, 999999)
        self.env = ArcadeSuiteEnv(
            seed=seed,
            grid_size=self.grid_size,
        )
        self.obs, self.info = self.env.reset(game_id=target_game)
        self.current_game_id = self.env.current_game_id
        self.step_idx = 0
        self.cum_reward = 0.0
        self.no_op_count = 0
        self.total_actions = 0
        self.causal_actions.clear()
        self.total_episodes += 1
        self.log(f"Juego {self.current_game_id} listo (Semilla: {seed}).")

    def next_game(self) -> None:
        games = list(GAME_CLASSES.keys())
        next_game_id = random.choice(games)
        self.selected_game_id = next_game_id
        if next_game_id in self.selector_game.options:
            self.selector_game.idx = self.selector_game.options.index(next_game_id)
        self.reset_env()

    def on_game_selected(self, game_id: str) -> None:
        self.selected_game_id = game_id
        self.log(f"Juego seleccionado en UI: {game_id}")
        self.reset_env()

    def set_preset_openai(self) -> None:
        self.input_api_url.text = "https://api.openai.com/v1"
        self.input_model.text = "gpt-4o-mini"
        self.chk_vision.checked = True
        self.log("Preset cargado: OpenAI (gpt-4o-mini)")

    def set_preset_hf(self) -> None:
        self.input_api_url.text = "https://api-inference.huggingface.co/v1"
        self.input_model.text = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.chk_vision.checked = True
        self.log("Preset cargado: HuggingFace (Llama-3.2-11B-Vision)")

    def set_preset_ollama(self) -> None:
        self.input_api_url.text = "http://localhost:11434/v1"
        self.input_model.text = "qwen2.5:1.5b"
        self.chk_vision.checked = False
        self.log("Preset cargado: Ollama Local (qwen2.5:1.5b)")

    def set_preset_lmstudio(self) -> None:
        self.input_api_url.text = "http://localhost:1234/v1"
        self.input_model.text = "qwen2.5-1.5b-instruct"
        self.chk_vision.checked = False
        self.log("Preset cargado: LM Studio Local (qwen2.5-1.5b)")

    def toggle_model_play(self) -> None:
        if self.mode == "model_playing":
            self.mode = "idle"
            self.btn_play.text = "Start Model Play"
            self.btn_play.color = COLOR_SUCCESS
            self.log("Modelo pausado.")
        else:
            api_key = self.input_api_key.text.strip()
            # Si es OpenAI y está vacío, advertir
            if "openai" in self.input_api_url.text.lower() and not api_key:
                self.log("ERROR: Se requiere API Key para OpenAI.")
                return
            self.mode = "model_playing"
            self.btn_play.text = "Pause Model Play"
            self.btn_play.color = COLOR_DANGER
            self.log("Modelo jugando activamente...")

    def toggle_manual_play(self) -> None:
        if self.mode == "manual_playing":
            self.mode = "idle"
            self.btn_manual.text = "Manual Play Mode"
            self.btn_manual.color = COLOR_ACCENT
            self.log("Modo manual desactivado.")
        else:
            self.mode = "manual_playing"
            self.btn_manual.text = "Exit Manual Mode"
            self.btn_manual.color = COLOR_DANGER
            self.log("Modo manual activado. Usa flechas, Espacio (A5), Q (A6), W (A7), E (A8).")

    def execute_action(self, action_id: int) -> float:
        if not self.env:
            return 0.0
            
        action_name = ACTION_LABELS.get(action_id, "wait")
        
        # Si la acción es localizada (A6), poner coordenadas del centro del agente o del grid
        action_data = None
        if action_id == 6:
            ax = self.info.get("agent_x", self.grid_size // 2)
            ay = self.info.get("agent_y", self.grid_size // 2)
            action_data = {"x": int(ax), "y": int(ay)}
            
        # Paso en el entorno
        res = self.env.step(action_id, action_data=action_data)
        self.obs = res.observation
        self.info = res.info
        
        # Sonido: Reproducir el sonido sintético real generado por el motor del juego en tiempo real
        if self.chk_sound.checked and res.audio_event and res.audio_event.kind != "silence":
            try:
                # Convertir la onda float32 [-1.0, 1.0] a PCM16 de 16 bits
                waveform_int16 = (res.audio_event.waveform * 32767).astype(np.int16)
                sound = pygame.mixer.Sound(buffer=waveform_int16.tobytes())
                sound.play()
            except Exception:
                pass
        
        # Calcular métricas
        self.total_actions += 1
        self.step_idx += 1
        self.cum_reward += res.reward
        
        is_no_op = self.info.get("no_op", False)
        if is_no_op:
            self.no_op_count += 1
        else:
            self.causal_actions.add(action_id)
            
        # Leer max_steps dinámicamente desde la interfaz de usuario en tiempo real
        try:
            self.max_steps = int(self.input_max_steps.text.strip())
        except ValueError:
            pass
            
        # Truncar el episodio si se alcanza el límite de pasos definido por el usuario
        is_truncated = res.truncated or (self.step_idx >= self.max_steps)
            
        if res.terminated or is_truncated:
            if res.reward > 0.5 or self.info.get("goal_reached", False):
                self.solved_episodes += 1
                self.log("¡EPISODIO COMPLETADO CON ÉXITO!")
            else:
                if is_truncated:
                    self.log(f"Episodio truncado por límite de pasos ({self.max_steps}).")
                else:
                    self.log("Episodio terminado.")
            self.reset_env()
            
        return res.reward

    def get_grid_image_base64(self) -> str:
        # Renderizar el grid a una superficie pequeña de Pygame
        surf = pygame.Surface((self.grid_size, self.grid_size))
        frame = np.asarray(self.obs.get("frame", np.zeros((self.grid_size, self.grid_size))))
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color_idx = int(frame[y, x]) % len(PALETTE_RGB)
                surf.set_at((x, y), PALETTE_RGB[color_idx])
                
        # Guardar a PNG en memoria
        buffer = BytesIO()
        pygame.image.save(surf, buffer, "png")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def query_model_action(self) -> int:
        api_url = self.input_api_url.text.strip()
        api_key = self.input_api_key.text.strip()
        model_name = self.input_model.text.strip()
        vision_enabled = self.chk_vision.checked
        
        # Obtener el límite de historial de conversación elegido por el usuario
        try:
            history_len = int(self.input_history_len.text.strip())
        except ValueError:
            history_len = 5
            
        self.log("Consultando al modelo...")
        
        # Construir prompt pidiendo razonamiento explícito
        system_prompt = (
            "You are an AI agent playing an ARC-style grid game. "
            "Your objective is to solve the puzzle by executing actions. "
            "The available actions are:\n"
            "A1: Move Up\n"
            "A2: Move Down\n"
            "A3: Move Left\n"
            "A4: Move Right\n"
            "A5: Primary Action (e.g. interact, launch)\n"
            "A6: Localized Action (interact at agent position)\n"
            "A7: Secondary Action\n"
            "A8: Wait / No-Op\n\n"
            "First, analyze the current board state, describe your observations, and explain your step-by-step reasoning. "
            "Then, at the very end of your response, output your chosen action sequence inside square brackets, "
            "separated by commas. Example: [A1, A3, A5]."
        )
        
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        # Construir la nueva entrada del usuario con la observación actual y la última acción ejecutada
        if vision_enabled:
            # Obtener imagen base64
            img_b64 = self.get_grid_image_base64()
            user_content = [
                {"type": "text", "text": f"Last executed action: {self.last_executed_action_str}. Current game: {self.current_game_id}. Step: {self.step_idx}. Cumulative Reward: {self.cum_reward:.2f}. Output your reasoning and next action sequence:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                }
            ]
        else:
            # Grid como texto plano
            frame = self.obs.get("frame", np.zeros((self.grid_size, self.grid_size)))
            grid_str = "\n".join(" ".join(str(int(cell)) for cell in row) for row in frame)
            user_content = f"Last executed action: {self.last_executed_action_str}. Current game: {self.current_game_id}. Step: {self.step_idx}. Cumulative Reward: {self.cum_reward:.2f}.\nGrid Board:\n{grid_str}\n\nOutput your reasoning and next action sequence:"
            
        # Guardar la entrada en el historial de conversación
        self.conversation_history.append({"role": "user", "content": user_content})
        
        # Gestor inteligente de memoria: acotar el historial al límite elegido (multiplicado por 2 para incluir pares User/Assistant)
        max_history_entries = history_len * 2
        if len(self.conversation_history) > max_history_entries:
            self.conversation_history = self.conversation_history[-max_history_entries:]
            
        # Construir el payload de mensajes garantizando que el system prompt sea siempre la primera entrada
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
        
        # Payload inicial con parámetros estándar
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 400
        }
        
        try:
            # Endpoint de chat completions
            url = f"{api_url}/chat/completions"
            
            # Reintentos dinámicos autocurativos para máxima compatibilidad universal (OpenAI o-series, DeepSeek, Claude, etc.)
            response = None
            for attempt in range(3):
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    break
                elif response.status_code == 400:
                    err_msg = response.text.lower()
                    adjusted = False
                    
                    # Si el modelo no soporta max_tokens (ej. OpenAI o1/o3/o4), cambiamos a max_completion_tokens
                    if "max_tokens" in err_msg and "max_tokens" in payload:
                        del payload["max_tokens"]
                        payload["max_completion_tokens"] = 2048  # Aumentado a 2048 para dar suficiente presupuesto a los tokens de razonamiento internos
                        adjusted = True
                        self.log("Auto-ajuste: 'max_tokens' no soportado. Cambiando a 'max_completion_tokens'...")
                        
                    # Si el modelo no soporta temperature (ej. modelos de razonamiento), lo eliminamos
                    if "temperature" in err_msg and "temperature" in payload:
                        del payload["temperature"]
                        adjusted = True
                        self.log("Auto-ajuste: 'temperature' no soportada. Eliminando parámetro...")
                        
                    if adjusted:
                        continue
                    else:
                        break
                else:
                    break
                    
            if response is None or response.status_code != 200:
                err_text = response.text[:100] if response is not None else "No response"
                status = response.status_code if response is not None else "None"
                self.log(f"API Error {status}: {err_text}")
                return 8 # Wait on error
                
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"].strip()
            
            # Guardar la respuesta del asistente en el historial de conversación
            self.conversation_history.append({"role": "assistant", "content": content})
            
            # Guardar el razonamiento completo para mostrarlo en el panel dedicado
            self.last_model_reasoning = content
            
            # Extraer secuencia de acciones dentro de corchetes de forma robusta
            import re
            match = re.search(r'\[\s*(A[1-8](?:\s*,\s*A[1-8])*)\s*\]', content)
            if match:
                action_block = match.group(1)
                actions = re.findall(r'A[1-8]', action_block)
            else:
                actions = re.findall(r'A[1-8]', content)
                
            if actions:
                # Ejecutar la primera acción encontrada
                action_str = actions[0]
                action_id = int(action_str[1])
                self.last_executed_action_str = f"{action_str} ({ACTION_LABELS[action_id]})"
                self.log(f"Model Response: {actions}")
                self.log(f"Executing Action: {action_str} ({ACTION_LABELS[action_id]})")
                return action_id
            else:
                self.last_executed_action_str = "A8 (Wait / No-Op)"
                self.log("No valid action format found, defaulting to A8 (Wait)")
                # Mostrar los primeros 80 caracteres de la respuesta para depurar por qué no se encontraron acciones
                self.log(f"Raw response preview: {content[:80]}...")
                return 8
        except Exception as e:
            self.last_executed_action_str = "A8 (Wait / No-Op)"
            self.log(f"Network/API Error: {str(e)[:80]}")
            return 8

    def draw(self) -> None:
        self.screen.fill(COLOR_BG)
        
        # 1. Dibujar el Grid de Juego (Izquierda)
        grid_rect = pygame.Rect(10, 10, self.grid_px, self.grid_px)
        pygame.draw.rect(self.screen, (10, 10, 10), grid_rect)
        pygame.draw.rect(self.screen, COLOR_BORDER, grid_rect, width=2)
        
        if self.env and "frame" in self.obs:
            frame = np.asarray(self.obs["frame"])
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    color_idx = int(frame[y, x]) % len(PALETTE_RGB)
                    rect = pygame.Rect(
                        10 + x * self.cell_scale,
                        10 + y * self.cell_scale,
                        self.cell_scale,
                        self.cell_scale
                    )
                    pygame.draw.rect(self.screen, PALETTE_RGB[color_idx], rect)
                    
        # 2. Dibujar Dashboard de Métricas en Vivo (Abajo del Grid)
        metrics_y = self.grid_px + 30
        pygame.draw.rect(self.screen, COLOR_PANEL, (10, metrics_y, self.grid_px, self.height - metrics_y - 10), border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, (10, metrics_y, self.grid_px, self.height - metrics_y - 10), width=1, border_radius=6)
        
        # Título métricas
        lbl = self.font_title.render("LIVE PERFORMANCE METRICS", True, COLOR_TEXT)
        self.screen.blit(lbl, (25, metrics_y + 15))
        
        # Valores de métricas
        no_op_rate = (self.no_op_count / max(1, self.total_actions)) * 100.0
        causal_cov = (len(self.causal_actions) / 8.0) * 100.0
        solved_rate = (self.solved_episodes / max(1, self.total_episodes)) * 100.0
        
        # Leer max_steps dinámicamente para que la UI se actualice en tiempo real al escribir
        try:
            self.max_steps = int(self.input_max_steps.text.strip())
        except ValueError:
            pass
            
        metrics = [
            ("Current Game:", f"{self.current_game_id.upper()}", COLOR_TEXT),
            ("Step / Max Steps:", f"{self.step_idx} / {self.max_steps}", COLOR_TEXT),
            ("Cumulative Reward:", f"{self.cum_reward:.4f}", COLOR_SUCCESS if self.cum_reward >= 0 else COLOR_DANGER),
            ("Motor Passivity (No-Op):", f"{no_op_rate:.2f}%", COLOR_DANGER if no_op_rate > 50 else COLOR_SUCCESS),
            ("Causal Coverage (Explor.):", f"{causal_cov:.2f}%", COLOR_SUCCESS if causal_cov > 50 else COLOR_TEXT),
            ("Solved Rate (Completion):", f"{solved_rate:.2f}% ({self.solved_episodes}/{self.total_episodes})", COLOR_SUCCESS)
        ]
        
        for idx, (name, val, color) in enumerate(metrics):
            y_pos = metrics_y + 55 + idx * 30
            # Nombre métrica
            lbl_name = self.font_bold.render(name, True, COLOR_TEXT_MUTED)
            self.screen.blit(lbl_name, (25, y_pos))
            # Valor métrica
            lbl_val = self.font_large.render(val, True, color) if "Solved" in name or "Reward" in name else self.font_bold.render(val, True, color)
            self.screen.blit(lbl_val, (290, y_pos))

        # 3. Dibujar Sidebar de Configuración (Medio)
        settings_rect = pygame.Rect(self.grid_px + 15, 10, 430, self.height - 20)
        pygame.draw.rect(self.screen, COLOR_PANEL, settings_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, settings_rect, width=1, border_radius=6)
        
        # Título Sidebar
        title_surf = self.font_title.render("MODEL CONNECTION SETTINGS", True, (255, 255, 255))
        self.screen.blit(title_surf, (self.grid_px + 25, 25))
        
        # Dibujar todos los elementos interactivos de la UI
        for elem in self.ui_elements:
            if isinstance(elem, (InputBox, Button, Checkbox, Selector)):
                elem.draw(self.screen, self.font)
                
        # 4. Consola de Logs en vivo (Abajo en la Sidebar de Configuración)
        console_y = 545
        console_rect = pygame.Rect(self.grid_px + 20, console_y, 420, 265)
        pygame.draw.rect(self.screen, COLOR_INPUT_BG, console_rect, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_BORDER, console_rect, width=1, border_radius=4)
        
        # Título Consola
        con_title = self.font_bold.render("LIVE CONSOLE LOGS", True, COLOR_ACCENT)
        self.screen.blit(con_title, (self.grid_px + 25, console_y - 20))
        
        # Dibujar líneas de log
        for idx, log_line in enumerate(self.console_logs[-13:]):  # Mostrar las últimas 13 líneas para que quepan en 265px de alto
            log_surf = self.font.render(log_line, True, COLOR_TEXT if "Executing" in log_line or "Response" in log_line else COLOR_TEXT_MUTED)
            self.screen.blit(log_surf, (self.grid_px + 28, console_y + 8 + idx * 18))

        # 5. Dibujar Panel de Razonamiento del Modelo (Derecha)
        reasoning_rect = pygame.Rect(self.grid_px + 460, 10, 410, self.height - 20)
        pygame.draw.rect(self.screen, COLOR_PANEL, reasoning_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, reasoning_rect, width=1, border_radius=6)
        
        # Título Panel Razonamiento
        reas_title = self.font_title.render("MODEL REASONING & THOUGHTS", True, COLOR_SUCCESS)
        self.screen.blit(reas_title, (self.grid_px + 475, 25))
        
        # Caja de texto para el razonamiento
        text_box_rect = pygame.Rect(self.grid_px + 475, 55, 380, self.height - 85)
        pygame.draw.rect(self.screen, COLOR_INPUT_BG, text_box_rect, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_BORDER, text_box_rect, width=1, border_radius=4)
        
        # Dibujar el texto del razonamiento con ajuste de línea (wrap)
        def draw_wrapped_text(surface, text, rect, font, color):
            words = text.split(' ')
            lines = []
            current_line = []
            for word in words:
                if '\n' in word:
                    sub_words = word.split('\n')
                    for i, sw in enumerate(sub_words):
                        if i > 0:
                            lines.append(' '.join(current_line))
                            current_line = []
                        if sw:
                            current_line.append(sw)
                else:
                    test_line = ' '.join(current_line + [word])
                    if font.size(test_line)[0] < rect.width - 16:
                        current_line.append(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
                
            y = rect.y + 8
            for line in lines:
                if y + font.get_linesize() > rect.bottom - 8:
                    break
                surf = font.render(line, True, color)
                surface.blit(surf, (rect.x + 8, y))
                y += font.get_linesize() + 2
                
        draw_wrapped_text(self.screen, self.last_model_reasoning, text_box_rect, self.font, COLOR_TEXT)

        pygame.display.flip()

    def run(self) -> None:
        last_model_query_time = 0
        
        while self.running:
            # Manejar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                # Pasar eventos a elementos de UI
                for elem in self.ui_elements:
                    elem.handle_event(event)
                    
                # Manejar controles manuales si está en modo manual
                if self.mode == "manual_playing" and event.type == pygame.KEYDOWN:
                    action_id = None
                    if event.key == pygame.K_UP:
                        action_id = 1
                    elif event.key == pygame.K_DOWN:
                        action_id = 2
                    elif event.key == pygame.K_LEFT:
                        action_id = 3
                    elif event.key == pygame.K_RIGHT:
                        action_id = 4
                    elif event.key == pygame.K_SPACE:
                        action_id = 5
                    elif event.key == pygame.K_q:
                        action_id = 6
                    elif event.key == pygame.K_w:
                        action_id = 7
                    elif event.key == pygame.K_e:
                        action_id = 8
                        
                    if action_id is not None:
                        reward = self.execute_action(action_id)
                        self.log(f"Manual Action: A{action_id} ({ACTION_LABELS[action_id]}). Reward: {reward:.4f}")

            # Lógica de juego del modelo
            if self.mode == "model_playing":
                now = time.time()
                # Consultar al modelo cada 1.5 segundos para que sea visible y no sature API
                if now - last_model_query_time > 1.5:
                    action_id = self.query_model_action()
                    reward = self.execute_action(action_id)
                    last_model_query_time = now

            # Dibujar pantalla
            self.draw()
            self.clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    app = PlayModelApp()
    app.run()
