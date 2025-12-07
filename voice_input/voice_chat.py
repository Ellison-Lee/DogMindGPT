#!/usr/bin/env python3
"""
语音输入 + LLM 对话整合脚本
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sounddevice as sd
import numpy as np
import queue
import time
import requests
import re
import sys
from faster_whisper import WhisperModel

# 音频设备配置
MIC_INDEX = None          # 麦克风设备ID，None表示自动检测
MIC_RATE = 48000          # 麦克风采样率
MIC_CHANNELS = None       # 麦克风通道数，None表示自动检测

# Whisper 模型配置
MODEL_SIZE = "small"      # Whisper模型大小 (tiny/small/medium/large)
DEVICE = "cpu"            # 运行设备 (cpu/cuda)
COMPUTE_TYPE = "int8"     # 计算精度 (int8/float16/float32)

# 语音活动检测 (VAD) 配置
VAD_THRESHOLD = 0.01     # 触发录音的音量阈值
SILENCE_DURATION = 1.0    # 说话停顿时间(秒)，超过此时间后开始识别

# LLM 配置
MODEL_NAME = "qwen3-lora-local:latest"  # Ollama模型名称
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API地址

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
PROMPT_PATH = os.path.join(PARENT_DIR, "dog_prompt.txt")  # 系统提示词文件路径

q = queue.Queue()


def get_audio_device():
    devices = sd.query_devices()
    input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    if not input_devices:
        raise RuntimeError("未找到可用的音频输入设备")
    
    if MIC_INDEX is not None and MIC_INDEX < len(devices):
        device_info = devices[MIC_INDEX]
        if device_info['max_input_channels'] > 0:
            return MIC_INDEX, int(device_info['default_samplerate']), device_info['max_input_channels']
    
    device_id, device_info = input_devices[0]
    return device_id, int(device_info['default_samplerate']), device_info['max_input_channels']


def process_audio_chunk(indata):
    data_float = indata.astype(np.float32) / 32768.0
    if len(data_float.shape) > 1 and data_float.shape[1] > 1:
        mono_data = data_float.mean(axis=1)
    else:
        mono_data = data_float.flatten()
    return mono_data[::3]


def callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}", flush=True)
    q.put(indata.copy())


def load_system_prompt(prompt_path):
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def filter_thinking_tags(text):
    if not text:
        return text
    patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reasoning>.*?</reasoning>',
        r'<thought>.*?</thought>',
        r'```',
    ]
    filtered_text = text
    for pattern in patterns:
        filtered_text = re.sub(pattern, '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
    filtered_text = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_text)
    return filtered_text.strip()


def chat_with_ollama(question, system_prompt=None, model_name=MODEL_NAME):
    payload = {
        "model": model_name,
        "prompt": question,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "num_predict": 512}
    }
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "response" in result:
            return filter_thinking_tags(result["response"].strip())
    except Exception:
        pass
    return None


def check_ollama_service():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = [m.get("name", "").strip() for m in response.json().get("models", [])]
        return MODEL_NAME.strip() in models
    except Exception:
        return False


def main():
    if not check_ollama_service():
        print("错误: 无法连接到 Ollama 服务或模型不存在")
        return
    
    system_prompt = load_system_prompt(PROMPT_PATH)
    
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    try:
        device_id, device_rate, device_channels = get_audio_device()
    except Exception as e:
        print(f"音频设备检测失败: {e}")
        return

    mic_id = device_id if MIC_INDEX is None else MIC_INDEX
    mic_rate = device_rate if MIC_RATE is None else MIC_RATE
    mic_channels = device_channels if MIC_CHANNELS is None else MIC_CHANNELS

    print(f"正在监听 (设备ID={mic_id}, 采样率={mic_rate}, 通道数={mic_channels})")
    print("请说话，按 Ctrl+C 退出\n")

    try:
        with sd.InputStream(device=mic_id, samplerate=mic_rate, channels=mic_channels,
                            blocksize=4096, dtype='int16', callback=callback):
            audio_buffer = []
            recording = False
            last_speech_time = time.time()
           
            while True:
                raw_chunk = q.get()
                processed_chunk = process_audio_chunk(raw_chunk)
                volume = np.sqrt(np.mean(processed_chunk**2))
               
                if volume > VAD_THRESHOLD:
                    if not recording:
                        print("\n检测到语音，开始记录...", end="", flush=True)
                        recording = True
                    last_speech_time = time.time()
                    audio_buffer.append(processed_chunk)
                    print(".", end="", flush=True)
                elif recording:
                    audio_buffer.append(processed_chunk)
                    if time.time() - last_speech_time > SILENCE_DURATION:
                        print("\n说话结束，正在识别...", end="", flush=True)
                        recording = False
                       
                        if audio_buffer:
                            final_audio = np.concatenate(audio_buffer)
                            start_t = time.time()
                            segments, _ = model.transcribe(final_audio, beam_size=2, language="zh")
                            text = "".join([s.text for s in segments])
                            end_t = time.time()
                           
                            print(f"\n识别结果 ({end_t - start_t:.2f}s): {text}")
                           
                            if text.strip():
                                llm_response = chat_with_ollama(text, system_prompt, MODEL_NAME.strip())
                                if llm_response:
                                    print(f"\n{'=' * 70}")
                                    print("LLM 回答:")
                                    print(llm_response)
                                    print(f"{'=' * 70}\n")
                       
                        audio_buffer = []
    except Exception as e:
        print(f"\n音频流打开失败: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出程序")
        sys.exit(0)

