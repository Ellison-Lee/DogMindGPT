#!/usr/bin/env python3
"""
使用 Ollama API 进行对话的脚本
调用模型: NLP-Final/models/merged-qwen3-1.7B-gguf/ggml-model-Q4_K_M.gguf

注意：使用前需要先将 GGUF 模型导入到 Ollama, 并启动llama serve
运行的cmd
  ollama create qwen3-lora-local -f Modelfile
  llama serve
"""

import requests
import json
import os
import sys
import re

# 配置参数
MODEL_NAME = "qwen3-lora-local:latest"  # Ollama 中的模型名称
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API 端点

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
PROMPT_PATH = os.path.join(SCRIPT_DIR, "dog_prompt.txt")  # 系统提示词文件路径


def load_system_prompt(prompt_path):
    """加载系统提示词"""
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def filter_thinking_tags(text):
    """过滤掉思考过程的标记"""
    if not text:
        return text
    
    # 移除常见的标记
    patterns = [
        r'<think>.*?</think>',  
        r'<think>.*?</think>',  
        r'<think>.*?</think>', 
        r'<thinking>.*?</thinking>',  
        r'<reasoning>.*?</reasoning>', 
        r'<thought>.*?</thought>',  
        r'<think>', 
        r'</think>',
        r'<think>',
        r'</think>',
        r'```',
    ]
    
    filtered_text = text
    for pattern in patterns:
        filtered_text = re.sub(pattern, '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理多余的空行
    filtered_text = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_text)
    
    return filtered_text.strip()


def chat_with_ollama(question, system_prompt=None, model_name=MODEL_NAME):
    """使用 Ollama /api/generate 端点进行对话"""
    payload = {
        "model": model_name,
        "prompt": question,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        }
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "response" in result:
            response_text = result["response"].strip()
            # 过滤掉思考过程的标记
            return filter_thinking_tags(response_text)
        return None
    except Exception:
        return None


def main():
    # 检查服务
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except:
        sys.exit(1)
    
    # 检查模型
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m.get("name", "").strip() for m in response.json().get("models", [])]
        if MODEL_NAME.strip() not in models:
            sys.exit(1)
    except:
        sys.exit(1)
    
    # 加载系统提示词
    system_prompt = load_system_prompt(PROMPT_PATH)
    
    # 交互式对话
    print("=" * 70)
    print("Ollama API 对话工具")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_question = input("请输入问题: ").strip()
            
            if user_question.lower() in ['quit', 'exit', '退出', 'q']:
                break
            
            if not user_question:
                continue
            
            response = chat_with_ollama(
                question=user_question,
                system_prompt=system_prompt,
                model_name=MODEL_NAME.strip()
            )
            
            if response:
                print("\n" + "-" * 70)
                print("回答:")
                print(response)
                print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
