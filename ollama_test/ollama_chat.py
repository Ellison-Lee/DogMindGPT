#!/usr/bin/env python3
"""
使用 ollama 进行交互式对话
读取 dog_prompt.txt 作为系统提示词，与用户进行交互
"""

import subprocess
import json
import os
import sys
from pathlib import Path


def check_ollama_serve():
    """检查 ollama serve 是否在运行"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_ollama_serve():
    """启动 ollama serve（后台运行）"""
    print("检查 ollama serve 状态...")
    
    if check_ollama_serve():
        print("ollama serve 已在运行")
        return True
    
    print("启动 ollama serve...")
    try:
        # 在后台启动 ollama serve
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 等待服务启动
        import time
        for i in range(10):
            time.sleep(1)
            if check_ollama_serve():
                print("ollama serve 启动成功")
                return True
        
        print("警告: ollama serve 启动超时，但将继续尝试")
        return False
        
    except Exception as e:
        print(f"启动 ollama serve 时发生错误: {e}")
        return False


def list_models():
    """列出所有已安装的模型"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("\n已安装的模型:")
        print("-" * 50)
        print(result.stdout)
        print("-" * 50)
        
        # 提取模型路径信息
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            print("\n模型文件位置:")
            print("默认位置: ~/.ollama/models/")
            print("(具体路径取决于 ollama 配置)")
        
        return True
        
    except Exception as e:
        print(f"列出模型时发生错误: {e}")
        return False


def load_system_prompt(prompt_file="dog_prompt.txt"):
    """加载系统提示词"""
    script_dir = Path(__file__).parent
    prompt_path = script_dir / prompt_file
    
    if not prompt_path.exists():
        print(f"警告: 未找到提示词文件 {prompt_path}")
        print("将使用默认提示词")
        return "你是一个有用的AI助手。"
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        print(f"已加载提示词文件: {prompt_path}")
        return prompt
    except Exception as e:
        print(f"读取提示词文件时发生错误: {e}")
        return "你是一个有用的AI助手。"


def chat_with_model(model_name, prompt):
    """与模型进行对话"""
    try:
        # 调用 ollama run 命令，直接传入提示词
        process = subprocess.run(
            ["ollama", "run",model_name, prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if process.returncode == 0:
            return process.stdout.strip()
        else:
            return f"错误: {process.stderr}"
            
    except subprocess.TimeoutExpired:
        return "错误: 请求超时"
    except Exception as e:
        return f"错误: {e}"


def interactive_loop(model_name="qwen3:1.7b"):
    """交互式对话循环"""
    system_prompt = load_system_prompt()
    
    print("\n" + "=" * 50)
    print("交互式对话已启动")
    print("=" * 50)
    print(f"使用模型: {model_name}")
    print("提示: 输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见!")
                break
            
            # 构建上下文，只包含系统提示词和当前用户输入
            context = f"{system_prompt}\n\n当前用户问题: {user_input}\n\n请回答:"
            
            response = chat_with_model(model_name, context)
            
            print(f"\n助手: {response}")
            
        except KeyboardInterrupt:
            print("\n\n中断对话，再见!")
            break
        except Exception as e:
            print(f"发生错误: {e}")


def main():
    """主函数"""
    model_name = "qwen3-lora-local:latest "
    
    print("=" * 50)
    print("Ollama 交互式对话工具")
    print("=" * 50)
    
    # 检查并启动 ollama serve
    if not start_ollama_serve():
        print("警告: 无法确认 ollama serve 状态，但将继续尝试")
    
    # 列出模型
    if not list_models():
        print("无法列出模型，请检查 ollama 是否正确安装")
        sys.exit(1)
    
    # 检查模型是否存在
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if model_name not in result.stdout:
            print(f"\n警告: 模型 {model_name} 未找到")
            print("请先运行 download_model.py 下载模型")
            response = input("是否继续尝试? (y/n): ")
            if response.lower() != 'y':
                print("退出")
                sys.exit(1)
    except Exception as e:
        print(f"检查模型时发生错误: {e}")
        sys.exit(1)
    
    # 开始交互式对话
    interactive_loop(model_name)


if __name__ == "__main__":
    main()

