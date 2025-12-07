#!/usr/bin/env python3
"""
使用 ollama 下载 qwen3:1.7b 模型
"""

import subprocess
import sys


def check_ollama_installed():
    """检查 ollama 是否已安装"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Ollama 已安装: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("错误: 未找到 ollama 命令，请先安装 ollama")
        print("安装方法: https://ollama.ai")
        return False
    except subprocess.CalledProcessError as e:
        print(f"错误: ollama 命令执行失败: {e}")
        return False


def download_model(model_name="qwen3:1.7b"):
    """下载指定的模型"""
    print(f"开始下载模型: {model_name}")
    print("这可能需要一些时间，请耐心等待...")
    
    try:
        # 使用 subprocess.run 实时显示输出
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时打印输出
        for line in process.stdout:
            print(line, end="")
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n模型 {model_name} 下载成功!")
            return True
        else:
            print(f"\n模型 {model_name} 下载失败，返回码: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"下载过程中发生错误: {e}")
        return False


def verify_model(model_name="qwen3:1.7b"):
    """验证模型是否已下载"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if model_name in result.stdout:
            print(f"验证成功: 模型 {model_name} 已存在于本地")
            return True
        else:
            print(f"警告: 模型 {model_name} 未在列表中")
            return False
            
    except Exception as e:
        print(f"验证模型时发生错误: {e}")
        return False


def main():
    """主函数"""
    model_name = "qwen2.5:1.5b"
    
    print("=" * 50)
    print("Ollama 模型下载工具")
    print("=" * 50)
    
    # 检查 ollama 是否安装
    if not check_ollama_installed():
        sys.exit(1)
    
    # 检查模型是否已存在
    if verify_model(model_name):
        print(f"\n模型 {model_name} 已存在，跳过下载")
        response = input("是否重新下载? (y/n): ")
        if response.lower() != 'y':
            print("取消下载")
            return
    
    # 下载模型
    if download_model(model_name):
        # 再次验证
        verify_model(model_name)
        print("\n下载完成!")
    else:
        print("\n下载失败，请检查网络连接和模型名称")
        sys.exit(1)


if __name__ == "__main__":
    main()

