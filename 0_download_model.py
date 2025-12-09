#!/usr/bin/env python3
"""
从 Hugging Face 下载 Qwen3-1.7B 模型
"""

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("错误: 未安装 PyTorch，请先安装:")
        print("  pip install torch")
        return False
    
    try:
        import transformers
        print(f"Transformers 版本: {transformers.__version__}")
    except ImportError:
        print("错误: 未安装 transformers，请先安装:")
        print("  pip install transformers")
        return False
    
    try:
        import huggingface_hub
        print(f"Hugging Face Hub 版本: {huggingface_hub.__version__}")
    except ImportError:
        print("错误: 未安装 huggingface_hub，请先安装:")
        print("  pip install huggingface_hub")
        return False
    
    return True


def check_model_exists(model_path):
    """检查模型是否已下载"""
    if os.path.exists(model_path):
        # 检查关键文件是否存在
        required_files = ['config.json', 'tokenizer.json']
        all_exist = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
        
        # 检查是否有模型文件（可能是 safetensors 或 bin 文件）
        has_model_file = any(
            f.endswith('.safetensors') or f.endswith('.bin') 
            for f in os.listdir(model_path) 
            if os.path.isfile(os.path.join(model_path, f))
        )
        
        if all_exist and has_model_file:
            return True
    
    return False


def download_model_from_hf(model_name="Qwen/Qwen3-1.7B", output_dir="models/Qwen3-1.7B"):
    """
    从 Hugging Face 下载模型
    
    Args:
        model_name: Hugging Face 模型名称
        output_dir: 本地保存目录
    """
    print(f"开始从 Hugging Face 下载模型: {model_name}")
    print(f"保存路径: {os.path.abspath(output_dir)}")
    print("这可能需要一些时间，请耐心等待...")
    print("=" * 70)
    
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 方法1: 使用 snapshot_download（更可靠，支持断点续传）
        print("\n[方法1] 使用 snapshot_download 下载...")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"\n✓ 模型文件下载完成!")
        except Exception as e:
            print(f"snapshot_download 失败: {e}")
            print("尝试使用 AutoModelForCausalLM 下载...")
            
            # 方法2: 使用 AutoModelForCausalLM（备用方法）
            print("\n[方法2] 使用 AutoModelForCausalLM 下载...")
            print("正在下载 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=None  # 使用默认缓存目录
            )
            tokenizer.save_pretrained(output_dir)
            print("✓ Tokenizer 下载完成")
            
            print("正在下载模型（这可能需要较长时间）...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                cache_dir=None
            )
            model.save_pretrained(output_dir)
            print("✓ 模型下载完成")
        
        # 验证下载的文件
        print("\n验证下载的文件...")
        if check_model_exists(output_dir):
            print("✓ 模型验证成功!")
            return True
        else:
            print("⚠ 警告: 模型文件可能不完整")
            return False
            
    except Exception as e:
        print(f"\n✗ 下载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_model(model_path="models/Qwen3-1.7B"):
    """验证模型是否可以正常加载"""
    print("\n" + "=" * 70)
    print("验证模型是否可以正常加载...")
    print("=" * 70)
    
    try:
        print("加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✓ Tokenizer 加载成功")
        
        print("加载模型配置（不加载完整模型以节省时间）...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ 模型配置加载成功")
        print(f"  模型类型: {config.model_type}")
        print(f"  参数量: {getattr(config, 'vocab_size', 'N/A')} 词汇表大小")
        
        print("\n✓ 模型验证完成，可以正常使用!")
        return True
        
    except Exception as e:
        print(f"\n✗ 模型验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    model_name = "Qwen/Qwen3-1.7B"
    output_dir = "models/Qwen3-1.7B"
    
    print("=" * 70)
    print("Hugging Face 模型下载工具")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"保存路径: {os.path.abspath(output_dir)}")
    print("=" * 70)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查模型是否已存在
    if check_model_exists(output_dir):
        print(f"\n模型已存在于: {output_dir}")
        response = input("是否重新下载? (y/n): ")
        if response.lower() != 'y':
            print("跳过下载，验证现有模型...")
            if verify_model(output_dir):
                print("\n现有模型验证通过!")
                return
            else:
                print("\n现有模型验证失败，建议重新下载")
                response = input("是否现在下载? (y/n): ")
                if response.lower() != 'y':
                    print("取消操作")
                    return
        else:
            # 删除现有模型目录
            import shutil
            if os.path.exists(output_dir):
                print(f"\n删除现有模型目录: {output_dir}")
                shutil.rmtree(output_dir)
    
    # 下载模型
    if download_model_from_hf(model_name, output_dir):
        # 验证模型
        if verify_model(output_dir):
            print("\n" + "=" * 70)
            print("✓ 下载和验证完成!")
            print(f"模型已保存到: {os.path.abspath(output_dir)}")
            print("=" * 70)
        else:
            print("\n⚠ 警告: 下载完成但验证失败，请检查模型文件")
            sys.exit(1)
    else:
        print("\n✗ 下载失败，请检查网络连接和模型名称")
        sys.exit(1)


if __name__ == "__main__":
    main()
