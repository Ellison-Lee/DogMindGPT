import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json

def load_model_with_4bit(model_path):
    """加载模型并使用 4bit 量化（如果支持）或使用 float16/bfloat16"""
    print(f"正在加载模型: {model_path}")
    
    # 检查是否有 CUDA 支持
    has_cuda = torch.cuda.is_available()
    print(f"CUDA 可用: {has_cuda}")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 尝试使用 4bit 量化（仅在 CUDA 可用时）
    if has_cuda:
        try:
            # 配置 4bit 量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,  # 嵌套量化（double quantization）
                bnb_4bit_quant_type="nf4",  # 使用 NF4 量化类型
                bnb_4bit_compute_dtype=torch.bfloat16  # 计算时使用 bfloat16
            )
            
            # 加载模型（使用 4bit 量化）
            print("加载模型（4bit 量化）...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            print("模型加载完成（4bit 量化）！")
            return model, tokenizer
        except Exception as e:
            print(f"4bit 量化失败: {e}")
            print("回退到 float16...")
    
    # 如果没有 CUDA 或 4bit 量化失败，使用 float16/bfloat16
    print("加载模型（float16，CPU/Apple Silicon 优化）...")
    # 对于 Apple Silicon，使用 bfloat16 或 float16
    if torch.backends.mps.is_available():
        dtype = torch.float16  # MPS 支持 float16
        device_map = "mps"
    else:
        dtype = torch.float16  # CPU 也使用 float16 以减少内存
        device_map = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"模型加载完成（{dtype}，设备: {device_map}）！")
    return model, tokenizer

def load_system_prompt(prompt_path):
    """加载系统提示词"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def format_chat_message(tokenizer, system_prompt, user_question):
    """格式化聊天消息"""
    # 使用 Qwen3 的聊天模板格式
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    # 使用 tokenizer 的 apply_chat_template 方法
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    else:
        # 如果没有 apply_chat_template，手动格式化
        prompt = f"{system_prompt}\n\n用户问题: {user_question}\n\n回答:"
    
    return prompt

def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.7):
    """生成回答"""
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 确定设备
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'hf_device_map'):
        # 对于多设备模型，使用第一个设备
        device = next(iter(model.hf_device_map.values()))
    else:
        device = next(model.parameters()).device
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    # 只取新生成的部分（去掉输入部分）
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()

def main():
    # 路径配置
    model_path = "/Users/lisheng/code/NLP-Final/models/merged-qwen3-1.7B"
    prompt_path = "/Users/lisheng/code/NLP-Final/ollama_test/dog_prompt.txt"
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return
    
    if not os.path.exists(prompt_path):
        print(f"错误: 提示词文件不存在: {prompt_path}")
        return
    
    # 加载系统提示词
    print("=" * 70)
    print("加载系统提示词...")
    system_prompt = load_system_prompt(prompt_path)
    print("系统提示词加载完成")
    print("=" * 70)
    
    # 加载模型（4bit 量化）
    model, tokenizer = load_model_with_4bit(model_path)
    
    # 交互式问答
    print("\n" + "=" * 70)
    print("开始问答（输入 'quit' 或 'exit' 退出）")
    print("=" * 70 + "\n")
    
    while True:
        try:
            # 获取用户输入
            user_question = input("请输入问题: ").strip()
            
            if user_question.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not user_question:
                continue
            
            # 格式化消息
            prompt = format_chat_message(tokenizer, system_prompt, user_question)
            
            # 生成回答
            print("\n正在生成回答...")
            response = generate_response(model, tokenizer, prompt)
            
            # 显示回答
            print("\n" + "-" * 70)
            print("回答:")
            print(response)
            print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

