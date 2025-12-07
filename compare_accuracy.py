#!/usr/bin/env python3
"""
比较模型量化前后回答准确率的脚本
对照组：全精度模型 (merged-qwen3-1.7B)
实验组：4bit量化模型 (ggml-model-Q4_K_M.gguf)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import json
import os
import random
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime

# 路径配置
CONTROL_MODEL_PATH = "/Users/lisheng/code/NLP-Final/models/merged-qwen3-1.7B"
EXPERIMENT_MODEL_NAME = "qwen3-lora-local:latest"  # Ollama 中的模型名称
DATASET_PATH = "/Users/lisheng/code/NLP-Final/data/generated_dataset.json"
PROMPT_PATH = "/Users/lisheng/code/NLP-Final/ollama_test/dog_prompt.txt"
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API 端点
OUTPUT_FILE = "/Users/lisheng/code/NLP-Final/output/accuracy_comparison_result.txt"

# 测试配置
SAMPLE_SIZE = 10  # 从数据集中随机抽取的问题数量

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """加载数据集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_system_prompt(prompt_path: str) -> str:
    """加载系统提示词"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_control_model(model_path: str):
    """加载对照组模型（全精度，不使用量化）"""
    print(f"正在加载对照组模型: {model_path}")
    
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
    
    # 使用全精度加载（不使用量化）
    print("加载模型（全精度）...")
    if torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = "mps"
    elif torch.cuda.is_available():
        dtype = torch.float16
        device_map = "cuda"
    else:
        dtype = torch.float32
        device_map = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"对照组模型加载完成（{dtype}，设备: {device_map}）！")
    return model, tokenizer

def format_chat_message(tokenizer, system_prompt: str, user_question: str) -> str:
    """格式化聊天消息（对照组）"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    else:
        prompt = f"{system_prompt}\n\n用户问题: {user_question}\n\n回答:"
    
    return prompt

def generate_control_response(model, tokenizer, prompt: str, temperature: float = 0.7) -> str:
    """生成对照组回答"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 确定设备
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'hf_device_map'):
        device = next(iter(model.hf_device_map.values()))
    else:
        device = next(model.parameters()).device
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
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
    
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()

def filter_thinking_tags(text: str) -> str:
    """过滤掉思考过程的标记"""
    if not text:
        return text
    
    # 先移除思考标记（保留代码块，因为可能包含JSON）
    patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reasoning>.*?</reasoning>',
        r'<thought>.*?</thought>',
    ]
    
    filtered_text = text
    for pattern in patterns:
        filtered_text = re.sub(pattern, '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
    
    # 移除代码块标记（但保留内容）
    # 匹配 ```json 或 ``` 开头的代码块标记
    filtered_text = re.sub(r'```(?:json)?\s*\n?', '', filtered_text)
    filtered_text = re.sub(r'\n?```\s*$', '', filtered_text)
    
    # 清理多余的空行
    filtered_text = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_text)
    return filtered_text.strip()


def generate_experiment_response(question: str, system_prompt: str, 
                                 base_url: str, model_name: str) -> str:
    """生成实验组回答（通过 Ollama API）"""
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
        response = requests.post(base_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "response" in result:
            response_text = result["response"].strip()
            return filter_thinking_tags(response_text)
        else:
            return ""
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return ""

def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    """从文本中提取JSON数组"""
    if not text or not text.strip():
        return []
    
    # 先过滤掉思考标记
    text = filter_thinking_tags(text)
    
    # 尝试直接解析整个文本
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    
    # 尝试提取代码块中的JSON
    json_pattern = r'```(?:json)?\s*(\[[\s\S]*?\])'
    matches = re.findall(json_pattern, text)
    if matches:
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
            except:
                continue
    
    # 尝试提取第一个出现的JSON数组（更精确的匹配）
    # 匹配从 [ 开始到匹配的 ] 结束的JSON数组
    json_array_pattern = r'(\[[\s\S]*?\])'
    matches = re.findall(json_array_pattern, text)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except:
            continue
    
    # 尝试查找包含 "tool" 的JSON对象数组
    # 匹配类似 [{"tool": ...}] 的模式
    tool_json_pattern = r'(\[\s*\{[^}]*"tool"[^}]*\}[^\]]*\])'
    matches = re.findall(tool_json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except:
            continue
    
    # 最后尝试：查找所有包含 "tool" 的JSON对象，然后组合成数组
    tool_object_pattern = r'\{[^}]*"tool"[^}]*\}'
    tool_matches = re.findall(tool_object_pattern, text, re.DOTALL)
    if tool_matches:
        result = []
        for match in tool_matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and 'tool' in obj:
                    result.append(obj)
            except:
                continue
        if result:
            return result
    
    return []

def normalize_answer(answer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """标准化答案格式，便于比较"""
    normalized = []
    for item in answer:
        if isinstance(item, dict) and 'tool' in item:
            normalized_item = {
                'tool': item['tool'],
                'parameters': item.get('parameters', {})
            }
            # 标准化参数值（将数字转为float，处理字符串数字）
            if normalized_item['parameters']:
                for key, value in normalized_item['parameters'].items():
                    if isinstance(value, (int, float)):
                        normalized_item['parameters'][key] = float(value)
                    elif isinstance(value, str):
                        try:
                            normalized_item['parameters'][key] = float(value)
                        except:
                            pass
            normalized.append(normalized_item)
    return normalized

def compare_answers(predicted: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> bool:
    """比较预测答案和期望答案是否一致（只比较tool值，不比较parameters）"""
    # 提取tool列表
    pred_tools = []
    for item in predicted:
        if isinstance(item, dict) and 'tool' in item:
            pred_tools.append(item['tool'])
    
    exp_tools = []
    for item in expected:
        if isinstance(item, dict) and 'tool' in item:
            exp_tools.append(item['tool'])
    
    # 只比较tool列表是否相同
    return pred_tools == exp_tools

def check_ollama_service() -> bool:
    """检查 Ollama 服务是否运行"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = [m.get("name", "").strip() for m in response.json().get("models", [])]
        return EXPERIMENT_MODEL_NAME.strip() in models
    except:
        return False

def run_single_experiment(dataset, system_prompt, control_model, control_tokenizer):
    """运行单次实验"""
    # 从数据集随机抽取问题
    sample_size = min(SAMPLE_SIZE, len(dataset))
    selected_samples = random.sample(dataset, sample_size)
    
    results = []
    control_correct = 0
    experiment_correct = 0
    
    for idx, sample in enumerate(selected_samples, 1):
        question = sample['question']
        expected_answer = sample['answer']
        
        print(f"\n[{idx}/{len(selected_samples)}] 问题: {question}")
        print(f"期望答案: {json.dumps(expected_answer, ensure_ascii=False)}")
        
        # 对照组回答
        print("\n生成对照组回答...")
        try:
            prompt = format_chat_message(control_tokenizer, system_prompt, question)
            control_response = generate_control_response(control_model, control_tokenizer, prompt)
            
            # 调试输出：打印原始响应
            print(f"对照组原始响应: {repr(control_response[:200])}...")  # 只打印前200个字符
            
            control_predicted = extract_json_from_text(control_response)
            control_is_correct = compare_answers(control_predicted, expected_answer)
            
            if control_is_correct:
                control_correct += 1
            
            print(f"对照组提取的JSON: {json.dumps(control_predicted, ensure_ascii=False)}")
            print(f"对照组结果: {'✅ 正确' if control_is_correct else '❌ 错误'}")
        except Exception as e:
            print(f"对照组生成错误: {e}")
            import traceback
            traceback.print_exc()
            control_predicted = []
            control_is_correct = False
        
        # 实验组回答
        print("\n生成实验组回答...")
        try:
            experiment_response = generate_experiment_response(
                question, system_prompt, OLLAMA_URL, EXPERIMENT_MODEL_NAME
            )
            
            # 调试输出：打印原始响应
            print(f"实验组原始响应: {repr(experiment_response[:200])}...")  # 只打印前200个字符
            
            experiment_predicted = extract_json_from_text(experiment_response)
            experiment_is_correct = compare_answers(experiment_predicted, expected_answer)
            
            if experiment_is_correct:
                experiment_correct += 1
            
            print(f"实验组提取的JSON: {json.dumps(experiment_predicted, ensure_ascii=False)}")
            print(f"实验组结果: {'✅ 正确' if experiment_is_correct else '❌ 错误'}")
        except Exception as e:
            print(f"实验组生成错误: {e}")
            import traceback
            traceback.print_exc()
            experiment_predicted = []
            experiment_is_correct = False
        
        # 保存结果
        results.append({
            'question': question,
            'expected_answer': expected_answer,
            'control_response_raw': control_response if 'control_response' in locals() else '',
            'control_predicted': control_predicted,
            'control_correct': control_is_correct,
            'experiment_response_raw': experiment_response if 'experiment_response' in locals() else '',
            'experiment_predicted': experiment_predicted,
            'experiment_correct': experiment_is_correct
        })
    
    # 计算准确率
    total = len(selected_samples)
    control_accuracy = control_correct / total
    experiment_accuracy = experiment_correct / total
    
    return results, control_accuracy, experiment_accuracy, control_correct, experiment_correct, total

def main():
    print("=" * 70)
    print("模型量化前后准确率比较")
    print("=" * 70)
    
    # 检查路径
    if not os.path.exists(CONTROL_MODEL_PATH):
        print(f"错误: 对照组模型路径不存在: {CONTROL_MODEL_PATH}")
        return
    
    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集路径不存在: {DATASET_PATH}")
        return
    
    if not os.path.exists(PROMPT_PATH):
        print(f"错误: 提示词文件不存在: {PROMPT_PATH}")
        return
    
    # 检查实验组Ollama服务是否运行
    print("\n检查实验组Ollama服务状态...")
    if not check_ollama_service():
        print("错误: 无法连接到 Ollama 服务或模型不存在")
        print("请确保 Ollama 服务正在运行: llama serve")
        print(f"请确保模型 '{EXPERIMENT_MODEL_NAME}' 已导入到 Ollama")
        return
    print("实验组Ollama服务运行正常")
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = load_dataset(DATASET_PATH)
    print(f"数据集大小: {len(dataset)} 条")
    
    # 加载系统提示词
    print("\n加载系统提示词...")
    system_prompt = load_system_prompt(PROMPT_PATH)
    
    # 加载对照组模型
    print("\n" + "=" * 70)
    print("加载对照组模型（全精度）...")
    control_model, control_tokenizer = load_control_model(CONTROL_MODEL_PATH)
    
    # 运行单次实验
    experiment_count = 1
    all_results = []

    print("\n" + "=" * 70)
    print(f"第 {experiment_count} 次实验")
    print("=" * 70)

    results, control_accuracy, experiment_accuracy, control_correct, experiment_correct, total = run_single_experiment(
        dataset, system_prompt, control_model, control_tokenizer
    )

    control_accuracy_pct = control_accuracy * 100
    experiment_accuracy_pct = experiment_accuracy * 100

    print("\n" + "=" * 70)
    print(f"第 {experiment_count} 次实验结果")
    print("=" * 70)
    print(f"对照组（全精度）准确率: {control_correct}/{total} = {control_accuracy_pct:.2f}% ({control_accuracy:.2f})")
    print(f"实验组（4bit量化）准确率: {experiment_correct}/{total} = {experiment_accuracy_pct:.2f}% ({experiment_accuracy:.2f})")

    all_results.append({
        'experiment_num': experiment_count,
        'results': results,
        'control_accuracy': control_accuracy,
        'experiment_accuracy': experiment_accuracy,
        'control_correct': control_correct,
        'experiment_correct': experiment_correct,
        'total': total
    })

    
    # 保存所有实验结果到文件
    print("\n" + "=" * 70)
    print("保存结果...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("模型量化前后准确率比较结果\n")
        f.write("=" * 70 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验次数: {experiment_count}\n")
        f.write(f"每次测试样本数: {SAMPLE_SIZE}\n")
        f.write(f"\n对照组模型: {CONTROL_MODEL_PATH}\n")
        f.write(f"实验组模型: {EXPERIMENT_MODEL_NAME} (Ollama)\n")
        f.write(f"\n" + "=" * 70 + "\n")
        f.write("最终准确率统计\n")
        f.write("=" * 70 + "\n")
        
        final_result = all_results[-1]
        f.write(f"对照组（全精度）: {final_result['control_correct']}/{final_result['total']} = {final_result['control_accuracy']*100:.2f}%\n")
        f.write(f"实验组（4bit量化）: {final_result['experiment_correct']}/{final_result['total']} = {final_result['experiment_accuracy']*100:.2f}%\n")
        f.write(f"准确率差异: {(final_result['experiment_accuracy'] - final_result['control_accuracy'])*100:.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("所有实验结果\n")
        f.write("=" * 70 + "\n\n")
        
        for exp_result in all_results:
            f.write(f"实验 {exp_result['experiment_num']}:\n")
            f.write(f"  对照组准确率: {exp_result['control_accuracy']*100:.2f}% ({exp_result['control_correct']}/{exp_result['total']})\n")
            f.write(f"  实验组准确率: {exp_result['experiment_accuracy']*100:.2f}% ({exp_result['experiment_correct']}/{exp_result['total']})\n")
            f.write("-" * 70 + "\n\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("最后一次实验详细结果\n")
        f.write("=" * 70 + "\n\n")
        
        final_results = all_results[-1]['results']
        for idx, result in enumerate(final_results, 1):
            f.write(f"[{idx}/{final_result['total']}] 问题: {result['question']}\n")
            f.write(f"期望答案: {json.dumps(result['expected_answer'], ensure_ascii=False)}\n")
            f.write(f"对照组原始响应: {result.get('control_response_raw', '')[:500]}\n")  # 限制长度
            f.write(f"对照组提取的JSON: {json.dumps(result['control_predicted'], ensure_ascii=False)}\n")
            f.write(f"对照组结果: {'✅ 正确' if result['control_correct'] else '❌ 错误'}\n")
            f.write(f"实验组原始响应: {result.get('experiment_response_raw', '')[:500]}\n")  # 限制长度
            f.write(f"实验组提取的JSON: {json.dumps(result['experiment_predicted'], ensure_ascii=False)}\n")
            f.write(f"实验组结果: {'✅ 正确' if result['experiment_correct'] else '❌ 错误'}\n")
            f.write("-" * 70 + "\n\n")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("所有实验完成！")
    print("=" * 70)
    print(f"总实验次数: {experiment_count}")
    final_result = all_results[-1]
    print(f"最终对照组（全精度）准确率: {final_result['control_correct']}/{final_result['total']} = {final_result['control_accuracy']*100:.2f}%")
    print(f"最终实验组（4bit量化）准确率: {final_result['experiment_correct']}/{final_result['total']} = {final_result['experiment_accuracy']*100:.2f}%")
    print(f"准确率差异: {(final_result['experiment_accuracy'] - final_result['control_accuracy'])*100:.2f}%")
    print(f"\n结果已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

