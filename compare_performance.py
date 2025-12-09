#!/usr/bin/env python3
"""
统计GGUF模型的性能指标：
- 延迟/延迟曲线
- 吞吐量
- 能耗（如果可用）
"""

import requests
import json
import os
import time
import statistics
import subprocess
import sys
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# 设置现代化配色方案
try:
    # 尝试使用 seaborn 样式（如果可用）
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
except ImportError:
    # 如果没有 seaborn，使用 matplotlib 默认样式
    plt.style.use('default')

# 定义现代化配色
MODERN_COLORS = {
    'primary': '#4A90E2',      # 现代蓝
    'secondary': '#7ED321',    # 现代绿
    'accent': '#F5A623',       # 现代橙
    'warning': '#D0021B',      # 现代红
    'neutral': '#9013FE',      # 现代紫
    'background': '#F8F9FA',   # 浅灰背景
}

# 配置
MODEL_NAME = "qwen3-lora-local:latest"  # Ollama 中的模型名称
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API 端点
PROMPT_PATH = "/Users/lisheng/Documents/Working File/code/NLP-Final/ollama_test/dog_prompt.txt"
OUTPUT_DIR = "/Users/lisheng/Documents/Working File/code/NLP-Final/output"

# 测试问题（不同长度）
TEST_QUESTIONS = [
    "向右走1米",
    "先前进5米，再左转60度，最后坐下",
    "给我站起来，然后向左转45度，再向前移动3米，最后坐下休息",
    "向右转15度，然后向前走0.5米，接着左转90度，前进2米，最后站起来",
    "先坐下，然后右转180度，再站起来，向前移动10米，左转45度，再前进5米，最后坐下",
]

def load_system_prompt(prompt_path: str) -> str:
    """加载系统提示词"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def filter_thinking_tags(text: str) -> str:
    """过滤掉思考过程的标记"""
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

def chat_with_ollama_model(question: str, system_prompt: str, 
                           ollama_url: str = OLLAMA_URL, 
                           model_name: str = MODEL_NAME) -> Tuple[Optional[Dict], float, Optional[int]]:
    """
    使用 Ollama API 进行对话，返回结果、延迟和生成的token数
    
    Returns:
        (result, latency_ms, generated_tokens)
    """
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
    
    start_time = time.time()
    try:
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 提取生成的token数（如果API返回）
        generated_tokens = None
        if 'prompt_eval_count' in result and 'eval_count' in result:
            generated_tokens = result.get('eval_count', None)
        
        # 过滤思考标记并构造类似原格式的result以便兼容
        response_text = result.get('response', '')
        filtered_response = filter_thinking_tags(response_text)
        
        formatted_result = {
            'choices': [{
                'message': {
                    'content': filtered_response
                }
            }],
            'usage': {
                'completion_tokens': generated_tokens
            }
        }
        
        return formatted_result, latency_ms, generated_tokens
        
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"请求错误: {e}")
        return None, latency_ms, None

def get_power_metrics() -> Optional[Dict[str, float]]:
    """
    获取能耗指标（macOS）
    返回: {"cpu_energy": float, "gpu_energy": float} 或 None
    """
    try:
        # 尝试使用powermetrics（需要sudo权限）
        # 这里只是示例，实际使用可能需要更复杂的实现
        result = subprocess.run(
            ["sysctl", "-n", "hw.cpufrequency"],
            capture_output=True,
            text=True,
            timeout=2
        )
        # 注意：实际能耗测量需要更专业的工具
        return None  # 简化实现，返回None
    except:
        return None

def run_performance_test(n_runs: int = 20) -> Dict:
    """
    运行性能测试
    
    Args:
        n_runs: 每个问题运行的次数
    
    Returns:
        包含所有性能指标的字典
    """
    print("=" * 70)
    print("开始性能测试")
    print("=" * 70)
    
    # 检查Ollama服务是否运行
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = [m.get("name", "").strip() for m in response.json().get("models", [])]
        if MODEL_NAME.strip() not in models:
            print(f"错误: 模型 '{MODEL_NAME}' 未找到")
            print(f"可用模型: {', '.join(models)}")
            return None
    except requests.exceptions.RequestException:
        print("错误: 无法连接到 Ollama 服务，请确保 Ollama 正在运行")
        print("启动命令: llama serve")
        return None
    
    # 加载系统提示词
    if not os.path.exists(PROMPT_PATH):
        print(f"错误: 提示词文件不存在: {PROMPT_PATH}")
        return None
    
    system_prompt = load_system_prompt(PROMPT_PATH)
    print(f"已加载系统提示词（长度: {len(system_prompt)} 字符）")
    
    # 存储所有测试结果
    all_results = {
        'latencies': [],  # 所有延迟（毫秒）
        'latencies_by_question': {},  # 按问题分组的延迟
        'throughputs': [],  # 吞吐量（tokens/s）
        'generated_tokens': [],  # 生成的token数
        'question_lengths': [],  # 问题长度
        'response_lengths': [],  # 响应长度
        'timestamps': [],  # 时间戳
    }
    
    print(f"\n测试配置:")
    print(f"  - 测试问题数: {len(TEST_QUESTIONS)}")
    print(f"  - 每个问题运行次数: {n_runs}")
    print(f"  - 总测试次数: {len(TEST_QUESTIONS) * n_runs}")
    print("\n开始测试...\n")
    
    # 运行测试
    total_tests = len(TEST_QUESTIONS) * n_runs
    current_test = 0
    
    for q_idx, question in enumerate(TEST_QUESTIONS):
        question_latencies = []
        question_throughputs = []
        question_tokens = []
        
        print(f"问题 {q_idx + 1}/{len(TEST_QUESTIONS)}: {question[:50]}...")
        
        for run in range(n_runs):
            current_test += 1
            print(f"  运行 {run + 1}/{n_runs}...", end='\r')
            sys.stdout.flush()
            
            result, latency_ms, generated_tokens = chat_with_ollama_model(
                question, system_prompt, OLLAMA_URL, MODEL_NAME
            )
            
            if result is None:
                print(f"\n  ⚠️  运行 {run + 1} 失败")
                continue
            
            # 提取响应内容
            response_content = ""
            if 'choices' in result and len(result['choices']) > 0:
                response_content = result['choices'][0]['message']['content']
            
            # 计算吞吐量（如果知道token数）
            throughput = None
            if generated_tokens and generated_tokens > 0:
                throughput = generated_tokens / (latency_ms / 1000)  # tokens/s
                question_throughputs.append(throughput)
                all_results['throughputs'].append(throughput)
            
            # 记录数据
            question_latencies.append(latency_ms)
            all_results['latencies'].append(latency_ms)
            all_results['timestamps'].append(time.time())
            all_results['question_lengths'].append(len(question))
            all_results['response_lengths'].append(len(response_content))
            
            if generated_tokens:
                question_tokens.append(generated_tokens)
                all_results['generated_tokens'].append(generated_tokens)
            
            # 短暂延迟，避免过载
            time.sleep(0.1)
        
        print(f"  完成！平均延迟: {statistics.mean(question_latencies):.2f} ms")
        
        # 保存每个问题的结果
        all_results['latencies_by_question'][f"Q{q_idx+1}"] = {
            'question': question,
            'latencies': question_latencies,
            'avg_latency': statistics.mean(question_latencies),
            'min_latency': min(question_latencies),
            'max_latency': max(question_latencies),
            'std_latency': statistics.stdev(question_latencies) if len(question_latencies) > 1 else 0,
            'throughputs': question_throughputs,
            'avg_throughput': statistics.mean(question_throughputs) if question_throughputs else None,
            'tokens': question_tokens,
            'avg_tokens': statistics.mean(question_tokens) if question_tokens else None,
        }
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    
    # 计算总体统计
    if all_results['latencies']:
        all_results['stats'] = {
            'total_requests': len(all_results['latencies']),
            'avg_latency_ms': statistics.mean(all_results['latencies']),
            'min_latency_ms': min(all_results['latencies']),
            'max_latency_ms': max(all_results['latencies']),
            'median_latency_ms': statistics.median(all_results['latencies']),
            'std_latency_ms': statistics.stdev(all_results['latencies']) if len(all_results['latencies']) > 1 else 0,
            'p50_latency_ms': np.percentile(all_results['latencies'], 50),
            'p95_latency_ms': np.percentile(all_results['latencies'], 95),
            'p99_latency_ms': np.percentile(all_results['latencies'], 99),
        }
        
        if all_results['throughputs']:
            all_results['stats']['avg_throughput_tokens_per_s'] = statistics.mean(all_results['throughputs'])
            all_results['stats']['max_throughput_tokens_per_s'] = max(all_results['throughputs'])
            all_results['stats']['min_throughput_tokens_per_s'] = min(all_results['throughputs'])
        
        if all_results['generated_tokens']:
            all_results['stats']['avg_generated_tokens'] = statistics.mean(all_results['generated_tokens'])
            all_results['stats']['total_generated_tokens'] = sum(all_results['generated_tokens'])
    
    return all_results

def plot_latency_curve(results: Dict, output_dir: str):
    """绘制延迟曲线"""
    if not results or 'latencies' not in results or not results['latencies']:
        print("没有延迟数据可绘制")
        return
    
    latencies = results['latencies']
    
    # 创建图形，使用现代化样式
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('white')
    
    # 1. 延迟时间序列图
    ax1 = axes[0, 0]
    ax1.plot(range(len(latencies)), latencies, color=MODERN_COLORS['primary'], 
             alpha=0.7, linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Request Index', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontsize=20, fontweight='bold')
    ax1.set_title('Latency Time Series', fontsize=20, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(MODERN_COLORS['background'])
    
    # 2. 延迟分布直方图
    ax2 = axes[0, 1]
    n, bins, patches = ax2.hist(latencies, bins=30, edgecolor='white', 
                                 alpha=0.8, color=MODERN_COLORS['primary'], linewidth=1.5)
    # 渐变色彩
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.Blues(0.3 + 0.7 * i / len(patches)))
    ax2.axvline(statistics.mean(latencies), color=MODERN_COLORS['warning'], 
                linestyle='--', linewidth=2, label=f'Mean: {statistics.mean(latencies):.2f} ms')
    ax2.axvline(statistics.median(latencies), color=MODERN_COLORS['secondary'], 
                linestyle='--', linewidth=2, label=f'Median: {statistics.median(latencies):.2f} ms')
    ax2.set_xlabel('Latency (ms)', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=20, fontweight='bold')
    ax2.set_title('Latency Distribution Histogram', fontsize=20, fontweight='bold', pad=10)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(MODERN_COLORS['background'])
    
    # 3. 按问题分组的延迟箱线图
    ax3 = axes[1, 0]
    if results.get('latencies_by_question'):
        question_data = []
        question_labels = []
        for q_key, q_data in results['latencies_by_question'].items():
            question_data.append(q_data['latencies'])
            question_labels.append(f"{q_key}\n({len(q_data['question'])} chars)")
        
        bp = ax3.boxplot(question_data, labels=question_labels, patch_artist=True,
                         boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color=MODERN_COLORS['warning']),
                         capprops=dict(linewidth=1.5))
        colors = [MODERN_COLORS['primary'], MODERN_COLORS['secondary'], 
                 MODERN_COLORS['accent'], MODERN_COLORS['neutral'], '#50E3C2']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        ax3.set_xlabel('Question', fontsize=20, fontweight='bold')
        ax3.set_ylabel('Latency (ms)', fontsize=20, fontweight='bold')
        ax3.set_title('Latency Distribution by Question', fontsize=20, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_facecolor(MODERN_COLORS['background'])
    
    # 4. 累积分布函数（CDF）
    ax4 = axes[1, 1]
    sorted_latencies = sorted(latencies)
    y = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
    ax4.plot(sorted_latencies, y * 100, color=MODERN_COLORS['primary'], 
             linewidth=3, label='CDF')
    ax4.axvline(np.percentile(latencies, 50), color=MODERN_COLORS['warning'], 
                linestyle='--', linewidth=2, label=f'P50: {np.percentile(latencies, 50):.2f} ms')
    ax4.axvline(np.percentile(latencies, 95), color=MODERN_COLORS['secondary'], 
                linestyle='--', linewidth=2, label=f'P95: {np.percentile(latencies, 95):.2f} ms')
    ax4.axvline(np.percentile(latencies, 99), color=MODERN_COLORS['accent'], 
                linestyle='--', linewidth=2, label=f'P99: {np.percentile(latencies, 99):.2f} ms')
    ax4.set_xlabel('Latency (ms)', fontsize=20, fontweight='bold')
    ax4.set_ylabel('Cumulative Probability (%)', fontsize=20, fontweight='bold')
    ax4.set_title('Latency Cumulative Distribution Function (CDF)', fontsize=20, fontweight='bold', pad=10)
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor(MODERN_COLORS['background'])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'latency_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"延迟曲线已保存到: {output_path}")
    plt.close()

def plot_throughput(results: Dict, output_dir: str):
    """绘制吞吐量图（去除极端值）"""
    if not results or 'throughputs' not in results or not results['throughputs']:
        print("没有吞吐量数据可绘制")
        return
    
    throughputs = results['throughputs']
    
    # 使用 IQR 方法去除极端值
    if len(throughputs) > 4:  # 需要足够的数据点才能计算 IQR
        q1 = np.percentile(throughputs, 25)
        q3 = np.percentile(throughputs, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 过滤极端值
        filtered_throughputs = [t for t in throughputs if lower_bound <= t <= upper_bound]
        outliers_count = len(throughputs) - len(filtered_throughputs)
        
        if outliers_count > 0:
            print(f"已过滤 {outliers_count} 个极端值（IQR方法）")
            throughputs = filtered_throughputs
    
    if not throughputs:
        print("过滤后没有数据可绘制")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    
    # 1. 吞吐量时间序列
    ax1 = axes[0]
    ax1.plot(range(len(throughputs)), throughputs, color=MODERN_COLORS['secondary'], 
             alpha=0.7, linewidth=2, marker='o', markersize=3)
    ax1.axhline(statistics.mean(throughputs), color=MODERN_COLORS['warning'], 
                linestyle='--', linewidth=2, 
                label=f'Mean: {statistics.mean(throughputs):.2f} tokens/s')
    ax1.set_xlabel('Request Index', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=20, fontweight='bold')
    ax1.set_title('Throughput Time Series', fontsize=20, fontweight='bold', pad=10)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(MODERN_COLORS['background'])
    
    # 2. 吞吐量分布直方图
    ax2 = axes[1]
    n, bins, patches = ax2.hist(throughputs, bins=30, edgecolor='white', 
                                 alpha=0.8, color=MODERN_COLORS['secondary'], linewidth=1.5)
    # 渐变色彩
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.Greens(0.3 + 0.7 * i / len(patches)))
    ax2.axvline(statistics.mean(throughputs), color=MODERN_COLORS['warning'], 
                linestyle='--', linewidth=2, 
                label=f'Mean: {statistics.mean(throughputs):.2f} tokens/s')
    ax2.set_xlabel('Throughput (tokens/s)', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=20, fontweight='bold')
    ax2.set_title('Throughput Distribution Histogram', fontsize=20, fontweight='bold', pad=10)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(MODERN_COLORS['background'])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'throughput.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"吞吐量图已保存到: {output_path}")
    plt.close()

def save_results(results: Dict, output_dir: str):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON格式的详细结果
    json_path = os.path.join(output_dir, 'performance_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存到: {json_path}")
    
    # 保存文本格式的摘要
    txt_path = os.path.join(output_dir, 'performance_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("模型性能测试结果摘要\n")
        f.write("=" * 70 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {MODEL_NAME} (Ollama)\n")
        f.write("\n")
        
        if 'stats' in results:
            stats = results['stats']
            f.write("总体统计:\n")
            f.write("-" * 70 + "\n")
            f.write(f"总请求数: {stats.get('total_requests', 0)}\n")
            f.write(f"平均延迟: {stats.get('avg_latency_ms', 0):.2f} ms\n")
            f.write(f"最小延迟: {stats.get('min_latency_ms', 0):.2f} ms\n")
            f.write(f"最大延迟: {stats.get('max_latency_ms', 0):.2f} ms\n")
            f.write(f"中位数延迟: {stats.get('median_latency_ms', 0):.2f} ms\n")
            f.write(f"标准差: {stats.get('std_latency_ms', 0):.2f} ms\n")
            f.write(f"P50延迟: {stats.get('p50_latency_ms', 0):.2f} ms\n")
            f.write(f"P95延迟: {stats.get('p95_latency_ms', 0):.2f} ms\n")
            f.write(f"P99延迟: {stats.get('p99_latency_ms', 0):.2f} ms\n")
            
            if 'avg_throughput_tokens_per_s' in stats:
                f.write(f"\n平均吞吐量: {stats['avg_throughput_tokens_per_s']:.2f} tokens/s\n")
                f.write(f"最大吞吐量: {stats['max_throughput_tokens_per_s']:.2f} tokens/s\n")
                f.write(f"最小吞吐量: {stats['min_throughput_tokens_per_s']:.2f} tokens/s\n")
            
            if 'avg_generated_tokens' in stats:
                f.write(f"\n平均生成token数: {stats['avg_generated_tokens']:.2f}\n")
                f.write(f"总生成token数: {stats['total_generated_tokens']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("按问题分组的统计:\n")
        f.write("=" * 70 + "\n")
        
        if 'latencies_by_question' in results:
            for q_key, q_data in results['latencies_by_question'].items():
                f.write(f"\n{q_key}: {q_data['question']}\n")
                f.write(f"  平均延迟: {q_data['avg_latency']:.2f} ms\n")
                f.write(f"  最小延迟: {q_data['min_latency']:.2f} ms\n")
                f.write(f"  最大延迟: {q_data['max_latency']:.2f} ms\n")
                f.write(f"  标准差: {q_data['std_latency']:.2f} ms\n")
                if q_data.get('avg_throughput'):
                    f.write(f"  平均吞吐量: {q_data['avg_throughput']:.2f} tokens/s\n")
                if q_data.get('avg_tokens'):
                    f.write(f"  平均生成token数: {q_data['avg_tokens']:.2f}\n")
    
    print(f"摘要已保存到: {txt_path}")

def main():
    """主函数"""
    print("=" * 70)
    print("GGUF模型性能测试工具")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 运行性能测试
    results = run_performance_test(n_runs=20)
    
    if not results:
        print("测试失败，退出")
        return
    
    # 打印统计信息
    if 'stats' in results:
        stats = results['stats']
        print("\n" + "=" * 70)
        print("性能统计摘要")
        print("=" * 70)
        print(f"总请求数: {stats.get('total_requests', 0)}")
        print(f"平均延迟: {stats.get('avg_latency_ms', 0):.2f} ms")
        print(f"P50延迟: {stats.get('p50_latency_ms', 0):.2f} ms")
        print(f"P95延迟: {stats.get('p95_latency_ms', 0):.2f} ms")
        print(f"P99延迟: {stats.get('p99_latency_ms', 0):.2f} ms")
        if 'avg_throughput_tokens_per_s' in stats:
            print(f"平均吞吐量: {stats['avg_throughput_tokens_per_s']:.2f} tokens/s")
    
    # 绘制图表
    print("\n正在生成图表...")
    plot_latency_curve(results, OUTPUT_DIR)
    plot_throughput(results, OUTPUT_DIR)
    
    # 保存结果
    save_results(results, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("所有结果已保存到:", OUTPUT_DIR)
    print("=" * 70)

if __name__ == "__main__":
    main()

