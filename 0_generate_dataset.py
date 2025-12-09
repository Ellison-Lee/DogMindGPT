#!/usr/bin/env python3
"""
使用 SiliconFlow API 生成机器人控制指令训练数据集
该脚本通过调用大语言模型生成自然语言指令及其对应的工具调用序列
"""

import os
import json
import time
import requests
from typing import List, Dict, Any
from datetime import datetime


class SiliconFlowDataGenerator:
    """使用 SiliconFlow API 生成训练数据的类"""
    
    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        初始化生成器
        
        Args:
            api_key: SiliconFlow API 密钥
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        调用 SiliconFlow API
        
        Args:
            prompt: 输入提示词
            temperature: 生成温度，控制随机性
            max_tokens: 最大生成token数
            
        Returns:
            LLM 生成的文本响应
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            print(f"API 调用失败: {e}")
            return ""
    
    def generate_training_samples(self, num_samples: int = 100, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        生成训练样本
        
        Args:
            num_samples: 要生成的样本数量
            batch_size: 每次生成的批次大小
            
        Returns:
            训练样本列表
        """
        samples = []
        
        # 定义系统提示词，说明可用的工具和数据格式
        system_prompt = """你是一个机器人控制指令生成专家。请生成机器人控制的自然语言指令及其对应的工具调用序列。

可用的工具有：
1. move_forward: 向前移动，参数为 distance (米，可正可负)
2. turn: 转向，参数为 angle (度，正数为右转，负数为左转)
3. sit: 坐下，无参数
4. stand: 站起来，无参数

请生成 {batch_size} 条训练数据，每条数据包含：
- question: 自然语言指令（中文）
- answer: 对应的工具调用序列（JSON格式的列表）

要求：
1. 指令要自然、多样化，符合日常使用习惯
2. 可以包含单个动作或多个动作的组合
3. 距离范围：0.5-10米
4. 角度范围：-180到180度
5. 确保生成的JSON格式正确

请直接返回JSON数组格式的数据，不要有其他解释文字。格式示例：
[
  {
    "question": "向右转15度，然后向前走0.5米。",
    "answer": [{"tool":"turn","parameters":{"angle":15}},{"tool":"move_forward","parameters":{"distance":0.5}}]
  },
  {
    "question": "给我站起来，然后向左转45度。",
    "answer": [{"tool":"stand","parameters":{}},{"tool":"turn","parameters":{"angle":-45}}]
  }
]
"""
        
        print(f"开始生成 {num_samples} 条训练样本...")
        print(f"使用模型: {self.model}")
        print("=" * 70)
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(samples))
            
            print(f"\n生成批次 {batch_idx + 1}/{num_batches} (目标: {current_batch_size} 条)...")
            
            # 构造提示词
            prompt = system_prompt.format(batch_size=current_batch_size)
            
            # 调用 LLM
            response = self.call_llm(prompt, temperature=0.8)
            
            if not response:
                print("[警告] 生成失败，跳过此批次")
                continue
            
            # 解析响应
            try:
                # 尝试提取 JSON 部分
                response = response.strip()
                
                # 如果响应被代码块包裹，提取出来
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1])
                    if response.startswith("json"):
                        response = response[4:].strip()
                
                batch_data = json.loads(response)
                
                if isinstance(batch_data, list):
                    # 验证每条数据的格式
                    valid_samples = []
                    for item in batch_data:
                        if self._validate_sample(item):
                            valid_samples.append(item)
                        else:
                            print(f"[警告] 无效样本: {item}")
                    
                    samples.extend(valid_samples)
                    print(f"[成功] 成功生成 {len(valid_samples)} 条有效样本 (总计: {len(samples)})")
                else:
                    print("[警告] 响应格式不正确，应为列表")
                    
            except json.JSONDecodeError as e:
                print(f"[警告] JSON 解析失败: {e}")
                print(f"响应内容: {response[:200]}...")
            
            # 避免请求过快
            if batch_idx < num_batches - 1:
                time.sleep(1)
        
        print("\n" + "=" * 70)
        print(f"[完成] 数据生成完成！共生成 {len(samples)} 条样本")
        
        return samples
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        验证样本格式是否正确
        
        Args:
            sample: 待验证的样本
            
        Returns:
            是否有效
        """
        if not isinstance(sample, dict):
            return False
        
        if 'question' not in sample or 'answer' not in sample:
            return False
        
        if not isinstance(sample['question'], str):
            return False
        
        if not isinstance(sample['answer'], list):
            return False
        
        # 验证工具调用格式
        valid_tools = {'move_forward', 'turn', 'sit', 'stand'}
        for action in sample['answer']:
            if not isinstance(action, dict):
                return False
            if 'tool' not in action or 'parameters' not in action:
                return False
            if action['tool'] not in valid_tools:
                return False
            if not isinstance(action['parameters'], dict):
                return False
        
        return True
    
    def save_dataset(self, samples: List[Dict[str, Any]], output_path: str):
        """
        保存数据集到文件
        
        Args:
            samples: 样本列表
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为 JSON 格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"\n[完成] 数据集已保存到: {output_path}")
        print(f"[统计] 总样本数: {len(samples)}")
        
        # 打印统计信息
        self._print_statistics(samples)
    
    def _print_statistics(self, samples: List[Dict[str, Any]]):
        """打印数据集统计信息"""
        print("\n" + "=" * 70)
        print("数据集统计信息:")
        print("=" * 70)
        
        # 统计工具使用频率
        tool_counts = {}
        action_length_counts = {}
        
        for sample in samples:
            actions = sample['answer']
            action_length = len(actions)
            action_length_counts[action_length] = action_length_counts.get(action_length, 0) + 1
            
            for action in actions:
                tool = action['tool']
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        print("\n工具使用统计:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count} 次")
        
        print("\n动作序列长度分布:")
        for length, count in sorted(action_length_counts.items()):
            print(f"  {length} 个动作: {count} 条样本")
        
        print("=" * 70)


def main():
    """主函数"""
    # 从环境变量获取 API 密钥
    API_KEY = os.getenv("SILICONFLOW_API_KEY")
    
    if not API_KEY:
        print("[错误] 未设置 SILICONFLOW_API_KEY 环境变量")
        print("请先设置 API 密钥:")
        print("  export SILICONFLOW_API_KEY='your-api-key-here'")
        return
    
    # ==================== 配置区域 ====================
    # 模型和生成参数配置
    MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 可选其他模型
    NUM_SAMPLES = 1000  # 生成样本数量
    BATCH_SIZE = 10  # 每批生成数量
    OUTPUT_PATH = "data/generated_dataset.json"
    # ================================================
    
    print("=" * 70)
    print("SiliconFlow 机器人控制指令数据集生成器")
    print("=" * 70)
    print(f"模型: {MODEL}")
    print(f"目标样本数: {NUM_SAMPLES}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"输出路径: {OUTPUT_PATH}")
    print("=" * 70)
    
    # 创建生成器
    generator = SiliconFlowDataGenerator(API_KEY, model=MODEL)
    
    # 生成数据
    samples = generator.generate_training_samples(
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE
    )
    
    # 保存数据集
    if samples:
        generator.save_dataset(samples, OUTPUT_PATH)
        print("\n[完成] 全部完成！")
    else:
        print("\n[错误] 未生成任何有效样本")


if __name__ == "__main__":
    main()

