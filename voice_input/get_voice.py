import os
# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sounddevice as sd
import numpy as np
import queue
import time
from faster_whisper import WhisperModel

# ================= 配置参数 =================
# 1. 硬件参数
MIC_INDEX = None          # None 表示自动检测，或指定设备 ID
MIC_RATE = 48000          # 麦克风原生采样率
MIC_CHANNELS = None       # None 表示自动检测，或指定通道数

# 2. Whisper 参数 (固定需求)
MODEL_RATE = 16000       # 模型必须用 16k
MODEL_SIZE = "small"      # Orin 上先跑 tiny，跑通了再换 small
DEVICE = "cpu"          # 显卡加速
COMPUTE_TYPE = "int8" # 半精度

# 3. 语音检测 (VAD) 参数
# 阈值需要根据 float归一化后的音量来定 (0.0 ~ 1.0)
# 之前 int16 的 RMS 大概几千，归一化后大概是 0.0x 到 0.x
VAD_THRESHOLD = 0.05     # 触发录音的音量阈值 (感觉不灵敏就调小，太灵敏就调大)
SILENCE_DURATION = 1.0   # 说话停顿 1 秒后开始识别
# ===========================================

# 队列用于在回调和主线程间传递音频数据
q = queue.Queue()

def get_audio_device():
    """自动检测可用的音频输入设备"""
    devices = sd.query_devices()
    input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    
    if not input_devices:
        raise RuntimeError("未找到可用的音频输入设备")
    
    # 优先使用指定的设备，否则使用第一个可用设备
    if MIC_INDEX is not None and MIC_INDEX < len(devices):
        device_info = devices[MIC_INDEX]
        if device_info['max_input_channels'] > 0:
            return MIC_INDEX, int(device_info['default_samplerate']), device_info['max_input_channels']
    
    # 使用第一个可用设备
    device_id, device_info = input_devices[0]
    channels = device_info['max_input_channels']
    samplerate = int(device_info['default_samplerate'])
    
    print(f"检测到音频设备: ID {device_id} - {device_info['name']} ({channels} 通道, {samplerate} Hz)")
    return device_id, samplerate, channels


def process_audio_chunk(indata):
    """
    核心处理函数：
    1. int16 -> float32
    2. 多通道 -> 单声道（如果需要）
    3. 48000Hz -> 16000Hz
    """
    # 1. 转为 float32 并归一化 (-1.0 ~ 1.0)
    data_float = indata.astype(np.float32) / 32768.0
   
    # 2. 多通道转单声道（如果数据是多维的）
    if len(data_float.shape) > 1 and data_float.shape[1] > 1:
        mono_data = data_float.mean(axis=1)
    else:
        mono_data = data_float.flatten()
   
    # 3. 降采样 (Resample)
    # 48000 / 16000 = 3，所以每隔 3 个点取一个数据即可
    resampled_data = mono_data[::3]
   
    return resampled_data

def callback(indata, frames, time_info, status):
    """音频回调：只负责搬运数据，不做复杂计算"""
    if status:
        print(f"Status: {status}", flush=True)
    # 放入队列
    q.put(indata.copy())

def main():
    print(f"正在加载 Whisper 模型 ({MODEL_SIZE})...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 自动检测音频设备
    try:
        device_id, device_rate, device_channels = get_audio_device()
    except Exception as e:
        print(f"音频设备检测失败: {e}")
        return

    # 使用检测到的设备参数，或使用配置的参数
    mic_id = device_id if MIC_INDEX is None else MIC_INDEX
    mic_rate = device_rate if MIC_RATE is None else MIC_RATE
    mic_channels = device_channels if MIC_CHANNELS is None else MIC_CHANNELS

    print(f"\n正在监听 (设备ID={mic_id}, 采样率={mic_rate}, 通道数={mic_channels})...")
    print(f"请说话 (阈值={VAD_THRESHOLD})")

    # 打开麦克风流
    try:
        with sd.InputStream(device=mic_id,
                            samplerate=mic_rate,
                            channels=mic_channels,
                            blocksize=4096,
                            dtype='int16',
                            callback=callback):
            audio_buffer = []  # 存储待识别的音频 (已经是 16k 的数据)
            recording = False
            last_speech_time = time.time()
           
            while True:
                # 1. 获取原始音频数据
                raw_chunk = q.get()
               
                # 2. 预处理 (变成 16k, 1ch, float32)
                processed_chunk = process_audio_chunk(raw_chunk)
               
                # 3. 计算音量 (用于 VAD)
                # 简单的 RMS (均方根)
                volume = np.sqrt(np.mean(processed_chunk**2))
               
                # --- 状态机逻辑 ---
                if volume > VAD_THRESHOLD:
                    if not recording:
                        print("\n检测到语音，开始记录...", end="", flush=True)
                        recording = True
                   
                    last_speech_time = time.time()
                    audio_buffer.append(processed_chunk)
                    print(".", end="", flush=True) # 打印点点表示正在录
               
                elif recording:
                    # 正在录音，但当前这块数据音量小 (可能是句中停顿)
                    audio_buffer.append(processed_chunk)
                   
                    # 检查停顿时间是否超过设定值
                    if time.time() - last_speech_time > SILENCE_DURATION:
                        print("\n说话结束，正在识别...", end="", flush=True)
                        recording = False
                       
                        if len(audio_buffer) > 0:
                            # 拼接所有音频块
                            final_audio = np.concatenate(audio_buffer)
                           
                            # --- 核心识别 ---
                            start_t = time.time()
                            # language="zh" 强制中文
                            segments, _ = model.transcribe(final_audio, beam_size=2, language="zh")
                           
                            text = "".join([s.text for s in segments])
                            end_t = time.time()
                           
                            print(f"\n识别结果 ({end_t - start_t:.2f}s): {text}")
                            print("-" * 40)
                       
                        # 清空 buffer
                        audio_buffer = []
               
                # (如果不 recording 且音量小，就丢弃数据，不存 buffer)
    except Exception as e:
        print(f"\n音频流打开失败: {e}")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出程序")