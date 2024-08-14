import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

# 读取CSV文件
# with open("./output/mooncake-c2-r1.csv", "r") as file:
name = "output-1p2d-burstgpt-30rr-cv1-llama7b"
with open(f"./{name}.csv", "r") as file:
    lines = file.readlines()
# 将每行数据转换为JSON对象
data = [json.loads(line) for line in lines]

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将时间相关的列转换为浮点数
time_columns = ["first_token_time", "inference_time", "max_time_between_tokens"]
df[time_columns] = df[time_columns].astype(float)

# TODO: cap before request temp: 筛去long context
df = df[df["max_time_between_tokens"] != -1]

# 将s_time转换为时间序列
df["s_time"] = pd.to_datetime(df["s_time"], unit="ms")
df = df.sort_values(by='s_time')

# 计算p50和p99
p50_first_token_time = df["first_token_time"].quantile(0.5)
p90_first_token_time = df["first_token_time"].quantile(0.90)
p99_first_token_time = df["first_token_time"].quantile(0.99)

p50_inference_time = df["inference_time"].quantile(0.5)
p90_inference_time = df["inference_time"].quantile(0.90)
p99_inference_time = df["inference_time"].quantile(0.99)

p50_max_time_between_tokens = df["max_time_between_tokens"].quantile(0.5)
p90_max_time_between_tokens = df["max_time_between_tokens"].quantile(0.90)
p99_max_time_between_tokens = df["max_time_between_tokens"].quantile(0.99)

# 绘制折线图
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df["s_time"], df["first_token_time"], label="first_token_time")
plt.axhline(y=p50_first_token_time, color="r", linestyle="--", label="p50")
plt.axhline(y=p90_first_token_time, color="b", linestyle="--", label="p90")
plt.axhline(y=p99_first_token_time, color="g", linestyle="--", label="p99")
plt.legend()
plt.title("First Token Time")
plt.xticks(rotation=45)  # 旋转横坐标标签以便更好地显示

plt.subplot(3, 1, 2)
plt.plot(df["s_time"], df["inference_time"], label="inference_time")
plt.axhline(y=p50_inference_time, color="r", linestyle="--", label="p50")
plt.axhline(y=p90_inference_time, color="b", linestyle="--", label="p90")
plt.axhline(y=p99_inference_time, color="g", linestyle="--", label="p99")
plt.legend()
plt.title("Inference Time")
plt.xticks(rotation=45) 

plt.subplot(3, 1, 3)
plt.plot(df['s_time'], df["max_time_between_tokens"], label="max_time_between_tokens")
plt.axhline(y=p50_max_time_between_tokens, color="r", linestyle="--", label="p50")
plt.axhline(y=p90_max_time_between_tokens, color="b", linestyle="--", label="p90")
plt.axhline(y=p99_max_time_between_tokens, color="g", linestyle="--", label="p99")
plt.legend()
plt.title("Max Time Between Tokens")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
plt.savefig(f"{name}.png")
