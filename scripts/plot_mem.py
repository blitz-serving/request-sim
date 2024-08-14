import pandas as pd
import matplotlib.pyplot as plt

name="gpu_usage_log_rr120_cv2"
# 读取 CSV 文件,没有表头
df = pd.read_csv(f'{name}.csv', header=None)

# 找到开头和结尾的连续 0 行
start = 0
while df.iloc[start][1] == 0:
    start += 1

end = len(df) - 1
while df.iloc[end][1] == 0:
    end -= 1

# 删除开头和结尾的连续 0 行
df = df.iloc[start:end+1]

# 将时间列转换为 datetime 格式
df['timestamp'] = pd.to_datetime(df[0])
df[2] = df[2].str.replace('%', '').astype(float)

# 绘制 GPU 内存占用率曲线
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['timestamp'], df[2], color='red', label='GPU Memory Utilization')
ax.set_xlabel('Time')
ax.set_ylabel('GPU Memory Utilization (%)', color='red')
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.tick_params('y', colors='red')
plt.title('GPU Memory Utilization over Time')
ax.legend()
plt.savefig(f"{name}.png")