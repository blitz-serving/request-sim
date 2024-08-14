import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from matplotlib.ticker import FormatStrFormatter

def plot_csv_data(file_path):
    # 读取CSV文件
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)
    
    # 将 e_time 转换为秒的整数
    df['e_time_seconds'] = df['e_time'].astype(float) // 1000
    
    # 绘制点状图：横坐标为 e_time，纵坐标为 total_time
    fig, ax1 = plt.subplots(figsize=(12, 6))
    print(df['e_time'].values)
    print(df['total_time'].values)
    ax1.scatter(df['e_time'].values, df['total_time'].values, color='blue', label='Llama-13b', s=1)
    ax1.set_xlabel('e_time (milliseconds)')
    ax1.set_ylabel('Latency (seconds)', color='blue')
    #y_formatter = FormatStrFormatter('%1.2f')
    #ax1.yaxis.set_major_formatter(y_formatter)
    #plt.xticks(np.arange(0, 700000, 100000))
    #plt.yticks(np.arange(0, 50, 5))
    plt.subplots_adjust(left=0.1)
    #ax1.tick_params(axis='y', labelcolor='blue')

    #plt.title('Total Time vs e_time with Output Tokens per Second')
    #fig.tight_layout()

    #plt.show()
    plt.savefig("fig.pdf")
    plt.savefig("fig.png")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Plot data from a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    
    args = parser.parse_args()

    # 调用绘图函数
    plot_csv_data(args.file_path)