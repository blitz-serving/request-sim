import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_csv_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 将 e_time 转换为秒的整数
    df['e_time_seconds'] = df['e_time'] // 1000
    
    # 绘制点状图：横坐标为 e_time，纵坐标为 total_time
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.scatter(df['e_time'], df['total_time'], color='blue', label='Llama-13b')
    ax1.set_xlabel('e_time (milliseconds)')
    ax1.set_ylabel('Latency (seconds)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 使用 groupby 按 e_time_seconds 分组，并计算每秒内的 output_token 总和
    grouped = df.groupby('e_time_seconds')['output_token'].sum()

    # 绘制柱形图：横坐标为秒，纵坐标为 output_token 总和
    ax2 = ax1.twinx()
    ax2.bar(grouped.index * 1000, grouped, alpha=0.3, color='orange', width=1000, label='Output Tokens (bar)')
    ax2.set_ylabel('Output Tokens (sum per second)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    #plt.title('Total Time vs e_time with Output Tokens per Second')
    fig.tight_layout()

    plt.show()
    plt.savefig("fig.pdf")
    plt.savefig("fig.png")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Plot data from a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    
    args = parser.parse_args()

    # 调用绘图函数
    plot_csv_data(args.file_path)