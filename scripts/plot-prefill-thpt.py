import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import json

def calculate_throughput_per_millisecond(csv_file):
    throughput = defaultdict(float)
    
    with open(csv_file, 'r') as csvfile:
        reader = [json.loads(line) for line in csvfile.readlines()]
        for row in reader:
            s_time = int(float(row['s_time']) + float(row['queue_time'])*1000)
            e_time = int(float(row['s_time']) + float(row['queue_time'])*1000 + float(row['first_token_time'])*1000)
            input_length = float(row['input_length'])
            
            # Calculate total milliseconds between s_time and e_time
            duration_ms = e_time - s_time + 1
            
            # Calculate throughput per millisecond
            throughput_per_millisecond = input_length / duration_ms
            
            # Distribute throughput across each millisecond
            for t in range(s_time, e_time + 1):
                throughput[t] += throughput_per_millisecond
    
    return throughput

def aggregate_throughput_per_second(throughput):
    throughput_per_second = defaultdict(float)
    
    for time_ms, value in throughput.items():
        time_sec = time_ms // 1000  # Convert milliseconds to seconds
        throughput_per_second[time_sec] += value
    
    return throughput_per_second

def plot_throughput(throughput_per_second, name):
    times = sorted(throughput_per_second.keys())
    values = [throughput_per_second[t] for t in times]
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, values, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (tokens/s)')
    plt.title('Prefill Throughput Over Time')
    plt.grid(True)
    #plt.show()
    plt.savefig(f"{name}-prefill-thpt.png")

# Usage
name = "output-1p2d-burstgpt-old-100wrr-cv2"
csv_file = f"{name}.csv"  # replace with your actual CSV file path
throughput = calculate_throughput_per_millisecond(csv_file)
throughput_per_second = aggregate_throughput_per_second(throughput)
plot_throughput(throughput_per_second, name)
