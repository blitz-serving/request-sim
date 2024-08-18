import csv

#name = "/huggingface/datasets/burstgpt-v1.csv"
name = "/huggingface/datasets/burstgpt-v2.csv"
# 读取 CSV 文件
data = []
with open(name, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

data = data[1:]

# 计算第四列的均值
column_4_sum = 0
input_sum = 0
api_sum = 0
api_cnt = 0
api_input_sum = 0
cnt = 0
for row in data:
    if row[5] == "Conversation log":
        column_4_sum += float(row[3])
        input_sum += float(row[2])
        cnt += 1
    else:
        api_input_sum += float(row[2])
        api_sum += float(row[3])
        api_cnt += 1

column_4_mean = column_4_sum / cnt
input_ave = input_sum / cnt

print(f"name: {name}")

print(f"conversation ave : {input_ave:.2f}, {column_4_mean:.2f}")

api_mean = api_sum / api_cnt
api_input_ave = api_input_sum / api_cnt
print(f"api ave {api_input_ave:.2f} , {api_mean:.2f}")

total_input_ave = (api_input_sum + input_sum) / (cnt + api_cnt)
total_output_ave = (column_4_sum + api_sum) / (cnt + api_cnt)
print(f"total ave {total_input_ave:.2f}, {total_output_ave:.2f}")