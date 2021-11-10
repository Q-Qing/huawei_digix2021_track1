"""
分割原始的device_active.csv
"""
import pandas as pd

df_active = pd.read_csv('2021_1_data/1_device_active.csv', sep="|")
print(df_active)
active_matrix = []
for index, row in df_active.iterrows():
    active_flag = [0]*60
    active_days = row[1].split('#')
    for day in active_days:
        active_flag[int(day)-1] = 1
    active_matrix.append(active_flag)

column_name = list(range(1, 61))
df = pd.DataFrame(active_matrix, columns=column_name)
df.insert(0, 'device_id', df_active['device_id'])
print(df)
df.to_csv('processed_data/device_active.csv', index=False)
