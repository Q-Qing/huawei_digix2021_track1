"""
处理个人信息，并将个人信息添加到device_active.cav中
"""

import pandas as pd

# 数据处理
df_userinfo = pd.read_csv('2021_1_data/2_user_info.csv', sep="|")
print(df_userinfo)
# 缺失统计
print(df_userinfo.isna().sum())
# 统计每列的数值分布
for i in range(6):
    print(df_userinfo.iloc[:, i+1].value_counts())

# processing missing value
# gender 0, age -1, is_vip 0, topics 提取数量
df_fill = df_userinfo.fillna({'gender':-1, 'age':-1, "is_vip": -1, "topics":0})
print(df_fill)

for index, row in df_fill.iterrows():
    if row['topics'] != 0:
        topic_num = len(row['topics'].split('#'))
        # print(row['topics'], topic_num)
        df_fill.loc[index, 'topics'] = topic_num

print(df_fill['topics'].value_counts())
df_fill.to_csv('processed_data/user_info1.csv', index=False)

# 数据融合
df_info = pd.read_csv('processed_data/user_info1.csv')
df_device = pd.read_csv('processed_data/device_active.csv')

df = df_device.merge(df_info, how='left', on='device_id')
print(df)
df.to_csv('processed_data/device_active_with_info.csv', index=False)


