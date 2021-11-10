"""
得到200w用户每日行为的统计数据，每天是否活跃，听歌次数，打开渠道
mixed_behavior.npy：200w * 60 * 3
"""
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

# 提取数量
df = pd.read_csv('2021_1_data/4_user_behavior.csv', sep="|")
print(df)
print(df['channel'].value_counts())


def num_extract(strs):
    # if not np.isnan(strs):
    num = len(str(strs).split('#'))
    # else:
    #     num = 0
    return num

total_nums = []
for index, row in df.iterrows():
    # print(row[2])
    num_pages = num_extract(row[2])
    num_music = num_extract(row[3])
    num_actions = num_extract(row[4])
    total_nums.append([num_pages, num_music, num_actions])

df_num = pd.DataFrame(total_nums, columns=['num_pages', 'num_music', 'num_actions'])
df_num.insert(0, 'device_id', df['device_id'])
df_num.insert(1, 'day', df['day'])
df_num.insert(5, 'channel', df['channel'])
print(df_num)
df_num.to_csv('processed_data/behavior_num.csv', index=False)


# 检查各数量是否一致（页面，歌曲，动作数量完全一致）
df_num = pd.read_csv('processed_data/behavior_num.csv')
same_num = 0
for index, row in df_num.iterrows():
    if row[2] == row[3] and row[3] == row[4]:
        same_num += 1

print(df_num)
print(same_num)


# 合并active信息和behavior信息，构建一个三维张量（用户数*天数*特征数=200w*60*3）
df_num = pd.read_csv('processed_data/behavior_num.csv')
print(df_num['channel'].value_counts())
df_num = df_num.fillna({'channel': -1})
print(df_num['channel'].value_counts())
channel_to_ix = {channel: i for i, channel in enumerate(df_num['channel'].unique())}
print(channel_to_ix)
channel_idx = [channel_to_ix[d] for d in df_num['channel']]
# 0 表示没有打开 1 表示缺失
df_num['channel_idx'] = np.array(channel_idx) + 1
print(df_num)

result_tensor = []
df_active = pd.read_csv('processed_data/device_active_with_info.csv')
for index, row in df_active.iterrows():
    device_id = row['device_id']
    # 60
    active_days = row[1:61].values
    user_beavior = np.zeros((60, 3))
    user_beavior[:, 0] = active_days
    df_behavior = df_num.loc[df_num['device_id'] == device_id]
    if df_behavior.empty:
        result_tensor.append(user_beavior.tolist())
    else:
        for index_b, row_b in df_behavior.iterrows():
            day = int(row_b['day'])
            user_beavior[day-1, 1] = row_b['num_pages']
            user_beavior[day-1, 2] = row_b['channel_idx']
        result_tensor.append(user_beavior.tolist())
result = np.array(result_tensor)
np.save('processed_data/mixed_behavior.npy', result)


