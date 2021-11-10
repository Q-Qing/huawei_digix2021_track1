"""
按照df_active_user中device_id的顺序，重新排列page_action_sequence
"""
import pandas as pd
import numpy as np


result = np.load('processed_data/page_action_sequence.npy')
id_list = np.load('processed_data/behavior_ids.npy')

df_active = pd.read_csv('processed_data/device_active_with_info.csv')

matched_result = np.zeros((2000000, 60, 300, 2), dtype='uint8')
for i in range(len(id_list)):
    id = id_list[i]
    id_index = df_active.index[df_active['device_id']==id].tolist()[0]
    matched_result[id_index, :, :, :] = result[i, :, :, :]
    if i % 10000 == 0:
        print(i)

np.save('processed_data/matched_page_action_sequence.npy', matched_result)

