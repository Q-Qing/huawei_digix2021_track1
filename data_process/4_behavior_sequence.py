"""
整合每天的行为序列
包括每天打开了的页面，听的歌曲
"""
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)


def repalce_by_dic(d, arrays):

    replace_list = [d[i] for i in arrays]
    return replace_list


def update_seq(row_b, seq):
    day = int(row_b['day'])
    pages = str(row_b['pages']).replace('null', '-1')
    pages = pages.replace('nan', '-1').split('#')
    pages = list(map(int, pages))
    music_ids = str(row_b['music_ids']).replace('null', '-1')
    music_ids = music_ids.replace('nan', '-1').split('#')
    music_ids = list(map(int, music_ids))
    actions = str(row_b['actions']).replace('null', '-1')
    actions = actions.replace('nan', '-1').split('#')
    actions = list(map(int, actions))
    if len(pages) <= 300:
        seq_len = len(pages)
    else:
        seq_len = 300
    seq[day - 1, 0:seq_len, 0] = pages[0:seq_len]
    seq[day - 1, 0:seq_len, 1] = actions[0:seq_len]
    seq[day - 1, 0:seq_len, 2] = music_ids[0:seq_len]
    return seq


def update_page_action_seq(row_b, seq, page_dict):
    day = int(row_b['day'])
    pages = str(row_b['pages']).replace('null', '-1')
    pages = pages.replace('nan', '-1').split('#')
    pages = list(map(int, pages))
    pages = repalce_by_dic(page_dict, pages)
    actions = list(map(int, str(row_b['actions']).split('#')))

    if len(pages) <= 300:
        seq_len = len(pages)
    else:
        seq_len = 300
    seq[day - 1, 0:seq_len, 0] = pages[0:seq_len]
    seq[day - 1, 0:seq_len, 1] = actions[0:seq_len]
    return seq


df = pd.read_csv('2021_1_data/4_user_behavior.csv', sep="|")
df = df.sort_values(by=['device_id'], ignore_index=True)
print(df)
print(df.dtypes)
uni_pages = [-1, 7, 9, 13, 16, 17, 18, 19, 20, 21, 22, 26, 28, 29, 30, 32, 34, 36, 38, 40, 44, 45, 47]
pages_to_ix = {page: i+1 for i, page in enumerate(uni_pages)}
print(pages_to_ix)

result_tensor = []
last_device = None
id_list = []
for index, row in df.iterrows():
    device_id = row['device_id']
    if last_device is None:
        print(index, device_id)
        user_behavior_seq = np.zeros((60, 300, 2), dtype='uint8')
        user_behavior_seq = update_page_action_seq(row, user_behavior_seq, pages_to_ix)
        last_device = device_id
        continue
    if device_id != last_device:
        print(index, device_id)
        id_list.append(last_device)
        result_tensor.append(user_behavior_seq)
        # if result_tensor is None:
        #     result_tensor = user_behavior_seq
        # else:
        #     result_tensor = np.append(result_tensor, user_behavior_seq, 0)
        user_behavior_seq = np.zeros((60, 300, 2), dtype='uint8')
        user_behavior_seq = update_page_action_seq(row, user_behavior_seq, pages_to_ix)
        last_device = device_id
    else:
        user_behavior_seq = update_page_action_seq(row, user_behavior_seq, pages_to_ix)
        # last_device = device_id

result = np.array(result_tensor)
np.save('processed_data/behavior_ids.npy', np.array(id_list))
np.save('processed_data/page_action_sequence.npy', result)
print('save npy')

