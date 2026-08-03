from tbparse import SummaryReader
import pandas as pd

import os

map_dirName_to_name = {
    "Reward_avg_reward": "avg reward",
    "Critic Loss_": "critic loss",
    "Critic loss_1": "critic loss 1",
    "Critic loss_2": "critic loss 2",
    "ValueNetwork Loss_": "value network loss",
    "Actor Loss_": "actor loss",
    "avg_num_steps_to_goal_avg_num_steps_to_goal": "avg number steps to goal",
    "avg_min_distance_avg_min_distance": "avg min distance",
    "success_rate_": "success rate"
}

def load_dataframe(log_dir):

    

    dir_list = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    
    df_list = []
    for dir_name in dir_list:

        reader = SummaryReader(f"{log_dir}/{dir_name}")

        df = reader.scalars

        name = map_dirName_to_name[dir_name]
        df = df.rename(columns={'step': 'step', "value": name})

        df = df.set_index(['step'])

        df = df.drop("tag", axis=1)

        df_list.append(df)
  
    df = pd.concat(df_list, axis=1)

    return df