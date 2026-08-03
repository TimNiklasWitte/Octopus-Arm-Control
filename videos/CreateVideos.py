import sys
sys.path.append("./..")


import torch
import imageio.v2 as imageio
import numpy as np
import random
import os

from OctopusArmGym import *

NUM_VIDEOS = 16
AGENT = "SAC_HER"


NUM_STEPS = 200
SEED = 42

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.append(f"./../{AGENT}")
    from Actor import Actor

    if AGENT == "SAC" or AGENT == "SAC_HER":

        actor = Actor(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                hidden_dim=256,
                action_scale=1
        )

    else:
        actor = Actor(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=256,
        )


    if AGENT == "SAC_HER":
        actor.load_state_dict(torch.load(f"./../{AGENT}/saved_models_5/actor"))
    else:
        actor.load_state_dict(torch.load(f"./../{AGENT}/saved_models/actor"))

    actor.to(device)
    actor.eval()

    env = OctopusArmGym()
    
    root = f"./{AGENT}"
    os.makedirs(root, exist_ok=True)

    for video_id in range(NUM_VIDEOS):

        seed = SEED + video_id

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        obs, _ = env.reset(seed=seed)
        
        
        temp_path = f"{root}/{video_id}.mp4"
        writer = imageio.get_writer(
            temp_path,
            fps=30
        )

  
        for _ in range(NUM_STEPS):

            state = np.concatenate([obs, env.goal])

            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():

                if AGENT == "PPO":
                    action, _ = actor(state)
                    action = action.cpu().numpy()[0]

                elif AGENT == "SAC" or AGENT == "SAC_HER":
                    action, _ = actor.get_action(state)
                    action = action.cpu().numpy()[0]
                
                else:
                    action = actor(state).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            writer.append_data(frame)

            if terminated or truncated:
                break

        writer.close()

        # Rename after knowing the result
        final_path = f"{root}/{video_id}_{terminated}.mp4"
        os.rename(temp_path, final_path)

    env.close()


if __name__ == "__main__":
    main()