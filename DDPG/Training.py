import sys
sys.path.append("./..")

import torch
import numpy as np
import copy
import tqdm
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from OctopusArmGym import *
from Actor import *
from Critic import *
from ReplayMemory import *


NUM_STEPS = 200

NUM_EPISODES_INIT = 500

NUM_EPISODES = 1000000
BATCH_SIZE = 128

GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = 1000000


MIN_REWARD = -1
MIN_Q = 1 / ( 1 - GAMMA) * MIN_REWARD


NUM_EPISODES_EVAL = 40

K = 4

num_interaction_episodes = 40
num_learning_steps = 40

def main():

    #
    # Device
    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #
    # Logging
    #

    file_path = f"./logs/"
    writer = SummaryWriter(file_path)

    #
    # Env
    #

    env = OctopusArmGym(sparse_reward=False)

    #
    # Actor & Critic
    #
    
    actor = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    critic = Critic(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    
    print("actor #params: ", sum(p.numel() for p in actor.parameters()))
    print("critic #params: ", sum(p.numel() for p in critic.parameters()))

    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    target_actor.eval()
    target_critic.eval()

    for param in target_actor.parameters():
        param.requires_grad = False
    for param in target_critic.parameters():
        param.requires_grad = False

    best_sucess_rate = 0

    #
    # ReplayMemory
    #

    replay_memory = ReplayMemory(capacity=MEMORY_SIZE, state_dim=OBS_DIM, action_dim=ACTION_DIM)

    # Fill replay memory 
    print("Fill ReplayMemory")
    for episode in tqdm.tqdm(range(NUM_EPISODES_INIT)):
        state, _ = env.reset()


        for step in range(NUM_STEPS):
            
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_memory.add(
                state,
                env.goal,
                action,
                reward,
                next_state,
                done
            )

            if done:
                break 

            state = next_state
        
      

          
    #
    # Training loop
    #

    best_sucess_rate = 0
    for episode in range(NUM_EPISODES):

        print(f"Episode: {episode}")

        #
        # Interact
        #

        print("Interaction:")
        for _ in tqdm.tqdm(range(num_interaction_episodes)):
            

            state, _ = env.reset()

            for step in range(NUM_STEPS):
    
                actor.eval()
                critic.eval()

                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                goal_tensor = torch.tensor(env.goal, dtype=torch.float32).to(device)
                state_tensor = torch.concat([state_tensor, goal_tensor])
                state_tensor = state_tensor.unsqueeze(dim=0) # add batch dim

                with torch.no_grad():
                    action = actor(state_tensor)

                noise = 0.1 * torch.randn_like(action)
                action = (action + noise).clamp(0, 1)
                action = action.cpu().numpy()
                action = action[0] # remove batch dim

                next_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                replay_memory.add(
                    state,
                    env.goal,
                    action,
                    reward,
                    next_state,
                    done
                )

            
                if done:
                    break
                
                state = next_state

        #
        # Learn
        #

        print("Learning:")
        for _ in tqdm.tqdm(range(num_learning_steps)):

            # sample batch

            states_b, goals_b, actions_b, rewards_b, next_states_b, dones_b = replay_memory.sample(BATCH_SIZE)

            states_b = torch.tensor(states_b).to(device)
            goals_b = torch.tensor(goals_b).to(device)
            actions_b = torch.tensor(actions_b).to(device)
            rewards_b = torch.tensor(rewards_b).to(device).unsqueeze(dim=1)
            next_states_b = torch.tensor(next_states_b).to(device)
            dones_b = torch.tensor(dones_b).to(device).unsqueeze(dim=1)

            states_b = torch.concat([states_b, goals_b], dim=1)
            next_states_b = torch.concat([next_states_b, goals_b], dim=1)

            # Critic update

            actor.eval()
            critic.train()

            critic.requires_grad_(True)

            with torch.no_grad():
                next_actions = target_actor(next_states_b)
                target_q = target_critic(next_states_b, next_actions)
                target_q = rewards_b + (1 - dones_b) * GAMMA * target_q
                target_q = torch.clip(target_q, min=MIN_Q, max=0)

            q_values = critic(states_b, actions_b)
    
            critic_loss = torch.nn.functional.mse_loss(q_values, target_q)
            critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic.optimizer.step()

            # Actor update
            actor.train()
            critic.eval()


            critic.requires_grad_(False)

            actor_actions = actor(states_b)
            actor_loss = -critic(states_b, actor_actions).mean()
            actor.optimizer.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor.optimizer.step()

            
            #
            # Update metrics
            #

            actor.loss_metric.update(actor_loss)
            critic.loss_metric.update(critic_loss)


            
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


        #
        # Evaluation
        #

        
        actor.eval()


        cnt_goal_reached = 0
        num_steps_to_goal_list = []
        min_distance_list = []

        print("Evaluation:")
        for _ in tqdm.tqdm(range(NUM_EPISODES_EVAL)):

            state, _ = env.reset()

            min_distance = 1000

            for step in range(NUM_STEPS):
                
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                goal_tensor = torch.tensor(env.goal, dtype=torch.float32).to(device)
                state_tensor = torch.concat([state_tensor, goal_tensor])
                state_tensor = state_tensor.unsqueeze(dim=0) # add batch dim

                with torch.no_grad():
                    action = actor(state_tensor)

                action = action.cpu().numpy()
                action = action[0] # remove batch dim

                next_state, reward, terminated, truncated, info = env.step(action)


                done = terminated or truncated
                
                if reward == 0:
                    cnt_goal_reached += 1

                    num_steps_to_goal_list.append(step)
                
                distance = info["distance"]

                if distance < min_distance:
                    min_distance = distance

                if done:
                    break
                
                state = next_state
            
            min_distance_list.append(min_distance)

        success_rate = cnt_goal_reached / NUM_EPISODES_EVAL

        avg_min_distance = np.mean(min_distance_list)

        if len(num_steps_to_goal_list) != 0:
            avg_num_steps_to_goal = np.mean(num_steps_to_goal_list)
        else:
            avg_num_steps_to_goal = NUM_STEPS

   
        actor_loss = actor.loss_metric.compute()
        actor.loss_metric.reset()

        critic_loss = critic.loss_metric.compute()
        critic.loss_metric.reset()

        #
        # Output
        #

        print()
        print(f" actor_loss: {actor_loss:.4f}")
        print(f"critic_loss: {critic_loss:.4f}")
        print()
        print(f" success_rate: {success_rate:.4f}")
        print()
        print("##########################")

        #
        # Update metrics
        #


        writer.add_scalars("Actor Loss", { "" : actor_loss}, episode)
        writer.add_scalars("Critic Loss", { "" : critic_loss}, episode)


        writer.add_scalars("success_rate", { "" : success_rate}, episode)
        writer.add_scalars("avg_min_distance", { "avg_min_distance" : avg_min_distance}, episode)

        writer.add_scalars("avg_num_steps_to_goal", { "avg_num_steps_to_goal" : avg_num_steps_to_goal}, episode)

        if len(num_steps_to_goal_list) != 0:
            writer.add_histogram("num_steps_to_goal", np.array(num_steps_to_goal_list), episode)

        writer.add_histogram("min_distances", np.array(min_distance_list), episode)

        writer.flush()


        #
        # Save model
        #

        if best_sucess_rate < success_rate:
            best_sucess_rate = success_rate

            torch.save(actor.state_dict(), f"./saved_models/actor")
            torch.save(critic.state_dict(), f"./saved_models/critic")

if __name__ == "__main__":
    main()
