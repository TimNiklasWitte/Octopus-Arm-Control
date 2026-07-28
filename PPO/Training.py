import sys
sys.path.append("./..")

import torch
import numpy as np
from gymnasium.vector import SyncVectorEnv
import copy
import tqdm
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

from Actor import *
from Critic import *
from RolloutBuffer import *
from OctopusArmGym import *


NUM_UPDATES = 10000000


NUM_ROLLOUT_EPISODES = 20
NUM_STEPS = 200

NUM_EPOCHS = 10

BATCH_SIZE = 256

GAMMA = 0.99
GAE_LAMBDA = 0.95


entropy_coef = 0.001


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
    # RolloutBuffer
    #

    rollout_buffer = RolloutBuffer(capacity=NUM_ROLLOUT_EPISODES*NUM_STEPS, state_dim=STATE_DIM, action_dim=ACTION_DIM)


    #
    # Actor & Critic
    #
    
    actor = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    critic = Critic(state_dim=STATE_DIM).to(device)
  

    #
    # Training loop
    #
    
    best_sucess_rate = 0

    for update_step in range(NUM_UPDATES):

        print(f"Update step: {update_step}")


        

        actor.eval()


        

        print("Rollout:")

        
        reward_list_all = []

        cnt_goal_reached = 0
        num_steps_to_goal_list = []
        min_distance_list = []

        for _ in tqdm.tqdm(range(NUM_ROLLOUT_EPISODES)):


            state, _ = env.reset()

            state_list = []
            action_list = []
            log_probs_list = []
            reward_list = []
            done_list = []

            min_distance = 10000

            for step in range(NUM_STEPS):
                
                state = np.concat([state, env.goal], axis=0)

                    
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                state_tensor = state_tensor.unsqueeze(dim=0) # add batch dim

                with torch.no_grad():
                    action, log_prob = actor(state_tensor)

                action = action.cpu().numpy()
                action = action[0] # remove batch dim

                log_prob = log_prob.cpu().numpy()
                log_prob = log_prob[0] # remove batch dim

                action_performed = np.clip(action, 0, 1)
                next_state, reward, terminated, truncated, info = env.step(action_performed)

                done = terminated or truncated
                

                state_list.append(state)
                action_list.append(action)
                log_probs_list.append(log_prob)
                reward_list.append(reward)
                reward_list_all.append(reward)
                done_list.append(terminated)

                state = next_state


                distance = info["distance"]

                if distance < min_distance:
                    min_distance = distance


                if done:
                    cnt_goal_reached += 1

                    num_steps_to_goal_list.append(step)

                if done:
                    break 
            
            min_distance_list.append(min_distance)


            num_rewards = len(reward_list)

            next_state = np.concat([state, env.goal], axis=0)

            state_list.append(next_state) # append last state -> bootstrap

            states = np.stack(state_list, axis=0)
            state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
                
            with torch.no_grad():
                values = critic(state_tensor)

            values = values.cpu().numpy()

            advantages = np.zeros(num_rewards, dtype=np.float32)

            for t in range(num_rewards):
                discount = 1
                a_t = 0
                for k in range(t, num_rewards):
                    a_t += discount*(reward_list[k] + GAMMA*values[k+1]*(1-int(done_list[k])) - values[k])
                    discount *= GAMMA * GAE_LAMBDA
                advantages[t] = a_t.item()

      
            for idx, (state, action, log_prob, advantage, value) in enumerate(zip(state_list, action_list, log_probs_list, advantages, values)):
                rollout_buffer.add(state=state, action=action, log_prob=log_prob, advantage=advantage, value=value.item())

    
        success_rate = cnt_goal_reached / NUM_ROLLOUT_EPISODES
        avg_min_distance = np.mean(min_distance_list)

        if len(num_steps_to_goal_list) != 0:
            avg_num_steps_to_goal = np.mean(num_steps_to_goal_list)
        else:
            avg_num_steps_to_goal = NUM_STEPS
  

        for _ in range(NUM_EPOCHS):
            states_batch_list, actions_batch_list, log_probs_batch_list, advantages_batch_list, values_batch_list = rollout_buffer.get_batches(BATCH_SIZE)


            for states, actions, log_probs_old, advantages, values in zip(states_batch_list, actions_batch_list, log_probs_batch_list, advantages_batch_list, values_batch_list):
                #
                # Update actor and critic
                #

            
                states = torch.tensor(states, device=device)
                actions = torch.tensor(actions, device=device)
                log_probs_old = torch.tensor(log_probs_old, device=device)
                advantages = torch.tensor(advantages, device=device)
                values = torch.tensor(values, device=device)

                #
                # Compute ratio
                #

                log_probs_new, entropy = actor.get_logprobs(states, actions)
                log_probs_new = log_probs_new.unsqueeze(dim=1)
                log_ratio = log_probs_new - log_probs_old
                ratio = torch.exp(log_ratio)

                #
                # Advantages
                #

                values_pred = critic(states)
                returns = advantages + values
                

                # normalization
                advantages = (advantages - advantages.mean()) / (advantages.std() + 0.00001)


                #
                # Policy loss
                #

                eps = 0.2


                policy_loss = - torch.min(
                    ratio * advantages,
                    torch.clip(ratio, 1 - eps, 1 + eps) * advantages
                ).mean() - entropy_coef * entropy.mean()

            
                actor.optimizer.zero_grad()


                policy_loss.backward()
                #torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor.optimizer.step()

                #
                # Value loss
                #
                
                value_loss = F.mse_loss(values_pred, returns)
            
                critic.optimizer.zero_grad()

                value_loss.backward()

                #torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic.optimizer.step()


                #
                # Update metrics
                #

                actor.loss_metric.update(policy_loss)
                critic.loss_metric.update(value_loss)


        rollout_buffer.reset()


        
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


        writer.add_scalars("Actor Loss", { "" : actor_loss}, update_step)
        writer.add_scalars("Critic Loss", { "" : critic_loss}, update_step)


        writer.add_scalars("success_rate", { "" : success_rate}, update_step)
        writer.add_scalars("avg_min_distance", { "avg_min_distance" : avg_min_distance}, update_step)

        writer.add_scalars("avg_num_steps_to_goal", { "avg_num_steps_to_goal" : avg_num_steps_to_goal}, update_step)

        if len(num_steps_to_goal_list) != 0:
            writer.add_histogram("num_steps_to_goal", np.array(num_steps_to_goal_list), update_step)

        writer.add_histogram("min_distances", np.array(min_distance_list), update_step)
        
    
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