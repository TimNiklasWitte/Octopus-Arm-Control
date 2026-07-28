import sys
sys.path.append("./..")

import torch
import numpy as np
import copy
import tqdm
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

from OctopusArmGym import *
from Actor import *
from Critic import *
from ValueNetwork import *
from ReplayMemory import *


NUM_EPOCHS = 1000
BATCH_SIZE = 256

GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = 1000000

REWARD_SCALE = 5

K = 4

NUM_STEPS = 200

NUM_EPISODES_INIT = 500

num_interaction_episodes = 40
num_learning_steps = 40

ALPHA = 1

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
    
    actor = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=256, action_scale=1).to(device)
    critic_1 = Critic(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=256).to(device)
    critic_2 = Critic(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=256).to(device)
    
    value_network = ValueNetwork(state_dim=STATE_DIM, hidden_dim=256).to(device)
    value_network_target = copy.deepcopy(value_network).to(device)

  
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

            state = next_state

            if done:
                break 
     

    #
    # Training loop
    #

    for epoch in range(NUM_EPOCHS):

        print(f"Epoch: {epoch}")

        #
        # Interaction
        #

        print("Interaction:")

        cnt_goal_reached = 0
        num_steps_to_goal_list = []
        min_distance_list = []

        for _ in tqdm.tqdm(range(num_interaction_episodes)):

            state, _ = env.reset()


            min_distance = 1000

            for step in range(NUM_STEPS):
 
        
                #actor.eval()
                #critic.eval()

                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                goal_tensor = torch.tensor(env.goal, dtype=torch.float32).to(device)
                state_tensor = torch.concat([state_tensor, goal_tensor])
                state_tensor = state_tensor.unsqueeze(dim=0) # add batch dim


                with torch.no_grad():
                    action, _ = actor.get_action(state_tensor, reparameterize=False)

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
                    cnt_goal_reached += 1

                    num_steps_to_goal_list.append(step)
                
                distance = info["distance"]

                if distance < min_distance:
                    min_distance = distance

                if done:
                    break


                state = next_state

            min_distance_list.append(min_distance)



        success_rate = cnt_goal_reached / num_interaction_episodes
        avg_min_distance = np.mean(min_distance_list)

        if len(num_steps_to_goal_list) != 0:
            avg_num_steps_to_goal = np.mean(num_steps_to_goal_list)
        else:
            avg_num_steps_to_goal = NUM_STEPS

        #
        # Learning
        #

        print("Learning:")
        for _ in tqdm.tqdm(range(num_learning_steps)):

            
      

            # sample batch
            states_b, goals_b, actions_b, rewards_b, next_states_b, dones_b = replay_memory.sample(BATCH_SIZE)

            states_b = torch.tensor(states_b).to(device)
            goals_b = torch.tensor(goals_b).to(device)
            actions_b = torch.tensor(actions_b).to(device)
            rewards_b = torch.tensor(rewards_b).to(device)#.unsqueeze(dim=1)
            next_states_b = torch.tensor(next_states_b).to(device)
            dones_b = torch.tensor(dones_b).to(device)#.unsqueeze(dim=1)

            
            states_b = torch.concat([states_b, goals_b], dim=-1)
            next_states_b = torch.concat([next_states_b, goals_b], dim=-1)

            #
            # Value network
            #

            value_network.optimizer.zero_grad()

            with torch.no_grad():
                actions, log_probs = actor.get_action(states_b, reparameterize=False)
                log_probs = log_probs.view(-1)

                q1_new_policy = critic_1(states_b, actions)
                q2_new_policy = critic_2(states_b, actions)
                critic_value = torch.min(q1_new_policy, q2_new_policy)
                critic_value = critic_value.view(-1)

                value_target = critic_value - ALPHA * log_probs
    
            value_pred = value_network(states_b)
            value_pred = value_pred.view(-1)
                
            value_loss = 0.5 * F.mse_loss(value_pred, value_target)
            value_loss.backward()
            value_network.optimizer.step()

            value_network.loss_metric.update(value_loss)


            #
            # Actor
            #

            actor.optimizer.zero_grad()

            
            actions, log_probs = actor.get_action(states_b, reparameterize=True)
            log_probs = log_probs.view(-1)

            q1_new_policy = critic_1.forward(states_b, actions)
            q2_new_policy = critic_2.forward(states_b, actions)
            critic_value = torch.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)
                
            actor_loss = actor_loss = (ALPHA * log_probs - critic_value).mean()
            
            actor_loss.backward()
            actor.optimizer.step()

            actor.loss_metric.update(actor_loss)


            #
            # Critic
            #

            critic_1.optimizer.zero_grad()
            critic_2.optimizer.zero_grad()

            with torch.no_grad():
                value_next_state_pred = value_network_target(next_states_b)
                value_next_state_pred = value_next_state_pred.view(-1)

                
                q_hat = REWARD_SCALE * rewards_b + (1 - dones_b) * GAMMA * value_next_state_pred

            q1_old_policy = critic_1(states_b, actions_b).view(-1)
            q2_old_policy = critic_2(states_b, actions_b).view(-1)

            critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

            critic_loss = critic_1_loss + critic_2_loss
            critic_loss.backward()
            critic_1.optimizer.step()
            critic_2.optimizer.step()

            critic_1.loss_metric.update(critic_1_loss)
            critic_2.loss_metric.update(critic_2_loss)

            # Soft update

            for target_param, param in zip(value_network_target.parameters(), value_network.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)





          

        value_network_loss = value_network.loss_metric.compute()
        value_network.loss_metric.reset()
        
        actor_loss = actor.loss_metric.compute()
        actor.loss_metric.reset()

        critic_1_loss = critic_1.loss_metric.compute()
        critic_1.loss_metric.reset()

        critic_2_loss = critic_2.loss_metric.compute()
        critic_2.loss_metric.reset()

        #
        # Output
        #

        print()
        print(f"value_network_loss: {value_network_loss:.4f}")
        print(f"        actor_loss: {actor_loss:.4f}")
        print(f"     critic_1_loss: {critic_1_loss:.4f}")
        print(f"     critic_2_loss: {critic_2_loss:.4f}")
        print()
        print(f" success_rate: {success_rate:.4f}")
        print()
        print("##########################")

        #
        # Update metrics
        #

        writer.add_scalars("ValueNetwork Loss", { "" : value_network_loss}, epoch)

        writer.add_scalars("Actor Loss", { "" : actor_loss}, epoch)

        writer.add_scalars("Critic loss", { "1" : critic_1_loss, "2": critic_2_loss}, epoch)

        writer.add_scalars("success_rate", { "" : success_rate}, epoch)
        writer.add_scalars("avg_min_distance", { "avg_min_distance" : avg_min_distance}, epoch)

        writer.add_scalars("avg_num_steps_to_goal", { "avg_num_steps_to_goal" : avg_num_steps_to_goal}, epoch)

        if len(num_steps_to_goal_list) != 0:
            writer.add_histogram("num_steps_to_goal", np.array(num_steps_to_goal_list), epoch)

        writer.add_histogram("min_distances", np.array(min_distance_list), epoch)
        
        writer.flush()

        #
        # Save model
        #

        if best_sucess_rate < success_rate:
            best_sucess_rate = success_rate

            torch.save(value_network.state_dict(), f"./saved_models/value_network")
            torch.save(actor.state_dict(), f"./saved_models/actor")
            torch.save(critic_1.state_dict(), f"./saved_models/critic_1")
            torch.save(critic_2.state_dict(), f"./saved_models/critic_2")
            

if __name__ == "__main__":
    main()