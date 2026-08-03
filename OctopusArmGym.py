import numpy as np
import gymnasium as gym
import mujoco

OBS_DIM = 127
STATE_DIM = 130
ACTION_DIM = 18
EPSILON = 0.1

class OctopusArmGym(gym.Env):


    def __init__(self, sparse_reward=True):
        super(OctopusArmGym, self).__init__()
        
        self.sparse_reward = sparse_reward

    

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("./../octopus_arm_6sites_6each.xml")
        self.data = mujoco.MjData(self.model)
        
        # Action space
        self.action_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(ACTION_DIM,), 
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(OBS_DIM,), 
            dtype=np.float32
        )
        
        #
        # Render
        #

        self.renderer = mujoco.Renderer(
            self.model,
            height=480,
            width=640,
        )

        self.target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "tgt"
        )


        #
        # Get id of end effector
        #

        self.end_effector_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "tip"
        )


        # Simulation steps per environment step
        self.n_substeps = 30
        
        # Initialize state
        self.state = None
        
        # Init goal
        self.goal = np.zeros(shape=(3,))
        
        self.reset()
    

    def _sample_reachable_goal(self):

        # Save current simulation state
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        # Random joint angles
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0.0

        n_steps = np.random.randint(50, 200)
        for i in range(n_steps):

            #self.data.ctrl[:] = self.action_space.sample()

            self.data.ctrl[:] = self.np_random.uniform(
                low=self.action_space.low,
                high=self.action_space.high,
            )
             
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)

        # Read ee position
        goal = self.data.site_xpos[self.end_effector_id].copy()

        # Restore simulation state
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)

        return goal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        #
        # Reset MuJoCo simulation
        #
        mujoco.mj_resetData(self.model, self.data)

        #
        # Reset initial joint positions
        #

        self.data.qpos[:] = 0
        # reset velocities
        self.data.qvel[:] = 0.0

        #
        # Step simulation to apply initial positions
        #
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        #
        # Reset goal
        #
        self.goal = self._sample_reachable_goal()
        
        self.state = self._get_observation()
        self.done = False
        return self.state, {}

    
    def _get_observation(self):

        joint_orientations = self.data.qpos.copy()

        joint_angular_velocities = self.data.qvel.copy()

        tendon_lengths = self.data.ten_length.copy()

        tendon_velocities = self.data.ten_velocity.copy()

        muscle_activations = self.data.ctrl.copy()

        tip_position = self.data.site_xpos[self.end_effector_id].copy()
      
        obs = np.concatenate([
            joint_orientations,       # 40
            joint_angular_velocities,  # 30
            tendon_lengths,            # 6
            tendon_velocities,         # 6
            muscle_activations,        # 6
            tip_position              # 3
        ])

        return obs.astype(np.float32)
    
    def step(self, action):
        
        #
        # Clip action to action space bounds
        #
     
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        #
        # Apply action
        #
        self.data.ctrl[:] = action
        
        #
        # Run multiple simulation substeps
        #

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        #
        # Get new observation
        #

        self.state = self._get_observation()

        #
        # Goal reached?
        #

        # Get hand position
        end_effector_pos = self.data.site_xpos[self.end_effector_id]

        dist = np.linalg.norm(end_effector_pos - self.goal)
        goal_reached = dist < EPSILON

        if self.sparse_reward:
            if goal_reached:
                reward = 0
                self.done = True
            else:
                reward = -1
        
        else:
            reward = -dist

        #
        # Termination
        #

        # Time limit
        truncated = self.data.time >= 10.0  
    
        terminated = goal_reached 
        
        info = {
            "hand_pos": end_effector_pos.copy(),
            "distance": dist
        }
        return self.state, reward, terminated, truncated, info
    
    
    def render(self, camera="front"):
        
        self.model.site_pos[self.target_site_id] = self.goal
        mujoco.mj_forward(self.model, self.data)


        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()
    
    def close(self):
        self.renderer.close()
