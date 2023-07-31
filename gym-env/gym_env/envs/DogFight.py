import gymnasium as gym
import numpy as np
import pygame

from copy import deepcopy
from typing import Optional, Union

class DogFightEnv(gym.Env):
    metadata = {
        "render_fps"   : 15,
        "render_modes" : ["human", "rgb_array"]
    }

    # Initialize the environment
    def __init__(self, time_limit, eval_mode, render_mode = None):
        # ====================================================================
        # Constants / less volatile variables
        # ====================================================================
        # Environment evaluation mode indicator
        self.EVAL_MODE = eval_mode

        # Episode time & time limit
        self.TAU = 1 / self.metadata["render_fps"]
        self.STEP_LIMIT = time_limit * self.metadata["render_fps"]

        # Rendering settings
        self.RENDER_MODE = render_mode
        self.SCREEN_SIZE = (800, 800)
        self.SCREEN = None
        self.CLOCK = None

        # Colors (w/o transparency)
        self.COLOR_RED    = (255,   0,   0, 255)
        self.COLOR_ORANGE = (255, 165,   0, 255)
        self.COLOR_YELLOW = (255, 255,   0, 255)
        self.COLOR_GREEN  = (  0, 255,   0, 255)
        self.COLOR_BLUE   = (  0,   0, 255, 255)
        self.COLOR_INDIGO = ( 75,   0, 130, 255)
        self.COLOR_VIOLET = (127,   0, 255, 255)
        self.COLOR_WHITE  = (255, 255, 255, 255)
        self.COLOR_GRAY   = (128, 128, 128, 255)
        self.COLOR_BLACK  = (  0,   0,   0, 255)

        # Colors (w/  transparency)
        self.COLOR_CLEAR_ALPHA  = (  0,   0,   0,   0)
        self.COLOR_RED_ALPHA    = (255,   0,   0,  32)
        self.COLOR_ORANGE_ALPHA = (255, 165,   0,  32)
        self.COLOR_YELLOW_ALPHA = (255, 255,   0,  32)
        self.COLOR_GREEN_ALPHA  = (  0, 255,   0,  32)
        self.COLOR_BLUE_ALPHA   = (  0,   0, 255,  32)
        self.COLOR_INDIGO_ALPHA = ( 75,   0, 130,  32)
        self.COLOR_VIOLET_ALPHA = (127,   0, 255,  32)
        self.COLOR_WHITE_ALPHA  = (255, 255, 255,  32)
        self.COLOR_GRAY_ALPHA   = (128, 128, 128,  32)
        self.COLOR_BLACK_ALPHA  = (  0,   0,   0,  32)

        # Game area settings (for out of bounds checks and other things)
        self.MIN_X = 0
        self.MAX_X = self.SCREEN_SIZE[0]
        self.MIN_Y = 0
        self.MAX_Y = self.SCREEN_SIZE[1]
        self.MAX_D = np.sqrt(self.MAX_X**2 + self.MAX_Y**2)
        self.ORIGIN = (0.5 * self.MAX_X, 0.5 * self.MAX_Y)
        self.WORLD_AREA = (0, 0, self.MAX_X, self.MAX_Y)

        # Turn directions:
        # Straight - no angle change so it is 0
        # Left - increase in angle so +1
        # Right - decrease in angle to -1
        self.STRAIGHT =  0
        self.LEFT     =  1
        self.RIGHT    = -1

        # Player properties
        self.PLAYER_RADIUS = 10
        # Player min speed: 10 seconds to cross game area
        # Player max speed:  5 seconds to cross game area
        self.PLAYER_MIN_SPEED = self.MAX_X / 10
        self.PLAYER_MAX_SPEED = self.MAX_X /  5
        # Player min acceleration: 0.25 seconds from max to min speed
        # Player max acceleration: 2.50 seconds from min to max speed
        self.PLAYER_MIN_ACCELERATION = (
            -(self.PLAYER_MAX_SPEED - self.PLAYER_MIN_SPEED) / 0.25
        )
        self.PLAYER_MAX_ACCELERATION = (
             (self.PLAYER_MAX_SPEED - self.PLAYER_MIN_SPEED) / 2.50
        )
        # Player min turn rate: 4.0 seconds to do a 180
        # Player max turn rate: 2.0 seconds to do a 180
        self.PLAYER_MIN_TURN_RATE = np.pi / 4.0
        self.PLAYER_MAX_TURN_RATE = np.pi / 2.0
        # Player observation radius is 1/4 the width of the game area
        self.PLAYER_OBSERVATION_RANGE = 0.35 * self.MAX_X

        # Player missile properties
        # Track the missiles for properly assigning delayed rewards outside env
        self.PLAYER_MISSILE_RADIUS = 3
        self.PLAYER_MISSILE_SPEED = 3.0 * self.PLAYER_MAX_SPEED
        self.PLAYER_MISSILE_RANGE = 3 * self.PLAYER_OBSERVATION_RANGE

        # Enemy properties
        self.ENEMY_RADIUS = 10
        # Enemy is 0.8 x player speed
        self.ENEMY_MIN_SPEED = 0.80 * self.PLAYER_MIN_SPEED
        self.ENEMY_MAX_SPEED = 0.80 * self.PLAYER_MAX_SPEED
        # Enemy min acceleration: 0.25 seconds from max to min speed
        # Enemy max acceleration: 2.50 seconds from min to max speed
        self.ENEMY_MIN_ACCELERATION = (
            -(self.ENEMY_MAX_SPEED - self.ENEMY_MIN_SPEED) / 0.25
        )
        self.ENEMY_MAX_ACCELERATION = (
             (self.ENEMY_MAX_SPEED - self.ENEMY_MIN_SPEED) / 2.50
        )
        # Enemy min turn rate: 4.0 seconds to do a 180
        # Enemy max turn rate: 2.0 seconds to do a 180
        self.ENEMY_MIN_TURN_RATE = np.pi / 4.0
        self.ENEMY_MAX_TURN_RATE = np.pi / 2.0
        # Enemy observation radius is 3/4 of the player's observation radius
        self.ENEMY_OBSERVATION_RANGE = 1.25 * self.PLAYER_OBSERVATION_RANGE

        # Enemy missile properties
        self.ENEMY_MISSILE_RADIUS = 3
        self.ENEMY_MISSILE_SPEED = 0.250 * self.PLAYER_MISSILE_SPEED
        self.ENEMY_MISSILE_RANGE = 3 * self.ENEMY_OBSERVATION_RANGE

        # Target properties
        self.TARGET_RADIUS = 30
        # Range from which player missiles will destroy the target
        self.TARGET_OPENING_RANGE = 0.35 * self.PLAYER_MISSILE_RANGE

        # Reward definitions
        self.REWARD_MISSILE_MISS = -0.75 * self.TAU
        self.REWARD_MISSILE_HIT_ENEMY = 50
        self.REWARD_MISSILE_HIT_TARGET = 100
        self.REWARD_PLAYER_COLLIDES_WITH_ENEMY = -500
        self.REWARD_PLAYER_LEAVES_GAME = -100
        self.REWARD_TIME_PENALTY = -2 * self.TAU
        self.REWARD_APPROACH_TARGET = abs(self.REWARD_TIME_PENALTY)
        self.N_REWARD_COMPONENTS = 7
        self.RIND_MISSILE_MISS = 0
        self.RIND_MISSILE_HIT_ENEMY = 1
        self.RIND_MISSILE_HIT_TARGET = 2
        self.RIND_PLAYER_COLLIDES_WITH_ENEMY = 3
        self.RIND_PLAYER_LEAVES_GAME = 4
        self.RIND_TIME_PENALTY = 5
        self.RIND_APPROACH_TARGET = 6

        # 0.) Jet absolute x position
        # 1.) Jet absolute y position
        # 2.) Angle to target (from player coord-system)
        # 3.) Distance to target
        # 4.) Enemy visibility
        # 5.) Angle to enemy (from player coord-system)
        # 6.) Distance to enemy
        # 7.) Enemy bullet visibility
        # 8.) Angle to enemy bullet (from player coord-system)
        # 9.) Distance to enemy bullet
        self.observation_space = gym.spaces.Box(
            low = np.array([
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min
            ]),
            high = np.array([
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max
            ]),
            dtype = np.float64
        )

        # Environment action space:
        # 0.) Down
        # 1.) Up
        # 2.) Left
        # 3.) Right
        # 4.) Shoot target
        # 5.) Shoot enemy
        self.action_space = gym.spaces.Discrete(6)
        self.ACTION_DOWN         = 0
        self.ACTION_UP           = 1
        self.ACTION_LEFT         = 2
        self.ACTION_RIGHT        = 3
        self.ACTION_SHOOT_TARGET = 4
        self.ACTION_SHOOT_ENEMY  = 5

        self.TRAJ_MOVE = 0
        self.TRAJ_SHOOT_TARGET = 1
        self.TRAJ_SHOOT_ENEMY  = 2

        # ====================================================================
        # Frequently updated variables (save most for getting / setting state)
        # ====================================================================
        self.counterfactual_trajectories = []
        self.prop_cfact_trajectories = []
        self.reset_start = True
        self.state = None
        self.perform_split_facts = False
        self.draw_actions = False
        self.rewind = False

        self.player = None
        self.enemy = None
        self.target = None

        self.step_count = 0
        self.player_last_dist = np.finfo(np.float64).max
        self.paused = False
        self.player_missile_id_counter = 0

        self.lower_surface   = None
        self.pobs_surface    = None
        self.eobs_surface    = None
        self.zone_surface    = None
        self.factual_surface = None

        self.action = None
        self.player_starting_pos = None

    # Take a single simulation step in the environment
    def step(self, action):
        self.action = action
        # Additional dictionary to track missile information for assigning
        # delayed rewards to the proper step (done outside the environment)
        step_info = {
            "shoot_act"    : False, # missile related action this step
            "shoot_id"     : None,  # the id of the missile shot during this step
            "hit_ids"      : [],    # the ids of the missiles that hit this step
            "miss_ids"     : [],    # the ids of the missiles that missed this step
            "hit_rewards"  : [],    # rewards for the missiles that hit
            "miss_rewards" : [],    # rewards (penalties) for the missiles that missed
            "dreward"      : [0] * self.N_REWARD_COMPONENTS,
            "dhit_ind"     : [],
            "dmis_ind"     : [],
        }

        if self.EVAL_MODE and self.RENDER_MODE == "human":
#            if self.paused and not self.perform_split_facts:
            if self.paused and not self.perform_split_facts:
                self.render()
                return np.array(self.state, dtype = np.float64), 0, False, False, step_info
#                if self.paused:
#                    return np.array(self.state, dtype = np.float64), 0, False, False, step_info

        if self.perform_split_facts:
            self.player_starting_pos = (self.player.x, self.player.y)

        p_shot_at_nothing = False
        if action == self.ACTION_SHOOT_TARGET:
            if self.player.distance_to(self.target) <= self.TARGET_OPENING_RANGE:
                fired_in_zone = True
            else:
                fired_in_zone = False
            angle = self.player.angle_to(self.target)
            new_missile = Missile(
                self.player.x,
                self.player.y,
                self.PLAYER_MISSILE_SPEED,
                self.player.angle + angle,
                self.PLAYER_MISSILE_RADIUS,
                self.PLAYER_MISSILE_RANGE,
                id = self.player_missile_id_counter,
                fired_in_zone = fired_in_zone,
                target = self.target
            )
            self.player.angle = new_missile.angle
            self.player.missiles.append(new_missile)
            step_info["shoot_id"] = self.player_missile_id_counter
            step_info["shoot_act"] = True
            self.player_missile_id_counter += 1
        elif action == self.ACTION_SHOOT_ENEMY:
            if not self.enemy.dead and self.player.distance_to(self.enemy) <= self.player.observation_range:
                fired_in_zone = False
                angle = self.player.angle_to(self.enemy)
                new_missile = Missile(
                    self.player.x,
                    self.player.y,
                    self.PLAYER_MISSILE_SPEED,
                    self.player.angle + angle,
                    self.PLAYER_MISSILE_RADIUS,
                    self.PLAYER_MISSILE_RANGE,
                    id = self.player_missile_id_counter,
                    fired_in_zone = fired_in_zone,
                    target = self.enemy
                )
                self.player.angle = new_missile.angle
                self.player.missiles.append(new_missile)
                step_info["shoot_id"] = self.player_missile_id_counter
                step_info["shoot_act"] = True
                self.player_missile_id_counter += 1
            else:
                p_shot_at_nothing = True

        # Handle the enemy actions (if it is still alive)
        if not self.enemy.dead:
            enemy_turn_direction = self.STRAIGHT
            distance_to_player = self.enemy.distance_to(self.player)
            player_in_range = (distance_to_player <= self.enemy.observation_range)

            # If the player is not in range
            if not player_in_range:
                # Does the enemy have a guess as to where the player went?
                if self.enemy.guess_position is not None:
                    # Approach the position where the enemy predicts the player to be
                    if self.enemy.distance_to(self.enemy.guess_position) > 0.5 * self.enemy.observation_range:
                        enemy_turn_direction = self.enemy.get_turn_direction_to(self.enemy.guess_position)
                    # If the enemy arrives to the predicted position and finds nothing, stop pursuing
                    else:
                        self.enemy.guess_position = None
                # Turn back to the center of the game area if the enemy has no target to pursue
                elif not self.enemy.in_area(*(self.WORLD_AREA)):
#                    enemy_turn_direction = self.enemy.get_turn_direction_to((self.ORIGIN[0], self.ORIGIN[1]))
                    enemy_turn_direction = self.enemy.get_turn_direction_to(self.ORIGIN)
            # If the player is in range
            else:
                # Chase the player
                enemy_turn_direction = self.enemy.get_turn_direction_to(self.player)
                # Constantly predict where the player will go in case the enemy loses vision
                # (Predict position in 2.5 sec)
                self.enemy.guess_position = (
                    self.player.x + 2.5 * self.player.speed * np.cos(self.player.angle),
                    self.player.y + 2.5 * self.player.speed * np.sin(self.player.angle)
                )
                # Enemy will shoot player if it is within firing FOV (only 1 missile at a time for now)
                if self.enemy.missile is None:
                    self.enemy.missile = Missile(
                        self.enemy.x,
                        self.enemy.y,
                        self.ENEMY_MISSILE_SPEED,
                        self.enemy.angle + self.enemy.angle_to(self.player),
                        self.ENEMY_MISSILE_RADIUS,
                        self.ENEMY_MISSILE_RANGE
                    )

        # Move the player
        p_oob = False
        old_pos = self.player.x, self.player.y
        if action == self.ACTION_SHOOT_TARGET or action == self.ACTION_SHOOT_ENEMY:
            self.player.move(self.TAU, self.player.prev_move_dir)
        else:
            self.player.move(self.TAU, action)
            self.player.prev_move_dir = action
        if not self.player.in_area(*(self.WORLD_AREA)):
            p_oob = True
            self.player.x, self.player.y = old_pos[0], old_pos[1]

        # Move the enemy and its missile
        if not self.enemy.dead:
            self.enemy.move(self.TAU, turn_direction = enemy_turn_direction)
        if self.enemy.missile is not None:
            self.enemy.missile.move(self.TAU)

        # ================================================================================
        # Player Missile Movement & Delayed Reward Handling
        # ================================================================================
        terminated = False
        for missile in self.player.missiles[:]:
            missile.move(self.TAU)
            # Missiles that reach their maximum range are considered a miss
            if missile.range < missile.distance_to(missile.origin):
                step_info["miss_ids"].append(missile.id)
                step_info["shoot_act"] = True
                step_info["miss_rewards"].append(self.REWARD_MISSILE_MISS)
                step_info["dmis_ind"].append(self.RIND_MISSILE_MISS)
                self.player.missiles.remove(missile)
            # Missile collides with enemy
            elif not self.enemy.dead and missile.collides_with(self.enemy) and missile.target == self.enemy:
                step_info["hit_ids"].append(missile.id)
                step_info["shoot_act"] = True
                step_info["hit_rewards"].append(self.REWARD_MISSILE_HIT_ENEMY)
                step_info["dhit_ind"].append(self.RIND_MISSILE_HIT_ENEMY)
                self.enemy.dead = True
                self.player.missiles.remove(missile)
            # Missile collides with target (terminating condition)
            elif missile.collides_with(self.target) and missile.fired_in_zone == True and missile.target == self.target:
                step_info["hit_ids"].append(missile.id)
                step_info["shoot_act"] = True
                step_info["hit_rewards"].append(self.REWARD_MISSILE_HIT_TARGET)
                step_info["dhit_ind"].append(self.RIND_MISSILE_HIT_TARGET)
                self.player.missiles.remove(missile)
                terminated = True
                break
        # ================================================================================

        # ================================================================================
        # Immediate Rewards
        # ================================================================================
        # The player gets hit by an enemy bullet (terminating)
        p_hit_by_missile = False
        if self.enemy.missile is not None:
            if self.enemy.missile.range < self.enemy.missile.distance_to(self.enemy.missile.origin):
                self.enemy.missile = None
            elif self.player.collides_with(self.enemy.missile):
                p_hit_by_missile = True
        # The player collides with the enemy (terminating condition)
        if p_hit_by_missile or (not self.enemy.dead and self.enemy.collides_with(self.player)):
            step_info["dreward"][self.RIND_PLAYER_COLLIDES_WITH_ENEMY] = self.REWARD_PLAYER_COLLIDES_WITH_ENEMY
            terminated = True
        else:
            if p_oob:
                step_info["dreward"][self.RIND_PLAYER_LEAVES_GAME] = self.REWARD_PLAYER_LEAVES_GAME
            # Constant negative reward to encourage the agent to finish sooner
            if p_shot_at_nothing:
                step_info["dreward"][self.RIND_MISSILE_MISS] = self.REWARD_MISSILE_MISS
            step_info["dreward"][self.RIND_TIME_PENALTY] = self.REWARD_TIME_PENALTY
            dist_to_target = self.player.distance_to(self.target)
            if dist_to_target < self.player_last_dist:
                step_info["dreward"][self.RIND_APPROACH_TARGET] = self.REWARD_APPROACH_TARGET
            self.player_last_dist = dist_to_target
        # ================================================================================

        if self.RENDER_MODE == "human":
            self.render()

        # Prepare to return the observations in a normalized manner
        px_norm = self.player.x / self.MAX_X
        py_norm = self.player.y / self.MAX_Y
        tdx = self.target.x - self.player.x
        tdy = self.target.y - self.player.y
        ta_norm = np.arctan2(tdy, tdx) / np.pi
        td_norm = self.player.distance_to(self.target) / self.MAX_D
        if self.enemy.dead or self.player.distance_to(self.enemy) > self.player.observation_range:
            enemy_visible = 0.
            ea_norm = 0.
            ed_norm = 0.
        else:
            enemy_visible = 1.
            edx = self.enemy.x - self.player.x
            edy = self.enemy.y - self.player.y
            ea_norm = np.arctan2(edy, edx) / np.pi
            ed_norm = self.player.distance_to(self.enemy) / self.MAX_D
        if self.enemy.missile is None or self.player.distance_to(self.enemy.missile) > self.player.observation_range:
            enemy_missile_visible = 0.
            ema_norm = 0.
            emd_norm = 0.
        else:
            enemy_missile_visible = 1.
            emdx = self.enemy.missile.x - self.player.x
            emdy = self.enemy.missile.y - self.player.y
            ema_norm = np.arctan2(emdy, emdx) / np.pi
            emd_norm = self.player.distance_to(self.enemy.missile) / self.MAX_D

        self.state = (
            px_norm,
            py_norm,
            ta_norm,
            td_norm,
            enemy_visible,
            ea_norm,
            ed_norm,
            enemy_missile_visible,
            ema_norm,
            emd_norm
        )

        self.step_count += 1
        truncated = (self.step_count >= self.STEP_LIMIT)

        reward = sum(step_info["dreward"])
        return np.array(self.state, dtype = np.float64), reward, terminated, truncated, step_info

    # Environment reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)

        # Reset the state
        self.counterfactual_trajectories = []
        self.prop_cfact_trajectories = []
        self.reset_start = True
        self.state = None
        self.perform_split_facts = False
        self.draw_actions = False
        self.rewind = False

        self.player = None
        self.enemy = None
        self.target = None

        self.step_count = 0
        self.player_last_dist = np.finfo(np.float64).max
        self.paused = False
        self.player_missile_id_counter = 0

        self.action = None
        self.player_starting_pos = None

        # Initialize the player agent
        self.player = Player(
            self.ORIGIN[0],
            self.ORIGIN[1],
            speed = self.PLAYER_MIN_SPEED,
            angle = 0.5 * np.pi,
            min_speed = self.PLAYER_MIN_SPEED,
            max_speed = self.PLAYER_MAX_SPEED,
            min_turn_rate = self.PLAYER_MIN_TURN_RATE,
            max_turn_rate = self.PLAYER_MAX_TURN_RATE,
            radius = self.PLAYER_RADIUS,
            observation_range = self.PLAYER_OBSERVATION_RANGE,
            prev_move_dir = 1
        )

        # Initialize the enemy somewhere outside the player's detection range
        exy = [self.player.x, self.player.y]
        while self.player.distance_to(exy) < self.player.observation_range:
            exy[0] = np.random.uniform(self.WORLD_AREA[0], self.WORLD_AREA[0] + self.WORLD_AREA[2])
            exy[1] = np.random.uniform(self.WORLD_AREA[1], self.WORLD_AREA[1] + self.WORLD_AREA[3])
        self.enemy = Enemy(
            exy[0],
            exy[1],
            angle = np.random.uniform(-np.pi, np.pi),
            speed = self.ENEMY_MIN_SPEED,
            min_speed = self.ENEMY_MIN_SPEED,
            max_speed = self.ENEMY_MAX_SPEED,
            min_turn_rate = self.ENEMY_MIN_TURN_RATE,
            max_turn_rate = self.ENEMY_MAX_TURN_RATE,
            radius = self.ENEMY_RADIUS,
            observation_range = self.ENEMY_OBSERVATION_RANGE,
        )
        self.enemy.angle = self.enemy.angle_to(self.player)

        # Initialize the target somewhere outside the player's detection range
        txy = [self.player.x, self.player.y]
        while self.player.distance_to(txy) < self.player.observation_range:
            txy[0] = np.random.uniform(self.WORLD_AREA[0] + 45, self.WORLD_AREA[0] + self.WORLD_AREA[2] - 45)
            txy[1] = np.random.uniform(self.WORLD_AREA[1] + 45, self.WORLD_AREA[1] + self.WORLD_AREA[3] - 45)
        self.target = Entity(
            txy[0],
            txy[1],
            radius = self.TARGET_RADIUS
        )

        # Prepare to return the observations in a normalized manner
        px_norm = self.player.x / self.MAX_X
        py_norm = self.player.y / self.MAX_Y
        tdx = self.target.x - self.player.x
        tdy = self.target.y - self.player.y
        ta_norm = np.arctan2(tdy, tdx) / np.pi
        td_norm = self.player.distance_to(self.target) / self.MAX_D

        self.state = (
            px_norm,
            py_norm,
            ta_norm,
            td_norm,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.
        )

        if self.RENDER_MODE == "human":
            self.render()

        self.reset_start = False

        return np.array(self.state, dtype = np.float64), {"dreward" : [0] * self.N_REWARD_COMPONENTS}

    def _get_jet_vertices(self, jet):
        jet_vertices = [
            (
                jet.x + jet.radius * np.cos(jet.angle),
                jet.y + jet.radius * np.sin(jet.angle)
            ),
            (
                jet.x + (0.75 * jet.radius) * np.cos(jet.angle + 2 * np.pi / 3),
                jet.y + (0.75 * jet.radius) * np.sin(jet.angle + 2 * np.pi / 3)
            ),
            (
                jet.x + (0.75 * jet.radius) * np.cos(jet.angle + 4 * np.pi / 3),
                jet.y + (0.75 * jet.radius) * np.sin(jet.angle + 4 * np.pi / 3)
            )
        ]
        return jet_vertices

    # Draw the environment for human rendering mode
    # Note: multiple separate transparent surfaces are required for each
    # transparent shape, or else they will not blend together properly
    # when blitted to the lower surface or directly to the screen
    def render(self):
        if self.RENDER_MODE is None:
            gym.logger.warn(
                "Env render method called without specified render mode"
            )
            return

        # Initialize screen with PyGame
        if self.SCREEN is None:
            pygame.init()
            if self.RENDER_MODE == "human":
                pygame.display.init()
                self.SCREEN = pygame.display.set_mode(self.SCREEN_SIZE)
            else:
                self.SCREEN = pygame.Surface(self.SCREEN_SIZE)

            self.lower_surface   = pygame.Surface(self.SCREEN_SIZE)
            self.pobs_surface    = pygame.Surface(self.SCREEN_SIZE, pygame.SRCALPHA)
            self.eobs_surface    = pygame.Surface(self.SCREEN_SIZE, pygame.SRCALPHA)
            self.zone_surface    = pygame.Surface(self.SCREEN_SIZE, pygame.SRCALPHA)
            self.factual_surface = pygame.Surface(self.SCREEN_SIZE, pygame.SRCALPHA)
            self.factual_surface.fill(self.COLOR_CLEAR_ALPHA)

            if self.CLOCK is None:
                self.CLOCK = pygame.time.Clock()

        # Reset / clear the surfaces each frame
        if not self.perform_split_facts:
            self.lower_surface.fill(self.COLOR_GRAY)
            self.pobs_surface.fill(self.COLOR_CLEAR_ALPHA)
            self.eobs_surface.fill(self.COLOR_CLEAR_ALPHA)
            self.zone_surface.fill(self.COLOR_CLEAR_ALPHA)

            # Draw the target firing zone and the target itself at the bottom of
            # the screen since it is considered a ground target
            pygame.draw.circle(
                self.zone_surface,
                self.COLOR_BLUE_ALPHA,
                (self.target.x, self.target.y),
                self.TARGET_OPENING_RANGE
            )
            self.lower_surface.blit(self.zone_surface, (0, 0))
            # Draw the target
            pygame.draw.circle(
                self.lower_surface,
                self.COLOR_BLUE,
                (self.target.x, self.target.y),
                self.target.radius
            )

            # Draw enemy then player obs. ranges
            if not self.enemy.dead:
                pygame.draw.circle(
                    self.eobs_surface,
                    self.COLOR_RED_ALPHA,
                    (self.enemy.x, self.enemy.y),
                    self.enemy.observation_range
                )
                self.lower_surface.blit(self.eobs_surface, (0, 0))
            pygame.draw.circle(
                self.pobs_surface,
                self.COLOR_GREEN_ALPHA,
                (self.player.x, self.player.y),
                self.player.observation_range
            )
            self.lower_surface.blit(self.pobs_surface, (0, 0))

            # Draw the jets
            for jet in [self.player, self.enemy]:
                if jet == self.player or (jet == self.enemy and not self.enemy.dead):
                    jet_vertices = self._get_jet_vertices(jet)
                    color = self.COLOR_GREEN if jet is self.player else self.COLOR_RED
                    pygame.draw.polygon(self.lower_surface, color, jet_vertices)

            # Draw enemy missile
            if self.enemy.missile is not None:
                pygame.draw.circle(
                    self.lower_surface,
                    self.COLOR_RED,
                    (self.enemy.missile.x, self.enemy.missile.y),
                    self.enemy.missile.radius
                )

            # Draw the player missiles
            for missile in self.player.missiles:
                pygame.draw.circle(
                    self.lower_surface,
                    self.COLOR_GREEN,
                    (missile.x, missile.y),
                    missile.radius
                )

            lower_flip_surf = pygame.transform.flip(self.lower_surface, False, True)
            self.SCREEN.blit(lower_flip_surf, (0, 0))

        # PyGame event handling and timing
        if self.RENDER_MODE == "human":
            if self.paused and self.EVAL_MODE and not self.rewind:
                if self.prop_cfact_trajectories:
                    for idx, point_pair in enumerate(self.prop_cfact_trajectories):
                        pygame.draw.line(
                            self.SCREEN,
                            self.COLOR_WHITE,
                            point_pair[0],
                            point_pair[1],
                            2
                        )
                        if point_pair[2] == self.TRAJ_SHOOT_ENEMY:
                            pygame.draw.circle(
                                self.SCREEN,
                                self.COLOR_RED,
                                point_pair[1],
                                0.35 * self.PLAYER_RADIUS
                            )
                        if point_pair[2] == self.TRAJ_SHOOT_TARGET:
                            pygame.draw.circle(
                                self.SCREEN,
                                self.COLOR_BLUE,
                                point_pair[1],
                                0.35 * self.PLAYER_RADIUS
                            )
#            elif not self.perform_split_facts:
            else:
                self.counterfactual_trajectories = []
                self.prop_cfact_trajectories = []

#            if self.draw_actions:
            if self.perform_split_facts:
                pygame.draw.line(
                    self.SCREEN,
                    self.COLOR_BLACK,
                    (self.player_starting_pos[0], self.MAX_Y - self.player_starting_pos[1]),
                    (self.player.x, self.MAX_Y - self.player.y),
                    2
                )
                if self.action == self.ACTION_SHOOT_ENEMY:
                    pygame.draw.circle(
                        self.SCREEN,
                        self.COLOR_RED,
                        (self.player.x, self.MAX_Y - self.player.y),
                        0.35 * self.PLAYER_RADIUS
                    )
                elif self.action == self.ACTION_SHOOT_TARGET:
                    pygame.draw.circle(
                        self.SCREEN,
                        self.COLOR_BLUE,
                        (self.player.x, self.MAX_Y - self.player.y),
                        0.35 * self.PLAYER_RADIUS
                    )
#                factual_surface_flip = pygame.transform.flip(self.lower_surface, False, True)
#                self.SCREEN.blit(factual_surface_flip, (0, 0))
            for event in pygame.event.get():
                if self.EVAL_MODE and not self.perform_split_facts:
                    if event.type == pygame.KEYDOWN and not self.reset_start:
                        if event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_RETURN:
                            if len(self.counterfactual_trajectories) > 0:
                                self.perform_split_facts = True
#                                self.paused = False
#                            print(f"counterfactual trajectory:\n{self.counterfactual_trajectories}")
                        elif event.key == pygame.K_x and self.prop_cfact_trajectories:
                            tmp = self.prop_cfact_trajectories.pop()
                            if tmp[2] == self.TRAJ_MOVE:
                                ny = round(abs(tmp[1][1] - tmp[0][1]) / (self.player.speed * self.TAU))
                                nx = round(abs(tmp[1][0] - tmp[0][0]) / (self.player.speed * self.TAU))
                                for i in range(max(nx, ny)):
                                    self.counterfactual_trajectories.pop()
                            else:
                                self.counterfactual_trajectories.pop()
                        elif event.key == pygame.K_e or event.key == pygame.K_t:
                            if not self.prop_cfact_trajectories:
                                pp0 = (self.player.x, self.MAX_Y - self.player.y)
                            else:
                                pp0 = (self.prop_cfact_trajectories[-1][1][0], self.prop_cfact_trajectories[-1][1][1])

                            if self.prop_cfact_trajectories:
                                if self.prop_cfact_trajectories[-1][2] == self.TRAJ_SHOOT_ENEMY or self.prop_cfact_trajectories[-1][2] == self.TRAJ_SHOOT_TARGET:
                                    if len(self.prop_cfact_trajectories) == 1:
                                        if self.player.prev_move_dir == self.ACTION_LEFT or self.player.prev_move_dir == self.ACTION_RIGHT:
                                            xmul = [-1, 1][self.player.prev_move_dir - self.ACTION_LEFT]
                                        else:
                                            xmul = 0
                                        if self.player.prev_move_dir == self.ACTION_DOWN or self.player.prev_move_dir == self.ACTION_UP:
                                            ymul = [1, -1][self.player.prev_move_dir - self.ACTION_DOWN]
                                        else:
                                            ymul = 0
                                    else:
                                        if (
                                            self.prop_cfact_trajectories[-2][1][0] == self.prop_cfact_trajectories[-2][0][0] and
                                            self.prop_cfact_trajectories[-2][1][1] == self.prop_cfact_trajectories[-2][0][1]
                                        ):
                                            lkbk = -1
                                        else:
                                            lkbk = -2
                                        if self.prop_cfact_trajectories[lkbk][1][0] == self.prop_cfact_trajectories[lkbk][0][0]:
                                            xmul = 0
                                            ymul = np.sign(self.prop_cfact_trajectories[lkbk][1][1] - self.prop_cfact_trajectories[lkbk][0][1])
                                        elif self.prop_cfact_trajectories[lkbk][1][1] == self.prop_cfact_trajectories[lkbk][0][1]:
                                            xmul = np.sign(self.prop_cfact_trajectories[lkbk][1][0] - self.prop_cfact_trajectories[lkbk][0][0])
                                            ymul = 0
                                else:
                                    xmul = 0
                                    ymul = 0
                            else:
                                xmul = 0
                                ymul = 0

                            dx = self.player.speed * self.TAU * xmul
                            dy = self.player.speed * self.TAU * ymul
                            if abs(dy) > abs(dx):
                                if not self.prop_cfact_trajectories:
                                    pp1 = (self.player.x, self.MAX_Y - self.player.y + dy)
                                else:
                                    pp1 = (self.prop_cfact_trajectories[-1][1][0], self.prop_cfact_trajectories[-1][1][1] + dy)
                            else:
                                if not self.prop_cfact_trajectories:
                                    pp1 = (self.player.x + dx, self.MAX_Y - self.player.y)
                                else:
                                    pp1 = (self.prop_cfact_trajectories[-1][1][0] + dx, self.prop_cfact_trajectories[-1][1][1])

                            if event.key == pygame.K_t:
                                self.counterfactual_trajectories.append(self.ACTION_SHOOT_TARGET)
#                                print("Requires shooting target")
                            else:
                                self.counterfactual_trajectories.append(self.ACTION_SHOOT_ENEMY)
#                                print("Requires shooting enemy")
                            self.prop_cfact_trajectories.append((pp0, pp1, self.TRAJ_SHOOT_TARGET if event.key == pygame.K_t else self.TRAJ_SHOOT_ENEMY))

                        elif event.key == pygame.K_c:
                            self.prop_cfact_trajectories = []
                            self.counterfactual_trajectories = []

                    elif event.type == pygame.MOUSEBUTTONDOWN and not self.reset_start:
                        if event.button == 1:
                            if not self.prop_cfact_trajectories:
                                pp0 = (self.player.x, self.MAX_Y - self.player.y)
                            else:
                                pp0 = (self.prop_cfact_trajectories[-1][1][0], self.prop_cfact_trajectories[-1][1][1])

                            xmul = round((event.pos[0] - pp0[0]) / (self.player.speed * self.TAU))
                            ymul = round((event.pos[1] - pp0[1]) / (self.player.speed * self.TAU))

                            dx = self.player.speed * self.TAU * xmul
                            dy = self.player.speed * self.TAU * ymul
                            if abs(dy) > abs(dx):
                                if not self.prop_cfact_trajectories:
                                    pp1 = (self.player.x, self.MAX_Y - self.player.y + dy)
                                else:
                                    pp1 = (self.prop_cfact_trajectories[-1][1][0], self.prop_cfact_trajectories[-1][1][1] + dy)
                            else:
                                if not self.prop_cfact_trajectories:
                                    pp1 = (self.player.x + dx, self.MAX_Y - self.player.y)
                                else:
                                    pp1 = (self.prop_cfact_trajectories[-1][1][0] + dx, self.prop_cfact_trajectories[-1][1][1])

                            if abs(xmul) > 0 or abs(ymul) > 0:
                                if abs(dy) > abs(dx):
#                                    print(f"Requires going {abs(ymul)} steps {'up' if ymul < 0 else 'down'}")
                                    self.counterfactual_trajectories += [self.ACTION_UP if ymul < 0 else self.ACTION_DOWN] * abs(ymul)
                                else:
#                                    print(f"Requires going {abs(xmul)} steps {'left' if xmul < 0 else 'right'}")
                                    self.counterfactual_trajectories += [self.ACTION_LEFT if xmul < 0 else self.ACTION_RIGHT] * abs(xmul)
                                self.prop_cfact_trajectories.append((pp0, pp1, self.TRAJ_MOVE))

            keys = pygame.key.get_pressed()
            self.rewind = self.EVAL_MODE and self.paused and not self.perform_split_facts and keys[pygame.K_r]

            pygame.display.flip()
            self.CLOCK.tick(self.metadata["render_fps"])

    # Close the environment
    def close(self):
        if self.SCREEN is not None:
            pygame.display.quit()
            pygame.quit()

#        self.counterfactual_trajectories = []
#        self.prop_cfact_trajectories = []
#        self.reset_start = True
#        self.state = None
#        self.perform_split_facts = False
#        self.draw_actions = False
#
#        self.player = None
#        self.enemy = None
#        self.target = None
#
#        self.step_count = 0
#        self.player_last_dist = np.finfo(np.float64).max
#        self.paused = False
#        self.player_missile_id_counter = 0
#
#        self.lower_surface   = None
#        self.pobs_surface    = None
#        self.eobs_surface    = None
#        self.zone_surface    = None
#        self.factual_surface = None
#
#        self.action = None
#        self.player_starting_pos = None

    def get_state(self):
        state_dict = {
            "state"             : deepcopy(self.state),
            "player"            : deepcopy(self.player),
            "enemy"             : deepcopy(self.enemy),
            "target"            : deepcopy(self.target),
            "step_count"        : self.step_count,
            "player_last_dist"  : self.player_last_dist,
            "player_missile_id" : self.player_missile_id_counter
        }
        return state_dict

    def set_state(self, state_dict):
        self.state = deepcopy(state_dict["state"])
        self.player = deepcopy(state_dict["player"])
        self.enemy = deepcopy(state_dict["enemy"])
        self.target = deepcopy(state_dict["target"])
        self.step_count = state_dict["step_count"]
        self.player_last_dist = state_dict["player_last_dist"]
        self.player_missile_id_counter = state_dict["player_missile_id"]

# More generic entity class for the game
class Entity():
    def __init__(self, x, y, speed = 0, angle = 0, radius = 1, fov = 0):
        self.x = x
        self.y = y

        if speed < 0:
            raise ValueError(
                "Entity speed must be >= 0"
            )
        self.speed = speed

        self.angle = np.arctan2(np.sin(angle), np.cos(angle))

        if radius <= 0.:
            raise ValueError(
                "Entity 'radius' argument must be greater than 0"
            )
        self.radius = radius

        if fov < 0:
            raise ValueError(
                "Jet fov must be >= 0"
            )
        self.fov = fov

    # Check if entity is in an area defined by lower-left coordinates and
    # the width and height
    def in_area(self, blx, bly, width, height):
        if (
            blx < self.x and
            bly < self.y and
            self.x < blx + width and
            self.y < bly + height
        ):
            return True
        else:
            return False

    # Is a point / entity in the field of view of this entity
    def in_fov(self, other):
        if not (
            isinstance(other, Entity) or
            (isinstance(other, (list, tuple)) and len(other) == 2)
        ):
            raise TypeError(
                "Entity 'in_fov' function must take another entity or an (x, y) coordinate pair"
            )

        return np.abs(self.angle_to(other)) < 0.5 * self.fov

    # Check for collision with other entity
    def collides_with(self, other):
        distance = self.distance_to(other)

        return distance < (self.radius + other.radius)

    # Get distance to point or other entity
    def distance_to(self, other):
        if isinstance(other, Entity):
            dx = other.x - self.x
            dy = other.y - self.y
        elif isinstance(other, (list, tuple)) and len(other) == 2:
            dx = other[0] - self.x
            dy = other[1] - self.y
        else:
            raise TypeError(
                "Entity 'distance_to' function must take another entity or an (x, y) coordinate pair"
            )

        return np.sqrt(dx ** 2 + dy ** 2, dtype = np.float64)

    def move(self, tau):
        # Update position
        self.x += self.speed * np.cos(self.angle) * tau
        self.y += self.speed * np.sin(self.angle) * tau

    # Get angle to point or other entity
    def angle_to(self, other):
        if isinstance(other, Entity):
            dx = other.x - self.x
            dy = other.y - self.y
        elif isinstance(other, (list, tuple)) and len(other) == 2:
            dx = other[0] - self.x
            dy = other[0] - self.y
        else:
            raise TypeError(
                "Entity 'angle_to' function must take another entity or an (x, y) coordinate pair"
            )

        abs_angle = np.arctan2(dy, dx)
        rel_angle = abs_angle - self.angle

        return np.arctan2(np.sin(rel_angle), np.cos(rel_angle), dtype = np.float64)

# Missiles have a limited range, and player missiles check for firing
# within the target zone
class Missile(Entity):
    def __init__(self, x, y, speed, angle, radius, range, fov = 0, id = None, fired_in_zone = False, target = None):
        super().__init__(x, y, speed, angle, radius, fov)        
        self.origin = (x, y)
        self.id = id
        self.fired_in_zone = fired_in_zone
        self.target = target

        if range <= 0:
            raise ValueError(
                "Missile range must be at least > 0"
            )
        self.range = range

# Jets have a more complicated move function
class Jet(Entity):
    def __init__(
        self,
        x,
        y,
        speed,
        angle,
        radius,
        min_speed,
        max_speed,
        min_turn_rate,
        max_turn_rate,
        observation_range,
        fov = 0,
    ):
        super().__init__(x, y, speed, angle, radius, fov)

        if min_speed < 0 or max_speed < 0:
            raise ValueError(
                "Jet min_speed and max_speed must be >= 0"
            )
        if not min_speed <= max_speed:
            raise ValueError(
                "Jet min_speed must be <= max_speed"
            )
        self.min_speed = min_speed
        self.max_speed = max_speed

        if min_turn_rate < 0 or max_turn_rate < 0:
            raise ValueError(
                "Jet min_turn_rate and max_turn_rate must be >= 0"
            )
        if not min_turn_rate <= max_turn_rate:
            raise ValueError(
                "Jet min_turn_rate must be <= max_turn_rate"
            )
        self.min_turn_rate = min_turn_rate
        self.max_turn_rate = max_turn_rate

        if observation_range <= radius:
            raise ValueError(
                "Jet observation_range must be at least >= radius"
            )
        self.observation_range = observation_range

        self.missile = None
        self.guess_position = None

    def get_turn_direction_to(self, other):
        if not (
            isinstance(other, Entity) or
            (isinstance(other, (list, tuple)) and len(other) == 2)
        ):
            raise TypeError(
                "Jet 'get_turn_direction_to' function must take another entity or an (x, y) coordinate pair"
            )

        da = self.angle_to(other)
        if da < 0:
            return -1
        elif 0 < da:
            return 1
        else:
            return 0

    def move(self, tau, acceleration = 0, turn_direction = 0):
        # Get the turn rate
        if turn_direction != 0:
            dv = self.max_speed - self.min_speed
            if dv == 0:
                turn_rate = self.min_turn_rate
            else:
                dw = self.max_turn_rate - self.min_turn_rate
                turn_rate = (
                    (self.max_speed - self.speed) / dv * dw + self.min_turn_rate
                )
                turn_rate = np.clip(turn_rate, self.min_turn_rate, self.max_turn_rate)
        else:
            turn_rate = 0

        # Update the angle
        self.angle += turn_direction * turn_rate * tau
        self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))

        # Accelerate
        self.speed += acceleration * tau
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)

        # Update position
        self.x += self.speed * np.cos(self.angle) * tau
        self.y += self.speed * np.sin(self.angle) * tau

# Player can have multiple missiles out
class Player(Jet):
    def __init__(
        self,
        x,
        y,
        speed,
        angle,
        radius,
        min_speed,
        max_speed,
        min_turn_rate,
        max_turn_rate,
        observation_range,
        prev_move_dir,
        fov = 0,
    ):
        super().__init__(
            x,
            y,
            speed,
            angle,
            radius,
            min_speed,
            max_speed,
            min_turn_rate,
            max_turn_rate,
            observation_range,
            fov
        )
        self.missiles = []
        self.prev_move_dir = prev_move_dir

    def move(self, tau, action):
        if action == 0:
            self.y -= self.speed * tau
        elif action == 1:
            self.y += self.speed * tau
        elif action == 2:
            self.x -= self.speed * tau
        else:
            self.x += self.speed * tau

# Enemies have a single missile
class Enemy(Jet):
    def __init__(
        self,
        x,
        y,
        speed,
        angle,
        radius,
        min_speed,
        max_speed,
        min_turn_rate,
        max_turn_rate,
        observation_range,
        fov = 0,
    ):
        super().__init__(
            x,
            y,
            speed,
            angle,
            radius,
            min_speed,
            max_speed,
            min_turn_rate,
            max_turn_rate,
            observation_range,
            fov
        )
        self.missile = None
        self.dead = False
