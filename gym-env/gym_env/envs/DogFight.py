import gymnasium as gym
import numpy as np
import pygame

from typing import Optional, Union

class DogFightEnv(gym.Env):
    metadata = {
        "render_fps" : 15,
        "render_modes" : ["human", "rgb_array"]
    }

    # Initialize the environment
    def __init__(self, render_mode = None):
        # Episode time & time limit
        self.tau = 1 / self.metadata["render_fps"]
        # Time limit of 60 seconds per episode
        max_episode_seconds = 60
        self.step_count = 0
        self.step_limit = max_episode_seconds * self.metadata["render_fps"]

        # Rendering settings
        self.render_mode = render_mode
        self.screen_size = (800, 800)
        self.screen = None
        self.clock = None
        self.color_white  = (255, 255, 255, 255)
        self.color_gray   = ( 64,  64,  64, 255)
        self.color_black  = (  0,   0,   0, 255)
        self.color_clear  = (  0,   0,   0,   0)
        self.color_green  = (  0, 255,   0, 255)
        self.color_red    = (255,   0,   0, 255)
        self.color_teal   = (  0, 255, 255, 255)
        self.color_blue   = (  0,   0, 255, 255)
        self.color_orange = (255, 165,   0, 255)

        # Game area settings (for out of bounds checks and other things)
        self.min_x = 0
        self.max_x = self.screen_size[0]
        self.min_y = 0
        self.max_y = self.screen_size[1]
        self.origin = (0.5 * self.max_x, 0.5 * self.max_y)
        self.world_area = (0, 0, self.max_x, self.max_y)

        # Turn directions:
        # Straight - no angle change so it is 0
        # Left - increase in angle so +1
        # Right - decrease in angle to -1
        self.STRAIGHT =  0
        self.LEFT     =  1
        self.RIGHT    = -1

        # Player properties
        self.player_radius = 10
        # Player min speed: 10 seconds to cross game area
        # Player max speed:  5 seconds to cross game area
        self.player_min_speed = self.max_x / 10
        self.player_max_speed = self.max_x /  5
        # Player min acceleration: 0.25 seconds from max to min speed
        # Player max acceleration: 2.50 seconds from min to max speed
        self.player_min_acceleration = (
            -(self.player_max_speed - self.player_min_speed) / 0.25
        )
        self.player_max_acceleration = (
             (self.player_max_speed - self.player_min_speed) / 2.50
        )
        # Player min turn rate: 4.0 seconds to do a 180
        # Player max turn rate: 2.0 seconds to do a 180
        self.player_min_turn_rate = np.pi / 4.0
        self.player_max_turn_rate = np.pi / 2.0
        # Player observation radius is 1/4 the width of the game area
        self.player_observation_range = 0.35 * self.max_x
        self.player_last_dist = None

        # Player missile properties
        # Track the missiles for properly assigning delayed rewards outside env
        self.player_missile_id_counter = 0
        self.player_missile_radius = 3
        self.player_missile_speed = 3.0 * self.player_max_speed
        self.player_missile_range = 3 * self.player_observation_range
        # Bounds of player missile's random angle offset
        self.player_missile_angle_offset = 0.04

        # Enemy properties
        self.enemy_radius = 10
        # Enemy is 0.8 x player speed
        self.enemy_min_speed = 0.80 * self.player_min_speed
        self.enemy_max_speed = 0.80 * self.player_max_speed
        # Enemy min acceleration: 0.25 seconds from max to min speed
        # Enemy max acceleration: 2.50 seconds from min to max speed
        self.enemy_min_acceleration = (
            -(self.enemy_max_speed - self.enemy_min_speed) / 0.25
        )
        self.enemy_max_acceleration = (
             (self.enemy_max_speed - self.enemy_min_speed) / 2.50
        )
        # Enemy min turn rate: 4.0 seconds to do a 180
        # Enemy max turn rate: 2.0 seconds to do a 180
        self.enemy_min_turn_rate = np.pi / 4.0
        self.enemy_max_turn_rate = np.pi / 2.0
        # Enemy observation radius is 3/4 of the player's observation radius
        self.enemy_observation_range = 1.25 * self.player_observation_range
        # 0.35 radians (~20 degree firing FOV for the enemy)
        self.enemy_firing_fov = 0.35

        # Enemy missile properties
        self.enemy_missile_radius = 3
        self.enemy_missile_speed = 0.250 * self.player_missile_speed
        self.enemy_missile_range = 3 * self.enemy_observation_range
        # Bounds of enemy missile's random angle offset
        self.enemy_missile_angle_offset = 0.04

        # Target properties
        self.target_radius = 30
        # Range from which player missiles will destroy the target
        self.target_opening_range = 0.35 * self.player_missile_range

        # Reward definitions
        self.reward_missile_miss = -0.75 * self.tau
        self.reward_missile_hit_enemy = 50
        self.reward_missile_hit_target = 100
        self.reward_player_collides_with_enemy = -500
        self.reward_player_leaves_game = -100
        self.reward_time_penalty = -self.tau
        self.reward_approach_target = abs(self.reward_time_penalty)

        # Environment observation space:
        #  0.) Jet absolute x position
        #  1.) Jet absolute y position
        #  2.) Target relative x position
        #  3.) Target relative y position
        #  4.) Target distance from player
        #  5.) Enemy visibility
        #  6.) Enemy relative x position
        #  7.) Enemy relative y position
        #  8.) Enemy distance from player
        #  9.) Enemy angle to player
        # 10.) Enemy bullet visibility
        # 11.) Enemy bullet relative x position
        # 12.) Enemy bullet relative y position
        # 13.) Enemy bullet distance from player
        # 14.) Enemy bullet angle to player
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
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
                np.finfo(np.float64).min,
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
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
            ]),
            dtype = np.float64
        )

        # Environment action space:
        # 0.) Forward
        # 1.) Backward
        # 2.) Left
        # 3.) Right
        # 4.) Shoot target
        # 5.) Shoot enemy
        self.action_space = gym.spaces.Discrete(6)

    # Take a single simulation step in the environment
    def step(self, action):
        # Additional dictionary to track missile information for assigning
        # delayed rewards to the proper step (done outside the environment)
        player_missile_step_info = {
            "shoot_act"    : False, # missile related action this step
            "shoot_id"     : None,  # the id of the missile shot during this step
            "hit_ids"      : [],    # the ids of the missiles that hit this step
            "miss_ids"     : [],    # the ids of the missiles that missed this step
            "hit_rewards"  : [],    # rewards for the missiles that hit
            "miss_rewards" : [],    # rewards (penalties) for the missiles that missed
        }

        p_shot_at_nothing = False
        if action == 4:
            if self.player.distance_to(self.target) <= self.target_opening_range:
                fired_in_zone = True
            else:
                fired_in_zone = False
            angle = self.player.angle_to(self.target)
            new_missile = Missile(
                self.player.x,
                self.player.y,
                self.player_missile_speed,
                self.player.angle + angle,
                self.player_missile_radius,
                self.player_missile_range,
                id = self.player_missile_id_counter,
                fired_in_zone = fired_in_zone,
                target = self.target
            )
            self.player.angle = new_missile.angle
            self.player.missiles.append(new_missile)
            player_missile_step_info["shoot_id"] = self.player_missile_id_counter
            player_missile_step_info["shoot_act"] = True
            self.player_missile_id_counter += 1
        elif action == 5:
            if not self.enemy.dead and self.player.distance_to(self.enemy) <= self.player.observation_range:
                fired_in_zone = False
                angle = self.player.angle_to(self.enemy)
                new_missile = Missile(
                    self.player.x,
                    self.player.y,
                    self.player_missile_speed,
                    self.player.angle + angle,
                    self.player_missile_radius,
                    self.player_missile_range,
                    id = self.player_missile_id_counter,
                    fired_in_zone = fired_in_zone,
                    target = self.enemy
                )
                self.player.angle = new_missile.angle
                self.player.missiles.append(new_missile)
                player_missile_step_info["shoot_id"] = self.player_missile_id_counter
                player_missile_step_info["shoot_act"] = True
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
                elif not self.enemy.in_area(*(self.world_area)):
                    roff = np.random.uniform(-100, 100), np.random.uniform(-100, 100)
                    enemy_turn_direction = self.enemy.get_turn_direction_to((self.origin[0] + roff[0], self.origin[1] + roff[1]))
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
                        self.enemy_missile_speed,
                        self.enemy.angle + self.enemy.angle_to(self.player),
                        self.enemy_missile_radius,
                        self.enemy_missile_range
                    )

        # Move the player
        p_oob = False
        old_pos = self.player.x, self.player.y
        if action == 4 or action == 5:
            self.player.move(self.tau, self.player.prev_move_dir)
        else:
            self.player.move(self.tau, action)
            self.player.prev_move_dir = action
        if not self.player.in_area(*(self.world_area)):
            p_oob = True
            self.player.x, self.player.y = old_pos[0], old_pos[1]

        # Move the enemy and its missile
        if not self.enemy.dead:
            self.enemy.move(self.tau, turn_direction = enemy_turn_direction)
        if self.enemy.missile is not None:
            self.enemy.missile.move(self.tau)

        # ================================================================================
        # Player Missile Movement & Delayed Reward Handling
        # ================================================================================
        terminated = False
        for missile in self.player.missiles[:]:
            missile.move(self.tau)
            # Missiles going out of bounds are considered a miss
            if not missile.in_area(*(self.world_area)):
                player_missile_step_info["miss_ids"].append(missile.id)
                player_missile_step_info["shoot_act"] = True
                player_missile_step_info["miss_rewards"].append(self.reward_missile_miss)
                self.player.missiles.remove(missile)
            # Missiles that reach their maximum range are considered a miss
            elif missile.range < missile.distance_to(missile.origin):
                player_missile_step_info["miss_ids"].append(missile.id)
                player_missile_step_info["shoot_act"] = True
                player_missile_step_info["miss_rewards"].append(self.reward_missile_miss)
                self.player.missiles.remove(missile)
            # Missile collides with enemy
            elif not self.enemy.dead and missile.collides_with(self.enemy) and missile.target == self.enemy:
                player_missile_step_info["hit_ids"].append(missile.id)
                player_missile_step_info["shoot_act"] = True
                player_missile_step_info["hit_rewards"].append(self.reward_missile_hit_enemy)
                self.enemy.dead = True
                self.player.missiles.remove(missile)
            # Missile collides with target (terminating condition)
            elif missile.collides_with(self.target) and missile.fired_in_zone == True and missile.target == self.target:
                player_missile_step_info["hit_ids"].append(missile.id)
                player_missile_step_info["shoot_act"] = True
                player_missile_step_info["hit_rewards"].append(self.reward_missile_hit_target)
                self.player.missiles.remove(missile)
                terminated = True
                break
        # ================================================================================

        # ================================================================================
        # Immediate Rewards
        # ================================================================================
        reward, truncated = 0, False

        # The player gets hit by an enemy bullet (terminating)
        p_hit_by_missile = False
        if self.enemy.missile is not None:
            if self.enemy.missile.range < self.enemy.missile.distance_to(self.enemy.missile.origin):
                self.enemy.missile = None
            elif self.player.collides_with(self.enemy.missile):
                p_hit_by_missile = True
        # The player collides with the enemy (terminating condition)
        if p_hit_by_missile or (not self.enemy.dead and self.enemy.collides_with(self.player)):
            reward += self.reward_player_collides_with_enemy
            terminated = True
        else:
            if p_oob:
                reward += self.reward_player_leaves_game
            # Constant negative reward to encourage the agent to finish sooner
            if p_shot_at_nothing:
                reward += self.reward_missile_miss
            reward += self.reward_time_penalty
            dist_to_target = self.player.distance_to(self.target)
            if dist_to_target < self.player_last_dist:
                reward += self.reward_approach_target
            self.player_last_dist = dist_to_target
        # ================================================================================

        if self.render_mode == "human":
            self.render()

        # Prepare to return the observations in a normalized manner
        if self.enemy.dead or self.player.distance_to(self.enemy) > self.player.observation_range:
            enemy_visible = 0.
            ex_norm = 0.
            ey_norm = 0.
            ed_norm = 0.
            ea_norm = 0.
        else:
            enemy_visible = 1.
            ex_norm = (self.enemy.x - self.player.x) / self.player.observation_range
            ey_norm = (self.enemy.y - self.player.y) / self.player.observation_range
            ed_norm = self.player.distance_to(self.enemy) / self.player_observation_range
            ea_norm = self.enemy.angle_to(self.player) / np.pi
        if (
            self.enemy.missile is not None and
            self.player.distance_to(self.enemy.missile) <= self.player.observation_range
        ):
            enemy_missile_visible = 1.
            emx_norm = (self.enemy.missile.x - self.player.x) / self.player.observation_range
            emy_norm = (self.enemy.missile.y - self.player.y) / self.player.observation_range
            emd_norm = self.player.distance_to(self.enemy.missile) / self.player_observation_range
            ema_norm = self.enemy.missile.angle_to(self.player) / np.pi
        else:
            enemy_missile_visible = 0.
            emx_norm = 0.
            emy_norm = 0.
            emd_norm = 0.
            ema_norm = 0.
        px_norm = self.player.x / self.max_x
        py_norm = self.player.y / self.max_y
        tx_norm = (self.target.x - self.player.x) / self.max_x
        ty_norm = (self.target.y - self.player.y) / self.max_y
        max_d = np.sqrt(self.max_x**2 + self.max_y**2)
        td_norm = self.player.distance_to(self.target) / max_d

        self.state = (
            px_norm,
            py_norm,
            tx_norm,
            ty_norm,
            td_norm,
            enemy_visible,
            ex_norm,
            ey_norm,
            ed_norm,
            ea_norm,
            enemy_missile_visible,
            emx_norm,
            emy_norm,
            emd_norm,
            ema_norm
        )

        # Check if we reached the maximum episode time limit and terminate if so
        self.step_count += 1
        if self.step_count >= self.step_limit:
            truncated = True

        return np.array(self.state, dtype = np.float64), reward, terminated, truncated, player_missile_step_info

    # Environment reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)
        self.step_count = 0

        # Initialize the player agent
        self.player = Player(
            self.origin[0],
            self.origin[1],
            speed = self.player_min_speed,
            angle = 0.5 * np.pi,
            min_speed = self.player_min_speed,
            max_speed = self.player_max_speed,
            min_turn_rate = self.player_min_turn_rate,
            max_turn_rate = self.player_max_turn_rate,
            radius = self.player_radius,
            observation_range = self.player_observation_range,
            prev_move_dir = 1
        )

        # Initialize the enemy somewhere outside the player's detection range
        exy = [self.player.x, self.player.y]
        while self.player.distance_to(exy) < self.player.observation_range:
            exy[0] = np.random.uniform(self.world_area[0], self.world_area[0] + self.world_area[2])
            exy[1] = np.random.uniform(self.world_area[1], self.world_area[1] + self.world_area[3])
        self.enemy = Enemy(
            exy[0],
            exy[1],
            angle = np.random.uniform(-np.pi, np.pi),
            speed = self.enemy_min_speed,
            min_speed = self.enemy_min_speed,
            max_speed = self.enemy_max_speed,
            min_turn_rate = self.enemy_min_turn_rate,
            max_turn_rate = self.enemy_max_turn_rate,
            radius = self.enemy_radius,
            observation_range = self.enemy_observation_range,
            fov = self.enemy_firing_fov
        )
        self.enemy.angle = self.enemy.angle_to(self.player)

        # Initialize the target somewhere outside the player's detection range
        txy = [self.player.x, self.player.y]
        while self.player.distance_to(txy) < self.player.observation_range:
            txy[0] = np.random.uniform(self.world_area[0], self.world_area[0] + self.world_area[2])
            txy[1] = np.random.uniform(self.world_area[1], self.world_area[1] + self.world_area[3])
        self.target = Entity(
            txy[0],
            txy[1],
            radius = self.target_radius
        )

        # Normalized initial state observation preparation
        px_norm = self.player.x / self.max_x
        py_norm = self.player.y / self.max_y
        tx_norm = (self.target.x - self.player.x) / self.max_x
        ty_norm = (self.target.y - self.player.y) / self.max_y
        max_d = np.sqrt(self.max_x**2 + self.max_y**2)
        td_norm = self.player.distance_to(self.target) / max_d

        self.state = (
            px_norm,
            py_norm,
            tx_norm,
            ty_norm,
            td_norm,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        )

        self.player_last_dist = self.player.distance_to(self.target)

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype = np.float64), {}

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
        if self.render_mode is None:
            gym.logger.warn(
                "Env render method called without specified render mode"
            )
            return

        # Initialize screen with PyGame
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(self.screen_size)
            else:
                self.screen = pygame.Surface(self.screen_size)
            self.lower_surface = pygame.Surface(self.screen_size)
            self.lower_surface = pygame.transform.flip(self.lower_surface, flip_x = True, flip_y = False)
            self.shadow_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            self.pobs_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            self.zone_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            self.tghost_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Reset / clear the surfaces each frame
        self.lower_surface.fill(self.color_black)
        self.shadow_surface.fill(self.color_black)
        self.zone_surface.fill(self.color_clear)
        self.tghost_surface.fill(self.color_black)
        self.pobs_surface.fill(self.color_clear)

        # Draw the target firing zone and the target itself at the bottom of
        # the screen since it is considered a ground target
        pygame.draw.circle(
            self.zone_surface,
            self.color_gray,
            (self.target.x, self.target.y),
            self.target_opening_range
        )
        self.lower_surface.blit(self.zone_surface, (0, 0))
        # Draw the target
        pygame.draw.circle(
            self.lower_surface,
            self.color_blue,
            (self.target.x, self.target.y),
            self.target.radius
        )

        # Draw the jets
        for jet in [self.player, self.enemy]:
            if jet == self.player or (jet == self.enemy and not self.enemy.dead):
                jet_vertices = self._get_jet_vertices(jet)
                color = self.color_green if jet is self.player else self.color_red
                pygame.draw.polygon(self.lower_surface, color, jet_vertices)

        # Draw enemy missile
        if self.enemy.missile is not None:
            pygame.draw.circle(
                self.lower_surface,
                self.color_orange,
                (self.enemy.missile.x, self.enemy.missile.y),
                self.enemy.missile.radius
            )

        # Draw the "shadow" to simulate the player's limited observation range
        pygame.draw.circle(self.shadow_surface, self.color_gray, (self.target.x, self.target.y), self.target_opening_range)
        pygame.draw.circle(self.shadow_surface, self.color_blue, (self.target.x, self.target.y), self.target.radius)
        pygame.draw.circle(self.pobs_surface, self.color_black, (self.player.x, self.player.y), self.player_observation_range)
        self.shadow_surface.blit(self.pobs_surface, (0, 0), special_flags = pygame.BLEND_RGBA_SUB)
        self.lower_surface.blit(self.shadow_surface, (0, 0))

        # Draw the player missiles
        for missile in self.player.missiles:
            pygame.draw.circle(
                self.lower_surface,
                self.color_white,
                (missile.x, missile.y),
                missile.radius
            )

        self.screen.blit(self.lower_surface, (0, 0))

        # PyGame event handling and timing
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    # Close the environment
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

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
            self.y += self.speed * tau
        elif action == 1:
            self.y -= self.speed * tau
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
