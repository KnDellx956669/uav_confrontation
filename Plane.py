import time
from stable_baselines3 import DQN
import pygame
import math
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# 初始化 Pygame
pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("飞机游戏")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


class Plane:
    def __init__(self, x, y, color=GREEN):
        self.x = x
        self.y = y
        self.angle = random.randint(0, 360)
        self.speed = 5
        self.color = color

    def draw(self):
        pygame.draw.polygon(screen, self.color, self.get_points())

    def get_points(self):
        length = 20
        wing_span = 10
        points = [
            (
            self.x + math.cos(math.radians(self.angle)) * length, self.y - math.sin(math.radians(self.angle)) * length),
            (self.x + math.cos(math.radians(self.angle - 150)) * wing_span,
             self.y - math.sin(math.radians(self.angle - 150)) * wing_span),
            (self.x + math.cos(math.radians(self.angle + 150)) * wing_span,
             self.y - math.sin(math.radians(self.angle + 150)) * wing_span)
        ]
        return points

    def move_forward(self):
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y -= math.sin(math.radians(self.angle)) * self.speed
        if self.x < 0:
            self.x = 0
            self.angle = random.randint(0, 360)
        elif self.x > SCREEN_WIDTH:
            self.x = SCREEN_WIDTH
            self.angle = random.randint(0, 360)
        if self.y < 0:
            self.y = 0
            self.angle = random.randint(0, 360)
        elif self.y > SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT
            self.angle = random.randint(0, 360)

    def rotate(self, direction):
        self.angle += direction * 5
        self.angle %= 360

    def detect_enemy(self, enemies):
        fov = 60  # 前方视野60度
        detect_radius = 100
        detected_enemies = []

        for enemy in enemies:
            dx = enemy.x - self.x
            dy = self.y - enemy.y
            distance = math.hypot(dx, dy)
            if distance <= detect_radius:
                angle_to_enemy = math.degrees(math.atan2(dy, dx)) % 360
                if abs((self.angle - angle_to_enemy + 180 + 360) % 360 - 180) < fov / 2:
                    detected_enemies.append(enemy)

        return detected_enemies

    def random_move(self):
        self.rotate(random.choice([-1, 0, 1]))
        self.move_forward()



class PlaneEnv(gym.Env):
    def __init__(self):
        super(PlaneEnv, self).__init__()
        self.players = [Plane(SCREEN_WIDTH // 4 * (i + 1), SCREEN_HEIGHT // 2, color=GREEN) for i in range(3)]
        self.enemies = [Plane(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), color=RED) for _ in
                        range(5)]

        # 状态空间: 每架玩家飞机的位置和角度，敌机的位置
        # 观测维度 = 3（每架飞机）* 3 + 5（敌机）* 3 = 24
        self.observation_space = spaces.Box(low=0, high=1, shape=(24,), dtype=np.float32)
        # 动作空间: 0 - 无操作，1 - 向左转，2 - 向右转，3 - 向前移动
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.players = [Plane(SCREEN_WIDTH // 4 * (i + 1), SCREEN_HEIGHT // 2, color=GREEN) for i in range(3)]
        self.enemies = [Plane(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), color=RED) for _ in
                        range(5)]
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for player in self.players:
            obs.extend([player.x / SCREEN_WIDTH, player.y / SCREEN_HEIGHT, player.angle / 360])
        for enemy in self.enemies:
            obs.extend([enemy.x / SCREEN_WIDTH, enemy.y / SCREEN_HEIGHT, enemy.angle / 360])

        # 用零填充到 24 的长度
        while len(obs) < 24:
            obs.extend([0, 0, 0])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        for i, player in enumerate(self.players):
            if action == 1:
                player.rotate(-1)
            elif action == 2:
                player.rotate(1)
            elif action == 3:
                player.move_forward()

        for enemy in self.enemies:
            enemy.random_move()

        reward = 0
        for player in self.players:
            detected_enemies = player.detect_enemy(self.enemies)
            for enemy in detected_enemies:
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
                    reward += 1
                    detected_players = enemy.detect_enemy(self.players)
                    if player in detected_players:
                        reward -= 0.5


        done = len(self.enemies) == 0

        return self._get_obs(), reward, done, False, {}

    def render(self):
        screen.fill(BLACK)
        for player in self.players:
            player.draw()
        for enemy in self.enemies:
            enemy.draw()
        pygame.display.flip()

    def close(self):
        pygame.quit()


def main():
    env = PlaneEnv()
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs, _ = env.reset()
    for _ in range(1000000):
        time.sleep(0.005)
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()
        if dones:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
