import pygame
import random
import numpy as np
from pygame.math import Vector2

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

class Snake:
    def __init__(self, speed=10, length=2):
        self.speed = speed
        self.length = length
        self.body = [Vector2(10, 10)]
        self.head = self.body[0]
        self.dir = Vector2(1, 0)  # Direction to start
        self.food = None

    def move(self):
        new_head = self.head + self.dir
        self.body.insert(0, new_head)
        if len(self.body) > self.length:
            self.body.pop()

    def render(self, screen):
        for segment in self.body:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(segment.x * 10, segment.y * 10, 10, 10))

    def danger(self, head_r, head_c):
        return (
            head_r >= SCREEN_WIDTH // 10 or head_r < 0 or head_c >= SCREEN_HEIGHT // 10 or head_c < 0
            or Vector2(head_r, head_c) in self.body[:-1]
        )

    def __repr__(self):
        return f"Snake(speed={self.speed}, length={self.length}, direction={self.dir})"


class Food:
    def __init__(self):
        self.position = Vector2(random.randint(0, SCREEN_WIDTH // 10 - 1), random.randint(0, SCREEN_HEIGHT // 10 - 1))

    def spawn(self):
        self.position = Vector2(random.randint(0, SCREEN_WIDTH // 10 - 1), random.randint(0, SCREEN_HEIGHT // 10 - 1))

    def render(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(self.position.x * 10, self.position.y * 10, 10, 10))


class Agent(Snake):
    def __init__(self, speed=10, length=2):
        super().__init__(speed, length)
        self.q_table = {}
        self.epsilon = 1.0  # Start with high exploration
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon_decay = 0.995  # Epsilon decay factor
        self.epsilon_min = 0.1  # Minimum epsilon value to stop exploration
        self.actions = [Vector2(-1, 0), Vector2(1, 0), Vector2(0, -1), Vector2(0, 1)]  # Left, Right, Up, Down
        self.dir = Vector2(1, 0)  # Initialize direction to the right

    def decide(self, env):
        state = self.state_representation(env)
        action = self.choose_action(state)
        self.dir = action
        return action

    def state_representation(self, env):
        food = env.food
        state = []

        # Direction-based features
        state.append(int(self.dir == Vector2(-1, 0)))  # Left
        state.append(int(self.dir == Vector2(1, 0)))   # Right
        state.append(int(self.dir == Vector2(0, -1)))  # Up
        state.append(int(self.dir == Vector2(0, 1)))   # Down

        # Food relative position features
        head_r, head_c = self.head.x, self.head.y  # Assuming head has x, y attributes
        food_r, food_c = food.position.x, food.position.y  # Assuming food has position.x and position.y

        state.append(int(food_r < head_r))  # Food is left of head
        state.append(int(food_r > head_r))  # Food is right of head
        state.append(int(food_c < head_c))  # Food is above head
        state.append(int(food_c > head_c))  # Food is below head

        # Unsafe moves
        state.append(int(self.danger(head_r + 1, head_c)))  # Check if moving down is unsafe
        state.append(int(self.danger(head_r - 1, head_c)))  # Check if moving up is unsafe
        state.append(int(self.danger(head_r, head_c + 1)))  # Check if moving right is unsafe
        state.append(int(self.danger(head_r, head_c - 1)))  # Check if moving left is unsafe

        return tuple(state)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore: choose a random action
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(self.actions))  # Initialize Q-values if not present
            return self.actions[np.argmax(self.q_table[state])]  # Choose the action with the highest Q-value

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        action_idx = self.actions.index(action)

        # Q-learning update rule
        max_future_q = np.max(self.q_table[next_state])  # Maximum future reward
        current_q = self.q_table[state][action_idx]
        
        # Update Q-value with a learning rate applied to the difference
        self.q_table[state][action_idx] += self.alpha * (reward + self.gamma * max_future_q - current_q)

    def get_reward(self, env):
        # Penalty for hitting the boundary or itself
        if self.head.x >= SCREEN_WIDTH // 10 or self.head.x < 0 or self.head.y >= SCREEN_HEIGHT // 10 or self.head.y < 0:
            return -100  # Hit boundary penalty
        for block in self.body[:-1]:
            if block == self.head:
                return -100  # Hit itself penalty

        # Reward for eating food
        if self.head == env.food.position:
            return 10  # Reward for eating food

        # Small penalty for each move to encourage faster play
        return -1

    def train(self, env):
        state = self.state_representation(env)
        action = self.decide(env)
        reward = self.get_reward(env)
        next_state = self.state_representation(env)
        self.update_q_table(state, action, reward, next_state)

        # Decay epsilon after each training step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        self.snake = Agent(5000)
        self.food = Food()
        self.font_style = pygame.font.Font(None, 25)

        self.game_over = False
        self.game_close = False

        self.episodes = 1000  # Number of training episodes
        self.game_loop()

    def danger(self, snake):
        return (
            snake.head.x >= SCREEN_WIDTH // 10 or snake.head.x < 0 or snake.head.y >= SCREEN_HEIGHT // 10 or snake.head.y < 0
            or snake.head in snake.body[:-1]
        )

    def score(self):
        value = self.font_style.render("Your Score: " + str(self.snake.length - 1), True, (255, 255, 255))
        self.screen.blit(value, [0, 0])

    def message(self, msg, color):
        mesg = self.font_style.render(msg, True, color)
        self.screen.blit(mesg, [SCREEN_WIDTH / 6, SCREEN_HEIGHT / 3])

    def game_loop(self):
        for episode in range(self.episodes):
            self.snake = Agent(5000)  # Reset snake at the start of each episode
            self.food.spawn()
            self.game_over = False
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_close = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.game_over = True

                self.snake.decide(self)  # Let the agent decide
                self.snake.move()

                # Check if snake hits itself or the boundaries
                if self.danger(self.snake):
                    self.game_over = True

                # Reward for eating food
                if self.snake.head == self.food.position:
                    self.food.spawn()
                    self.snake.length += 1

                # Train the agent
                self.snake.train(self)

                # Rendering
                self.screen.fill((25, 25, 30))
                self.food.render(self.screen)
                self.snake.render(self.screen)
                self.score()
                pygame.display.update()
                self.clock.tick(self.snake.speed)

            print(f"Episode {episode+1}/{self.episodes} completed.")
        pygame.quit()
        quit()
