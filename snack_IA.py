import pygame
import random
import numpy as np
import sys
import pickle
import argparse
import os
import csv
import threading

BLOCK_SIZE = 20
SPEED = 40

q_lock = threading.Lock()
csv_lock = threading.Lock()

# Classe du jeu Snake
class SnakeGame:
    # Initialise la fenêtre de jeu, les paramètres et lance la réinitialisation du jeu
    def __init__(self, width=640, height=480, block_size=BLOCK_SIZE, speed=SPEED, display=True):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        self.display = display
        if self.display:
            pygame.init()
            self.display_surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()
        self.reset()

    # Réinitialise les paramètres du jeu (position de la tête, du serpent, score et nourriture)
    def reset(self):
        self.direction = "RIGHT"
        self.head = [self.width // 2, self.height // 2]
        self.snake = [self.head.copy(),
                      [self.head[0] - self.block_size, self.head[1]],
                      [self.head[0] - 2 * self.block_size, self.head[1]]]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    # Place la nourriture à une position aléatoire sur la grille en évitant le serpent
    def _place_food(self):
        x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    # Exécute un pas de jeu en traitant l'action (liste de 3 éléments : [tout droit, droite, gauche])
    def play_step(self, action):
        self.frame_iteration += 1
        if self.display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        self._move(action)
        self.snake.insert(0, self.head.copy())
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 95 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        if self.display:
            self._update_ui()
            self.clock.tick(self.speed)
        return reward, game_over, self.score

    # Met à jour l'affichage graphique du jeu
    def _update_ui(self):
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (200, 0, 0)
        green = (0, 255, 0)
        self.display_surface.fill(black)
        for pt in self.snake:
            pygame.draw.rect(self.display_surface, green, pygame.Rect(pt[0], pt[1], self.block_size, self.block_size))
        pygame.draw.rect(self.display_surface, red, pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        font = pygame.font.Font(None, 36)
        text = font.render("Score: " + str(self.score), True, white)
        self.display_surface.blit(text, [0, 0])
        pygame.display.flip()

    # Met à jour la direction et la position de la tête en fonction de l'action choisie
    def _move(self, action):
        directions = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = directions.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = directions[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = directions[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):
            new_dir = directions[(idx - 1) % 4]
        else:
            new_dir = directions[idx]
        self.direction = new_dir
        x, y = self.head
        if self.direction == "RIGHT":
            x += self.block_size
        elif self.direction == "LEFT":
            x -= self.block_size
        elif self.direction == "UP":
            y -= self.block_size
        elif self.direction == "DOWN":
            y += self.block_size
        self.head = [x, y]

    # Vérifie si la tête ou un point donné est en collision avec le mur ou le corps du serpent
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt[0] < 0 or pt[0] >= self.width or pt[1] < 0 or pt[1] >= self.height:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    # Retourne l'état actuel du jeu sous forme d'un vecteur d'entiers
    def get_state(self):
        point_l = self._get_next_point('LEFT')
        point_r = self._get_next_point('RIGHT')
        point_s = self._get_next_point('STRAIGHT')
        danger_straight = 1 if self._is_collision(point_s) else 0
        danger_right = 1 if self._is_collision(point_r) else 0
        danger_left = 1 if self._is_collision(point_l) else 0
        dir_up = 1 if self.direction == "UP" else 0
        dir_down = 1 if self.direction == "DOWN" else 0
        dir_left = 1 if self.direction == "LEFT" else 0
        dir_right = 1 if self.direction == "RIGHT" else 0
        food_left = 1 if self.food[0] < self.head[0] else 0
        food_right = 1 if self.food[0] > self.head[0] else 0
        food_up = 1 if self.food[1] < self.head[1] else 0
        food_down = 1 if self.food[1] > self.head[1] else 0
        state = [
            danger_straight, danger_right, danger_left,
            dir_up, dir_down, dir_left, dir_right,
            food_left, food_right, food_up, food_down
        ]
        return np.array(state, dtype=int)

    # Calcule le point suivant par rapport à la tête en fonction d'une direction relative
    def _get_next_point(self, relative_direction):
        directions = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = directions.index(self.direction)
        if relative_direction == 'STRAIGHT':
            new_dir = directions[idx]
        elif relative_direction == 'RIGHT':
            new_dir = directions[(idx + 1) % 4]
        elif relative_direction == 'LEFT':
            new_dir = directions[(idx - 1) % 4]
        else:
            new_dir = directions[idx]
        x, y = self.head
        if new_dir == "RIGHT":
            x += self.block_size
        elif new_dir == "LEFT":
            x -= self.block_size
        elif new_dir == "UP":
            y -= self.block_size
        elif new_dir == "DOWN":
            y += self.block_size
        return [x, y]

# Agent d'IA utilisant le Q-learning
class Agent:
    # Initialise l'agent, ses hyperparamètres et le dictionnaire des Q-valeurs
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9
        self.learning_rate = 0.1
        self.Q = {}
        self.actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Convertit l'état (numpy array) en clé (tuple) pour le dictionnaire
    def get_state_key(self, state):
        return tuple(state.tolist())

    # Retourne la Q-valeur pour un état et une action donnée
    def get_Q(self, state, action):
        key = (self.get_state_key(state), tuple(action))
        return self.Q.get(key, 0)

    # Met à jour la Q-valeur pour un état et une action en utilisant la formule du Q-learning
    def update_Q(self, state, action, reward, next_state, game_over):
        key = (self.get_state_key(state), tuple(action))
        with q_lock:
            old_value = self.Q.get(key, 0)
            next_max = max([self.get_Q(next_state, a) for a in self.actions])
            target = reward if game_over else reward + self.gamma * next_max
            new_value = old_value + self.learning_rate * (target - old_value)
            self.Q[key] = new_value

    # Sélectionne l'action à réaliser en fonction de l'état courant et de la stratégie epsilon-greedy
    def get_action(self, state):
        epsilon = max(80 - self.n_games, 0)
        if random.randint(0, 200) < epsilon:
            move = random.choice(self.actions)
        else:
            qs = [self.get_Q(state, a) for a in self.actions]
            max_index = qs.index(max(qs))
            move = self.actions[max_index]
        return move

    # Méthode de mémorisation d'expérience (non implémentée)
    def remember(self, state, action, reward, next_state, game_over):
        pass

    # Entraîne la mémoire courte sur une expérience unique
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.update_Q(state, action, reward, next_state, game_over)

    # Entraîne sur la mémoire longue (non implémentée)
    def train_long_memory(self):
        pass

# Entraînement multi-threadé de l'IA sur un nombre total d'épisodes
def train(total_episodes):
    agent = Agent()
    if os.path.isfile("q_values.pkl"):
        with open("q_values.pkl", "rb") as f:
            agent.Q = pickle.load(f)
        print("Q-values chargées depuis la session précédente.")
    else:
        print("Aucun fichier de Q-values trouvé. Entraînement à partir de zéro.")

    num_threads = 4
    episodes_per_thread = total_episodes // num_threads
    remainder = total_episodes % num_threads
    csv_file = "training_data.csv"
    episode_offset = 0
    if os.path.isfile(csv_file):
        with open(csv_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            if len(rows) > 1:
                try:
                    episode_offset = int(rows[-1][0])
                except ValueError:
                    episode_offset = 0

    # Fonction worker exécutée par chaque thread pour lancer les épisodes
    def worker(episodes_to_run):
        nonlocal episode_offset
        for _ in range(episodes_to_run):
            game = SnakeGame(display=False)
            game.reset()
            state = game.get_state()
            done = False
            while not done:
                action = agent.get_action(state)
                reward, done, score = game.play_step(action)
                next_state = game.get_state()
                agent.train_short_memory(state, action, reward, next_state, done)
                state = next_state
            with csv_lock:
                episode_offset += 1
                with open(csv_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([episode_offset, score])
            with csv_lock:
                agent.n_games += 1
            print(f"Episode: {episode_offset}, Score: {score}")

    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Episode", "Score"])

    threads = []
    for i in range(num_threads):
        ep_count = episodes_per_thread + (1 if i == 0 and remainder > 0 else 0)
        t = threading.Thread(target=worker, args=(ep_count,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    with open("q_values.pkl", "wb") as f:
        pickle.dump(agent.Q, f)
    print("Entraînement multi-threadé terminé. Q-values sauvegardées dans 'q_values.pkl' et données de parties dans 'training_data.csv'.")

# Lance une démo graphique de l'IA en chargeant les Q-valeurs sauvegardées
def demo():
    agent = Agent()
    try:
        with open("q_values.pkl", "rb") as f:
            agent.Q = pickle.load(f)
    except FileNotFoundError:
        print("Fichier 'q_values.pkl' non trouvé. Veuillez d'abord entraîner l'IA avec le mode 'train'.")
        return
    game = SnakeGame(display=True)
    while True:
        game.reset()
        state = game.get_state()
        done = False
        while not done:
            action = agent.get_action(state)
            reward, done, score = game.play_step(action)
            state = game.get_state()
            if done:
                print("Score final:", score)
                pygame.time.delay(1000)
                break

# Interface en ligne de commande pour choisir le mode d'exécution : entraînement ou démo
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement et démo de l'IA pour Snake")
    parser.add_argument("mode", choices=["train", "demo"],
                        help="Choisissez 'train' pour entraîner l'IA ou 'demo' pour lancer la démo graphique.")
    parser.add_argument("episodes", nargs="?", type=int, default=1000,
                        help="Nombre d'épisodes d'entraînement (utilisé uniquement en mode train).")
    args = parser.parse_args()
    if args.mode == "train":
        train(args.episodes)
    elif args.mode == "demo":
        demo()
