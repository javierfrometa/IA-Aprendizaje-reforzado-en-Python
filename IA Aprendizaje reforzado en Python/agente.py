# Código Obtenido originalmente del curso online de freeCodeCamp.org Oython + PyTorch + Pygame Reinforcement Learning Snake AI
# para el módulo de proyecto final del grado de DAM https://github.com/patrickloeber/snake-ai-pytorch
# El código ha sido modificado para adaptarlo a las necesidades del proyecto
import torch
import random
import numpy
from collections import deque
from juego import SnakeGameAI, Direction, Point
from modelo import QNet, QTrainer
from grafica import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agente:

    def __init__(self):
        self.n_games = 0  # numero de partidas
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        # Si excedemos la memoria máxima, eliminaremos la memoria más antigua. popleft() se utiliza para eliminar la memoria más antigua.
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNet(11, 256, 3)  # 11 entradas, 256 capas ocultas, 3 salidas
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # Usaremos una red neuronal para entrenar a nuestro agente. La red neuronal tendrá 11 entradas, 256 capas ocultas y 3 salidas.
    # Las 11 entradas son los 11 estados que hemos definido en la función get_state().
    # Las 3 salidas son las 3 posibles acciones que puede realizar el agente. Las 256 capas ocultas son el número de neuronas de la capa oculta.
    # La capa oculta es la capa que está entre la capa de entrada y la capa de salida.
    # La capa oculta es donde la red neuronal hace predicciones.
    # La capa oculta es donde la red neuronal aprende.
    def get_state(self, game):
        # La cabeza de la serpiente es el primer elemento de la lista de la serpiente.
        head = game.snake[0]

        # Creamos 4 puntos que representan la posición de la cabeza de la serpiente si se mueve hacia la izquierda, derecha, arriba o abajo.
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # state es una lista de 11 elementos que consiste en booleans que indican si la serpiente está en peligro o no.
        # Los 3 primeros elementos de la lista de estado indican si la serpiente está en peligro si se mueve recto, derecha o izquierda.
        # Los siguientes 4 elementos de la lista de estados indican si la serpiente se mueve a la izquierda, derecha, arriba o abajo.
        # Los 4 últimos elementos de la lista de estados indican si la comida está a la izquierda, derecha o arriba de la serpiente.
        state = [
            # Peligro recto
            # Check right
            (dir_r and game.is_collision(point_r)) or
            # Check left
            (dir_l and game.is_collision(point_l)) or
            # Check up
            (dir_u and game.is_collision(point_u)) or
            # Check down
            (dir_d and game.is_collision(point_d)),

            # Peligro derecha
            # Check right
            (dir_u and game.is_collision(point_r)) or
            # Check left
            (dir_d and game.is_collision(point_l)) or
            # Check up
            (dir_l and game.is_collision(point_u)) or
            # Check down
            (dir_r and game.is_collision(point_d)),

            # Peligro izquierda
            # Check right
            (dir_d and game.is_collision(point_r)) or
            # Check left
            (dir_u and game.is_collision(point_l)) or
            # Check up
            (dir_r and game.is_collision(point_u)) or
            # Check down
            (dir_l and game.is_collision(point_d)),

            # Dirección
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Comida ubicación
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return numpy.array(state, dtype=int)

    # Guardar en el buffer de memory un tuple con experiencias pasadas
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    # Entrenar a la red neuronal con experiencias pasadas
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # sample aleatorio de mini-batch de la memoria
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # unpack mini-batch into separate lists
        self.trainer.train_step(states, actions, rewards, next_states,
                                dones)  # call trainer's train_step method with mini-batch data

    # Entrenar a la red neuronal con experiencias recientes
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Elegir la acción que el agente realizará, si explorar nuevas acciones o explotar las acciones que ha aprendido
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games  # calculate exploration rate epsilon based on number of games played
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:  # choose random action with probability epsilon
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # choose action with the highest predicted Q-value from the model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


# Entrenar a la red neuronal
def train():
    plot_scores = []  # list to store scores for plotting
    plot_mean_scores = []  # list to store mean scores for plotting
    total_score = 0  # variable to keep track of total score
    record = 0  # variable to keep track of the highest score
    agent = Agente()  # create an instance of Agent class
    game = SnakeGameAI()  # create an instance of SnakeGameAI class
    while True:
        # obtener el estado actual del juego como entrada para el agente
        state_old = agent.get_state(game)

        # obtener la acción que el agente realizará
        final_move = agent.get_action(state_old)

        # realizar accion y obtener reward, done y score
        reward, done, score = game.play_step(final_move)

        # obtener el nuevo estado del juego como entrada para el agente
        state_new = agent.get_state(game)

        # entrenar la memoria a corto plazo del agente con los datos actuales de acción-recompensa
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # almacenar los datos actuales de acción-recompensa en la memoria del agente para el entrenamiento a largo plazo
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()  # reset game for new episode
            agent.n_games += 1  # increment number of games played by agent
            agent.train_long_memory()  # train agent's long-term memory with stored data in memory

            if score > record:
                record = score  # update the highest score
                agent.model.save()

            print('Juego', agent.n_games, 'Puntuación', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
