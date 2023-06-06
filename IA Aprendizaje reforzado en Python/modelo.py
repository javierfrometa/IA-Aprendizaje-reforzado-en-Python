# Código Obtenido originalmente del curso online de freeCodeCamp.org Oython + PyTorch + Pygame Reinforcement Learning Snake AI
# para el módulo de proyecto final del grado de DAM https://github.com/patrickloeber/snake-ai-pytorch
# El código ha sido modificado para adaptarlo a las necesidades del proyecto
import torch
import torch.nn
import torch.optim
import torch.nn.functional
import os


# Define una red neuronal para Q-learning
class QNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define la primera capa lineal con input_size unidades de entrada y hidden_size unidades de salida
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        # Definir la segunda capa lineal con hidden_size unidades de entrada y output_size unidades de salida
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    # Definir el paso hacia adelante de la red neuronal
    def forward(self, x):
        # Aplicar activación ReLU a la salida de la primera capa lineal
        x = torch.nn.functional.relu(self.linear1(x))
        # Pasar la salida por la segunda capa lineal
        x = self.linear2(x)
        return x

    # Método para guardar el modelo en un fichero
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # Guardar el diccionario de estados del modelo en el archivo especificado
        torch.save(self.state_dict(), file_name)


#  Clase para entrenar la red neuronal Q
class QTrainer:
    def __init__(self, model, lr, gamma):
        # Constructor for the QTrainer class
        # Takes the Q-network model, learning rate (lr), and discount factor (gamma) as input
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Initialize an Adam optimizer to update the model's weights
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # Define the mean squared error (MSE) loss as the criterion for training
        self.criterion = torch.nn.MSELoss()

    # Método para realizar un único paso de entrenamiento para la red Q
    def train_step(self, state, action, reward, next_state, done):
        # Takes the current state, action, reward, next state, and done flag as input
        # Convert the input data to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # If the input data has shape (x,), reshape it to (1, x) for batch processing
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Calculate predicted Q values with current state
        pred = self.model(state)

        # Create a target tensor to store the updated Q values
        target = pred.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                # Update Q values using Bellman equation: Q_new = r + y * max(next_predicted Q value)
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the Q value for the chosen action in the target tensor
            target[idx][torch.argmax(action[idx]).item()] = q_new

        # Update the model weights using gradient descent
        self.optimizer.zero_grad()
        # Calculate the loss between predicted Q values and target Q values
        loss = self.criterion(target, pred)
        # Perform backpropagation and update the model weights
        loss.backward()

        self.optimizer.step()
