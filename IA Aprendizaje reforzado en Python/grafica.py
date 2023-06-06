# Código Obtenido originalmente del curso online de freeCodeCamp.org Oython + PyTorch + Pygame Reinforcement Learning Snake AI
# para el módulo de proyecto final del grado de DAM https://github.com/patrickloeber/snake-ai-pytorch
# El código ha sido modificado para adaptarlo a las necesidades del proyecto
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Entrenamiento...')
    plt.xlabel('Número de partidas')
    plt.ylabel('Puntuación')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
