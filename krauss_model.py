import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import os
from PIL import Image


def full_krauss_simulation(num, acc_coeff=1, br_coeff=1, eps=1, v_max=40, road_length=1000, num_cars=30, time=20, reaction_time=1):

    path = f'{os.getcwd()}/simulations/{str(num).rjust(2, "0")}'
    os.mkdir(path)

    history_x, history_v = krauss_simulation(acc_coeff, br_coeff, eps, v_max, road_length, num_cars, time, reaction_time)
    avg_speed, max_speed, min_speed, avg_position, std_position, avg_gaps, max_gaps, min_gaps = get_stats(history_x, history_v, time, road_length)
    plot_space(road_length, time, history_x, history_v)

    plot_to_gif(path)
    plot_speed_stats(path, avg_speed, max_speed, min_speed, time)
    plot_position_stats(path, avg_position, std_position, time)
    plot_gaps_stats(path, avg_gaps, max_gaps, min_gaps, time)
    plot_history(path, road_length, time, history_x, history_v)

def krauss_simulation(acc_coeff=1, br_coeff=1, eps=1, v_max=40, road_length=1000, num_cars=30, time=20, reaction_time=1):
    x, v = generate_cars(road_length, num_cars, v_max)
    history_x = np.empty((time, num_cars))
    history_v = np.empty((time, num_cars))
    for i in range(time):
        x, v = krauss_step(x, v, acc_coeff, br_coeff, eps, v_max, reaction_time, road_length, history_v=history_v, history_x=history_x, step=i)
    return history_x, history_v

def generate_cars(road_length, num_cars, v_max):
    x = np.sort(np.random.uniform(0, road_length, num_cars))
    v = np.random.uniform(0, v_max, num_cars)

    return x, v

def krauss_step(x, v, acc_coeff, br_coeff, eps, v_max, reaction_time, road_length, history_v, history_x, step):
    n = len(v)
    v_new = np.empty(n)
    x_new = np.empty(n)
    gaps = np.array([x[(i+1) % n] - x[i] for i in range(n)])
    gaps[-1] += road_length
    for i in range(n):
        full_stop_time = (v[i] + v[(i+1) % n])/2*br_coeff
        gap = gaps[i]
        gap_des = reaction_time * v[(i+1) % n]
        v_safe = v[(i+1) % n] + (gap - gap_des)/(reaction_time + full_stop_time)
        v_des = min([v_max, v[i] + acc_coeff, v_safe])
        v_new[i] = max(0, v_des - np.random.uniform(0, eps))

    history_v[step] = v
    history_x[step] = x % road_length

    for i in range(n):
        x_new[i] = x[i] + v_new[i]
    return np.sort(x_new), v_new


def plot_space(road_length, time, history_x, history_v, ):
    n = len(history_x[0])
    for i in range(time):
        plt.scatter(history_x[i], [0]*n, c=history_v[i], cmap='viridis', vmax=40, vmin=0)
        plt.colorbar()
        plt.xlim((0, road_length))
        plt.title(f't = {i} $\Delta t = 1$')
        plt.savefig(f'graphs/{str(i+1).rjust(4, "0")}.png')
        plt.clf()

def plot_to_gif(path):
    source = f'{os.getcwd()}/graphs'
    images = [Image.open(source + '/' + file) for file in os.listdir(source)]
    images[0].save(f'{path}/simulation_gif.gif', 
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0
                   )
    for file in os.listdir(f'{os.getcwd()}/graphs'):
        os.remove(os.path.join(f'{os.getcwd()}/graphs', file))

def get_stats(history_x, history_v, time, road_length):
    n = np.size(history_x, 1)
    avg_speed = [np.mean(history_v[i]) for i in range(time)]
    max_speed = [np.max(history_v[i]) for i in range(time)]
    min_speed = [np.min(history_v[i]) for i in range(time)]
    avg_position = [np.mean(history_x[i]) for i in range(time)]
    std_position = [np.std(history_x[i]) for i in range(time)]

    gaps = [[(history_x[j][(i+1) % n] - history_x[j][i]) % road_length for i in range(n)] for j in range(time)]
    gaps_avg = [np.mean(gaps[i]) for i in range(time)]
    gaps_max = [np.max(gaps[i]) for i in range(time)]
    gaps_min = [np.min(gaps[i]) for i in range(time)]
    
    return avg_speed, max_speed, min_speed, avg_position, std_position, gaps_avg, gaps_max, gaps_min

def plot_speed_stats(path, avg_speed, max_speed, min_speed, time):
    t = np.arange(time)
    plt.plot(t, avg_speed, color='red', label='$v_{śr}$')
    plt.plot(t, max_speed, color='green', label='$v_{max}$')
    plt.plot(t, min_speed, color='blue', label='$v_{min}$')
    plt.legend()
    plt.xlabel('Czas')
    plt.ylabel('Prędkość')
    plt.savefig(f'{path}/speed_statistics.png')
    plt.clf()

def plot_position_stats(path, avg_position, std_position, time):
    t = np.arange(time)
    plt.plot(t, avg_position, color='red', label='$x_{śr}$')
    plt.plot(t, std_position, color='blue', label='$\sigma_x$')
    plt.legend()
    plt.xlabel('Czas')
    plt.ylabel('Wartość x')
    plt.savefig(f'{path}/position_statistics.png')
    plt.clf()

def plot_gaps_stats(path, avg_gaps, max_gaps, min_gaps, time):
    t = np.arange(time)
    plt.plot(t, avg_gaps, color='red', label='$d_{śr}$')
    plt.plot(t, max_gaps, color='green', label='$d_{max}$')
    plt.plot(t, min_gaps, color='blue', label='$d_{min}$')
    plt.legend()
    plt.xlabel('Czas')
    plt.ylabel('Odległości między samochodami')
    plt.savefig(f'{path}/gaps_statistics.png')
    plt.clf()

def plot_history(path, road_length, time, history_x, history_v):
    n = len(history_x[0])
    for i in range(time):
        plt.scatter(history_x[i], [i]*n, c=history_v[i], cmap='viridis', vmax=40, vmin=0)
    plt.colorbar()
    plt.xlim((0, road_length))
    plt.gca().invert_yaxis()
    plt.savefig(f'{path}/simulation.png')
    plt.clf()

if __name__ == '__main__':
    #full_krauss_simulation(0, time=200)
    #full_krauss_simulation(1, time=1000, num_cars=100, road_length=5000)
    #full_krauss_simulation(2, time=2000, num_cars=300, road_length=5000)
    full_krauss_simulation(5)

