import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import os
from PIL import Image

class KraussSimulation:

    def __init__(self, num, acc_coeff=1, br_coeff=1, eps=1, v_max=40, road_length=1000, num_cars=30, time=20, reaction_time=1):
        self.num = num
        self.acc_coeff = acc_coeff
        self.br_coeff = br_coeff
        self.eps = eps
        self.v_max = v_max
        self.road_length = road_length
        self.num_cars = num_cars
        self.time = time
        self.reaction_time = reaction_time
        self.x, self.v = self._generate_cars()
        self.history_x = np.empty((self.time, self.num_cars))
        self.history_v = np.empty((self.time, self.num_cars))
    
    def _generate_cars(self):
        x = np.sort(np.random.uniform(0, self.road_length, self.num_cars))
        v = np.random.uniform(0, self.v_max, self.num_cars)

        return x, v

    def simulation(self):

        path = f'{os.getcwd()}/Krauss/simulations/{str(self.num).rjust(2, "0")}'
        #os.mkdir(path)
        history_x, history_v = self._krauss_simulation()
        avg_speed, max_speed, min_speed, avg_position, std_position, avg_gaps, max_gaps, min_gaps = self._get_stats()
        self._plot_space()
        self._plot_to_gif(path)
        self._plot_speed_stats(path, avg_speed, max_speed, min_speed)
        self._plot_position_stats(path, avg_position, std_position)
        self._plot_gaps_stats(path, avg_gaps, max_gaps, min_gaps)
        self._plot_history(path)

    def _krauss_simulation(self):

        for i in range(self.time):
            self.x, self.v = self._krauss_step(i)
        return self.history_x, self.history_v
    
    def _krauss_step(self, step):
        n = len(self.v)
        v_new = np.empty(n)
        x_new = np.empty(n)
        gaps = np.array([self.x[(i+1) % n] - self.x[i] for i in range(n)])
        gaps[-1] += self.road_length
        for i in range(n):
            full_stop_time = (self.v[i] + self.v[(i+1) % n])/2*self.br_coeff
            gap = gaps[i]
            gap_des = self.reaction_time * self.v[(i+1) % n]
            v_safe = self.v[(i+1) % n] + (gap - gap_des)/(self.reaction_time + full_stop_time)
            v_des = min([self.v_max, self.v[i] + self.acc_coeff, v_safe])
            v_new[i] = max(0, v_des - np.random.uniform(0, self.eps))

        self.history_v[step] = self.v
        self.history_x[step] = self.x % self.road_length

        for i in range(n):
            x_new[i] = self.x[i] + v_new[i]
        return np.sort(x_new), v_new
    
    def _get_stats(self):
        n = np.size(self.history_x, 1)
        avg_speed = [np.mean(self.history_v[i]) for i in range(self.time)]
        max_speed = [np.max(self.history_v[i]) for i in range(self.time)]
        min_speed = [np.min(self.history_v[i]) for i in range(self.time)]
        avg_position = [np.mean(self.history_x[i]) for i in range(self.time)]
        std_position = [np.std(self.history_x[i]) for i in range(self.time)]

        gaps = [[(self.history_x[j][(i+1) % n] - self.history_x[j][i]) % self.road_length for i in range(n)] for j in range(self.time)]
        gaps_avg = [np.mean(gaps[i]) for i in range(self.time)]
        gaps_max = [np.max(gaps[i]) for i in range(self.time)]
        gaps_min = [np.min(gaps[i]) for i in range(self.time)]
    
        return avg_speed, max_speed, min_speed, avg_position, std_position, gaps_avg, gaps_max, gaps_min
    
    def _plot_space(self):
        n = len(self.history_x[0])
        for i in range(self.time):
            plt.scatter(self.history_x[i], [0]*n, c=self.history_v[i], cmap='viridis', vmax=40, vmin=0)
            plt.colorbar()
            plt.xlim((0, self.road_length))
            plt.title(f't = {i} $\Delta t = 1$')
            plt.savefig(f'Krauss/graphs/{str(i+1).rjust(4, "0")}.png')
            plt.clf()
    
    def _plot_to_gif(self, path):
        source = f'{os.getcwd()}/Krauss/graphs'
        images = [Image.open(source + '/' + file) for file in os.listdir(source)]
        images[0].save(f'{path}/simulation_gif.gif', 
                    save_all=True,
                    append_images=images[1:],
                    duration=100,
                    loop=0
                    )
        for file in os.listdir(f'{os.getcwd()}/Krauss/graphs'):
            os.remove(os.path.join(f'{os.getcwd()}/Krauss/graphs', file))
    
    def _plot_speed_stats(self, path, avg_speed, max_speed, min_speed):
        t = np.arange(self.time)
        plt.plot(t, avg_speed, color='red', label='$v_{śr}$')
        plt.plot(t, max_speed, color='green', label='$v_{max}$')
        plt.plot(t, min_speed, color='blue', label='$v_{min}$')
        plt.legend()
        plt.xlabel('Czas')
        plt.ylabel('Prędkość')
        plt.savefig(f'{path}/speed_statistics.png')
        plt.clf()

    def _plot_position_stats(self, path, avg_position, std_position):
        t = np.arange(self.time)
        plt.plot(t, avg_position, color='red', label='$x_{śr}$')
        plt.plot(t, std_position, color='blue', label='$\sigma_x$')
        plt.legend()
        plt.xlabel('Czas')
        plt.ylabel('Wartość x')
        plt.savefig(f'{path}/position_statistics.png')
        plt.clf()

    def _plot_gaps_stats(self, path, avg_gaps, max_gaps, min_gaps):
        t = np.arange(self.time)
        plt.plot(t, avg_gaps, color='red', label='$d_{śr}$')
        plt.plot(t, max_gaps, color='green', label='$d_{max}$')
        plt.plot(t, min_gaps, color='blue', label='$d_{min}$')
        plt.legend()
        plt.xlabel('Czas')
        plt.ylabel('Odległości między samochodami')
        plt.savefig(f'{path}/gaps_statistics.png')
        plt.clf()

    def _plot_history(self, path):
        n = len(self.history_x[0])
        for i in range(self.time):
            plt.scatter(self.history_x[i], [i]*n, c=self.history_v[i], cmap='viridis', vmax=40, vmin=0)
        plt.colorbar()
        plt.xlim((0, self.road_length))
        plt.gca().invert_yaxis()
        plt.savefig(f'{path}/simulation.png')
        plt.clf()

    ###########################################################

if __name__ == '__main__':
    #full_krauss_simulation(0, time=200)
    #full_krauss_simulation(1, time=1000, num_cars=100, road_length=5000)
    #full_krauss_simulation(2, time=2000, num_cars=300, road_length=5000)
    KraussSimulation(6).simulation()

