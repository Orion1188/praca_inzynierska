import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import os
from PIL import Image


class QLearningSimulation:
    def __init__(
        self,
        num,
        acc_coeff=0.2,
        br_coeff=0.6,
        eps=1,
        v_max=5,
        road_length=200,
        num_cars=100,
        time=100,
        reaction_time=1,
        v_disc=41,
        v_prev_disc=21,
        gap_disc=21,
        alpha=0.1,
        gamma=0.99,
        create_gif=False
    ):
        self.num = num
        self.acc_coeff = acc_coeff
        self.br_coeff = br_coeff
        self.eps = eps
        self.v_max = v_max
        self.road_length = road_length
        self.num_cars = num_cars
        self.time = time
        self.reaction_time = reaction_time
        self.alpha = alpha
        self.gamma = gamma
        self.x, self.v = self._generate_cars()
        self.history_x = np.empty((self.time, self.num_cars))
        self.history_v = np.empty((self.time, self.num_cars))
        self.v_disc = np.arange(0, v_max, v_disc)
        self.v_prev_disc = np.arange(0, v_max, v_prev_disc)
        self.gap_disc = np.arange(0, v_max, gap_disc)
        self.q_table_steady = np.zeros((v_disc, v_prev_disc, gap_disc))
        self.q_table_accelerate = np.zeros((v_disc, v_prev_disc, gap_disc))
        self.create_gif = create_gif

    def _generate_cars(self):
        x = np.sort(np.random.uniform(0, self.road_length, self.num_cars))
        v = np.random.uniform(0, self.v_max, self.num_cars)

        return x, v

    def simulation(self):
        path = f'{os.getcwd()}/RL/simulations/{str(self.num).rjust(2, "0")}'
        os.mkdir(path)
        self._qlearning_simulation()
        (
            avg_speed,
            max_speed,
            min_speed,
            avg_position,
            std_position,
            avg_gaps,
            max_gaps,
            min_gaps,
        ) = self._get_stats()
        if self.create_gif:
            self._plot_space()
            self._plot_to_gif(path)
        self._plot_speed_stats(path, avg_speed, max_speed, min_speed)
        self._plot_position_stats(path, avg_position, std_position)
        self._plot_gaps_stats(path, avg_gaps, max_gaps, min_gaps)
        self._plot_history(path)
        print(np.sum(self.q_table_accelerate > self.q_table_steady))

    def _qlearning_simulation(self):
        for i in range(self.time):
            self.x, self.v = self._qlearning_step(i)
        # return self.history_x, self.history_v

    def _qlearning_step(self, step):
        n = len(self.v)
        v_new = np.empty(n)
        x_new = np.empty(n)
        gaps = np.array([self.x[(i + 1) % n] - self.x[i] for i in range(n)])
        gaps[-1] += self.road_length
        for i in range(n):

            # wyznaczenie dyskretnych współrzędnych obecnego stanu w Q-table

            state = [
                np.searchsorted(self.v_disc, self.v[i]),
                np.searchsorted(self.v_prev_disc, self.v[(i - 1) % n]),
                np.searchsorted(self.gap_disc, gaps[i]),
            ]

            # wyznaczenie prędkości przy założeniu, że przyspieszymy/będziemy zmuszeni żeby zwolnić

            full_stop_time = (self.v[i] + self.v[(i + 1) % n]) / 2 * self.br_coeff
            gap_des = self.reaction_time * self.v[(i + 1) % n]
            v_safe = self.v[(i + 1) % n] + (gaps[i] - gap_des) / (
                self.reaction_time + full_stop_time
            )
            v_des = min([self.v_max, self.v[i] + self.acc_coeff, v_safe])

            # podjęcie decyzji

            if v_des > self.v[i]:

                if (
                    self.q_table_accelerate[state[0], state[1], state[2]]
                    > self.q_table_steady[state[0], state[1], state[2]]
                ):
                    action = "accelerate"
                    state_new = [
                        np.searchsorted(self.v_disc, v_des),
                        state[1],
                        state[2],
                    ]
                    v_new[i] = v_des
                elif (
                    self.q_table_accelerate[state[0], state[1], state[2]]
                    == self.q_table_steady[state[0], state[1], state[2]]
                ):
                    if np.random.random() > 0.5:
                        action = "accelerate"
                        state_new = [
                            np.searchsorted(self.v_disc, v_des),
                            state[1],
                            state[2],
                        ]
                        v_new[i] = v_des
                    else:
                        action = "steady"
                        state_new = [
                            np.searchsorted(self.v_disc, self.v[i]),
                            state[1],
                            state[2],
                        ]
                        v_new[i] = self.v[i]
                else:
                    action = "steady"
                    state_new = [
                        np.searchsorted(self.v_disc, self.v[i]),
                        state[1],
                        state[2],
                    ]
                    v_new[i] = self.v[i]

            else:
                action = "slowdown"
                v_new[i] = v_des

        # zapisanie poprzedniego stanu

        self.history_v[step] = self.v
        self.history_x[step] = self.x % self.road_length

        # wprowadzenie nowego stanu

        for i in range(n):
            x_new[i] = self.x[i] + v_new[i]

        # aktualizacja Q-table

        for i in range(n):
            if action == "steady":
                reward = v_new[(i - 1) % n] - self.history_v[step][(i - 1) % n]
                self.q_table_steady[state] = (1 - self.alpha) * self.q_table_steady[state] + self.alpha * (reward + self.gamma * self.q_table_steady[state_new])
            elif action == "accelerate":
                reward = v_new[(i - 1) % n] - self.history_v[step][(i - 1) % n]
                self.q_table_accelerate[state] = (1 - self.alpha) * self.q_table_accelerate[state] + self.alpha * (reward + self.gamma * self.q_table_accelerate[state_new])

        return np.sort(x_new), v_new

    def _get_stats(self):
        n = np.size(self.history_x, 1)
        avg_speed = [np.mean(self.history_v[i]) for i in range(self.time)]
        max_speed = [np.max(self.history_v[i]) for i in range(self.time)]
        min_speed = [np.min(self.history_v[i]) for i in range(self.time)]
        avg_position = [np.mean(self.history_x[i]) for i in range(self.time)]
        std_position = [np.std(self.history_x[i]) for i in range(self.time)]

        gaps = [
            [
                (self.history_x[j][(i + 1) % n] - self.history_x[j][i])
                % self.road_length
                for i in range(n)
            ]
            for j in range(self.time)
        ]
        gaps_avg = [np.mean(gaps[i]) for i in range(self.time)]
        gaps_max = [np.max(gaps[i]) for i in range(self.time)]
        gaps_min = [np.min(gaps[i]) for i in range(self.time)]

        return (
            avg_speed,
            max_speed,
            min_speed,
            avg_position,
            std_position,
            gaps_avg,
            gaps_max,
            gaps_min,
        )

    def _plot_space(self):
        n = len(self.history_x[0])
        for i in range(self.time):
            plt.scatter(
                self.history_x[i],
                [0] * n,
                c=self.history_v[i],
                cmap="viridis",
                vmax=self.v_max,
                vmin=0,
            )
            plt.colorbar()
            plt.xlim((0, self.road_length))
            plt.title(f"t = {i} $\Delta t = 1$")
            plt.savefig(f'Krauss/graphs/{str(i+1).rjust(4, "0")}.png')
            plt.clf()

    def _plot_to_gif(self, path):
        source = f"{os.getcwd()}/Krauss/graphs"
        images = [Image.open(source + "/" + file) for file in os.listdir(source)]
        images[0].save(
            f"{path}/simulation_gif.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
        for file in os.listdir(f"{os.getcwd()}/Krauss/graphs"):
            os.remove(os.path.join(f"{os.getcwd()}/Krauss/graphs", file))

    def _plot_speed_stats(self, path, avg_speed, max_speed, min_speed):
        t = np.arange(self.time)
        plt.plot(t, avg_speed, color="red", label="$v_{śr}$")
        plt.plot(t, max_speed, color="green", label="$v_{max}$")
        plt.plot(t, min_speed, color="blue", label="$v_{min}$")
        plt.legend()
        plt.xlabel("Czas")
        plt.ylabel("Prędkość")
        plt.savefig(f"{path}/speed_statistics.png")
        plt.clf()

    def _plot_position_stats(self, path, avg_position, std_position):
        t = np.arange(self.time)
        plt.plot(t, avg_position, color="red", label="$x_{śr}$")
        plt.plot(t, std_position, color="blue", label="$\sigma_x$")
        plt.legend()
        plt.xlabel("Czas")
        plt.ylabel("Wartość x")
        plt.savefig(f"{path}/position_statistics.png")
        plt.clf()

    def _plot_gaps_stats(self, path, avg_gaps, max_gaps, min_gaps):
        t = np.arange(self.time)
        plt.plot(t, avg_gaps, color="red", label="$d_{śr}$")
        plt.plot(t, max_gaps, color="green", label="$d_{max}$")
        plt.plot(t, min_gaps, color="blue", label="$d_{min}$")
        plt.legend()
        plt.xlabel("Czas")
        plt.ylabel("Odległości między samochodami")
        plt.savefig(f"{path}/gaps_statistics.png")
        plt.clf()

    def _plot_history(self, path):
        n = len(self.history_x[0])
        for i in range(self.time):
            plt.scatter(
                self.history_x[i],
                [i] * n,
                c=self.history_v[i],
                cmap="viridis",
                vmax=self.v_max,
                vmin=0,
            )
        plt.colorbar()
        plt.xlim((0, self.road_length))
        plt.gca().invert_yaxis()
        plt.savefig(f"{path}/simulation.png")
        plt.clf()

    ###########################################################

if __name__ == "__main__":
    
    for i in range(5):
        QLearningSimulation(i+1, time=5000).simulation()
