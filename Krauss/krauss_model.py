import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image


class KraussSimulation:

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
        jam_speed_coeff=0.2,
        jam_gap_coeff=0.2,
        jam_cars_involved_coeff=0.1,
        only_stats=True,
        create_gif=False
    ):
        '''
            Parametry symulacji:
            - num: numer symulacji, potrzebny do utworzenia folderu w simulations.
            - acc_coeff: współczynnik odpowiadający za przyspieszenie.
            - br_coeff: współczynnik odpowiadający za hamowanie.
            - eps: parametr szumu (z przedziału [0, 1]).
            - v_max: prędkość maksymalna.
            - road_length: długość trasy.
            - num_cars: liczba samochodów.
            - time: liczba kroków symulacji.
            - reaction time: czas reakcji kierowcy.

            Pozostałe parametry:
            - jam_speed_coeff - parametr wyznaczający minimalną prędkość, przy której samochód nie jest uznawany za uczestnika korku. Prędkość jest wyznaczana jako iloczyn współczynnika i prędkości przy jednorodnym ruchu.
            - jam_gap_coeff - parametr wyznaczający minimalny odstęp, przy którym samochód nie jest uznawany za uczestnika korku. Odstęp jest wyznaczany jako iloczyn współczynnika i odstępu przy jednorodnym ruchu.
            - jam_cars_involved_coeff - parametr wskazujący jaka część wszystkich samochodów musi stać w korku, żeby uznać obecność zatoru.
            - only_stats - jeśli True wykonanie symulacji nie generuje wykresów do odpowiedniego folderu.
            - create_gif - (działa pod warunkiem, że only_stats==False) jeśli True zostanie wygenerowany gif całej symulacji.
        '''
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
        self.create_gif = create_gif
        self.jam_speed_coeff = jam_speed_coeff
        self.jam_gap_coeff = jam_gap_coeff
        self.jam_cars_involved_coeff = jam_cars_involved_coeff
        self.only_stats = only_stats

    def _generate_cars(self):
        '''
            Generuje stan początkowy (samochody stoją w równych odstępach)
        '''
        # x = np.sort(np.random.uniform(0, self.road_length, self.num_cars))
        # v = np.random.uniform(0, self.v_max, self.num_cars)
        x = np.linspace(0, self.road_length, self.num_cars, endpoint=False)
        v = np.zeros(self.num_cars)

        return x, v

    def simulation(self):
        '''
            Wykonuje symulację dla parametrów zadanych dla instancji klasy.
            Jeśli only_stats==False zostaną dodatkowo wygenerowane wykresy dla symulacji.
            Jeśli create_gif==True zostanie wygenerowany gif całej symulacji (działa pod warunkiem, że only_stats==False).
        '''
        self._krauss_simulation()
        if not self.only_stats:
            path = f'{os.getcwd()}/Krauss/simulations/{str(self.num).rjust(2, "0")}'
            os.mkdir(path)
            if self.create_gif:
                self._plot_space()
                self._plot_to_gif(path)
            self._plot_speed_stats(path)
            self._plot_traffic_flow_stats(path)
            self._plot_jam_stats(path)
            self._plot_history(path)

    def _krauss_simulation(self):
        '''
            Wykonuje zadaną liczbę kroków symulacji.
        '''
        for i in range(self.time):
            self.x, self.v = self._krauss_step(i)

    def _krauss_step(self, step):
        '''
            Pojedynczy krok symulacji
        '''
        n = len(self.v)
        v_new = np.empty(n)
        x_new = np.empty(n)
        
        # Wyznaczam odstęp między obecnym a następnym samochodem
        gaps = np.array([(self.x[(i + 1) % n] - self.x[i]) % self.road_length for i in range(n)])
        
        for i in range(n): # Wyznaczam v_des dla każdego samochodu zgodnie z wzorami z artykułu
            full_stop_time = (self.v[i] + self.v[(i + 1) % n]) / (2 * self.br_coeff)
            gap = gaps[i]
            gap_des = self.reaction_time * self.v[(i + 1) % n]
            v_safe = self.v[(i + 1) % n] + (gap - gap_des) / (
                self.reaction_time + full_stop_time
            )
            v_des = min([self.v_max, self.v[i] + self.acc_coeff, v_safe])
            if (v_des == self.v_max) or (v_des == v_safe): # odejmuję szum w przypadku gdy prędkość jest ograniczona przez v_max lub v_safe
                v_new[i] = max(0, v_des - np.random.uniform(0, self.eps))
            else:
                v_new[i] = max(0, v_des)

        self.history_v[step] = self.v
        self.history_x[step] = self.x % self.road_length

        for i in range(n):
            x_new[i] = (self.x[i] + v_new[i]) % self.road_length
        return x_new, v_new

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
                s=3,
            )
            plt.colorbar()
            plt.xlim((0, self.road_length))
            plt.title(f"t = {i}")
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

    def _plot_speed_stats(self, path):
        t = np.arange(self.time)
        avg_speed = [np.mean(self.history_v[i]) for i in range(self.time)]
        max_speed = [np.max(self.history_v[i]) for i in range(self.time)]
        min_speed = [np.min(self.history_v[i]) for i in range(self.time)]
        plt.figure(dpi=1200)
        plt.plot(t, avg_speed, color="red", label="$v_{śr}$")
        plt.plot(t, max_speed, color="green", label="$v_{max}$")
        plt.plot(t, min_speed, color="blue", label="$v_{min}$")
        plt.legend()
        plt.xlabel("Czas")
        plt.ylabel("Prędkość")
        plt.savefig(f"{path}/velocity.png")
        plt.clf()

    def _plot_traffic_flow_stats(self, path):
        t = np.arange(self.time)
        avg_speed = np.array([np.mean(self.history_v[i]) for i in range(self.time)])
        traffic_flow = avg_speed * self.num_cars / self.road_length
        plt.figure(dpi=1200)
        plt.plot(t, traffic_flow, color="red", label="Przepływ")
        plt.legend()
        plt.xlabel("Czas")
        plt.ylabel("Przepływ")
        plt.savefig(f"{path}/flow.png")
        plt.clf()
    
    def get_jam_stats(self):
        g_hom = self.road_length / self.num_cars
        v_hom = g_hom
        gaps = np.array([
            [
                (self.history_x[j][(i + 1) % self.num_cars] - self.history_x[j][i])
                % self.road_length
                for i in range(self.num_cars)
            ]
            for j in range(self.time)
        ])
        
        traffic_jam_state = self.jam_cars_involved_coeff * self.num_cars
        for i in range(self.time):
            jam_velocity_condition = self.history_v[i] < self.jam_speed_coeff * v_hom
            jam_gap_condition = gaps[i] < self.jam_gap_coeff * g_hom
            if np.sum(jam_gap_condition | jam_velocity_condition) > traffic_jam_state and i > 10:
                return i
        return None

    def _plot_jam_stats(self, path):
        t = np.arange(self.time)
        g_hom = self.road_length / self.num_cars
        v_hom = g_hom
        gaps = np.array([
            [
                (self.history_x[j][(i + 1) % self.num_cars] - self.history_x[j][i])
                % self.road_length
                for i in range(self.num_cars)
            ]
            for j in range(self.time)
        ])
        
        traffic_jam_state = self.jam_cars_involved_coeff * self.num_cars
        jam_cars = np.empty(self.time)
        for i in range(self.time):
            jam_velocity_condition = self.history_v[i] < self.jam_speed_coeff * v_hom
            jam_gap_condition = gaps[i] < self.jam_gap_coeff * g_hom
            jam_cars[i] = np.sum(jam_gap_condition | jam_velocity_condition)
        
        plt.figure(dpi=1200)
        plt.plot(t, jam_cars, color="blue", label="Liczba pojazdów")
        plt.axhline(traffic_jam_state, 0, self.time, color="red", label="Stan korku")
        plt.legend()
        plt.xlabel("Czas")
        plt.ylabel("Liczba samochodów stojących w korku")
        plt.savefig(f"{path}/gaps_statistics.png")
        plt.clf()

    def _plot_history(self, path):
        n = len(self.history_x[0])
        plt.figure(dpi=1200)
        for i in range(self.time):
            plt.scatter(
                self.history_x[i],
                [i] * n,
                c=self.history_v[i],
                cmap="viridis",
                vmax=self.v_max,
                vmin=0,
                s=1,
            )
        plt.colorbar()
        plt.xlim((0, self.road_length))
        plt.gca().invert_yaxis()
        # plt.axis('scaled')
        plt.savefig(f"{path}/simulation.png")
        plt.clf()

def multi_simulation_krauss(number_of_simulations, eps, time):
    avg_velocity = 0
    avg_congestion_time = 0 
    for i in range(number_of_simulations):
        sim = KraussSimulation(i, eps=eps, time=time)
        sim.simulation()
        avg_velocity += np.mean(sim.history_v[20:])
        jam_stats = sim.get_jam_stats()
        if jam_stats == None or avg_congestion_time == None:
            avg_congestion_time = None
        else:
            avg_congestion_time += jam_stats
    if avg_congestion_time == None:
        return avg_velocity/number_of_simulations, None
    else:
        return avg_velocity/number_of_simulations, avg_congestion_time/number_of_simulations

def jam_emergence_and_average_velocity(eps_list, time_list, sim_num=1000):
    velocity_results = np.empty(len(eps_list))
    congestion_results = np.empty(len(eps_list))
    for i in range(len(eps_list)):
        eps = eps_list[i]
        time = time_list[i]
        velocity_results[i], congestion_results[i] = multi_simulation_krauss(sim_num, eps, time)
        print(eps)
    sim_stats = pd.DataFrame()
    sim_stats['eps'] = eps_list
    sim_stats['time'] = time_list
    sim_stats['congestion_time'] = congestion_results
    sim_stats['avg_velocity'] = velocity_results
    sim_stats.to_csv(f'Krauss/stats/congestion_and_velocity_stats.csv')

    ###########################################################

if __name__ == "__main__":
    sim = KraussSimulation(4, eps=0.5, time=10000)
    sim.simulation()
    print(np.mean(sim.history_v))
    print(sim.get_jam_stats())
    # eps_list = [0.5, 0.75, 1]
    # time_list = [1000, 1000, 1000]
    # jam_emergence_and_average_velocity(eps_list, time_list, sim_num=100)
