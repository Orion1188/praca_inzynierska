import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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
        create_gif=False,
        jam_speed_coeff=0.2,
        jam_gap_coeff=0.2,
        jam_cars_involved_coeff=0.1,
        only_stats=True,
        history_plot=False
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
            - v_disc: liczba przedziałów, na które dzielimy możliwe prędkości dla obecnego agenta. Potrzebna do wyznaczenia stanu.
            - v_prev_disc: liczba przedziałów, na które dzielimy możliwe prędkości dla poprzedniego agenta. Potrzebna do wyznaczenia stanu.
            - gap_disc: liczba przedziałów, na które dzielimy odstęp pomiędzy obecnym a poprzednim agentem. Potrzebna do wyznaczenia stanu.
            
            Q tabele:
            - q_table_steady - tabela wartości Q dla wszystkich możliwych stanów i akcji "steady", czyli jazdy ze stałą prędkością.
            - q_table_accelerate - tabela wartości Q dla wszystkich możliwych stanów i akcji "accelerate", czyli przyspieszenia.
            
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
        self.alpha = alpha
        self.gamma = gamma
        self.x, self.v = self._generate_cars()
        self.history_x = np.empty((self.time, self.num_cars))
        self.history_v = np.empty((self.time, self.num_cars))
        self.v_disc = np.linspace(0, v_max, v_disc)
        self.v_prev_disc = np.linspace(0, v_max, v_prev_disc)
        # self.gap_disc = np.linspace(0, road_length, gap_disc)
        self.gap_disc = np.linspace(0, 5, gap_disc)
        self.q_table_steady = np.zeros((v_disc, v_prev_disc, gap_disc))
        self.q_table_accelerate = np.zeros((v_disc, v_prev_disc, gap_disc))
        self.jam_speed_coeff = jam_speed_coeff
        self.jam_gap_coeff = jam_gap_coeff
        self.jam_cars_involved_coeff = jam_cars_involved_coeff
        self.only_stats = only_stats
        self.create_gif = create_gif
        self.history_plot = history_plot

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
        self._qlearning_simulation()
        path = f'{os.getcwd()}/RL/simulations/{str(self.num).rjust(2, "0")}'
        os.mkdir(path)
        if not self.only_stats:
            if self.create_gif:
                self._plot_space()
                self._plot_to_gif(path)
            self._plot_speed_stats(path)
            self._plot_traffic_flow_stats(path)
            self._plot_jam_stats(path)
            if self.history_plot:
                self._plot_history(path)

    def _qlearning_simulation(self):
        '''
            Wykonuje zadaną liczbę kroków symulacji.
        '''
        for i in range(self.time):
            self.x, self.v = self._qlearning_step(i)
            print(i)

    def _qlearning_step(self, step):
        '''
            Pojedynczy krok symulacji.
        '''
        n = self.num_cars
        
        v_new = np.empty(n)
        x_new = np.empty(n)
        
        # Wyznaczam odstępy pomiędzy samochodami
        gaps = np.array([(self.x[(i + 1) % n] - self.x[i]) % self.road_length for i in range(n)])
        
        # Tablice w których zapiszę stan obecny, przyszły i wykonaną akcję dla każdego samochodu
        states = [None]*n
        states_new = [[None]*3]*n
        actions = [None]*n
        
        for i in range(n):

            # Wyznaczenie współrzędnych obecnego stanu w Q-table
            states[i] = [
                np.searchsorted(self.v_disc, self.v[i]),
                np.searchsorted(self.v_prev_disc, self.v[(i - 1) % n]),
                np.searchsorted(self.gap_disc, min(gaps[(i - 1) % n], 4.99))
            ]

            # Wyznaczenie prędkości przy założeniu, że przyspieszymy/będziemy zmuszeni żeby zwolnić
            full_stop_time = (self.v[i] + self.v[(i + 1) % n]) / (2 * self.br_coeff)
            gap_des = self.reaction_time * self.v[(i + 1) % n]
            v_safe = self.v[(i + 1) % n] + (gaps[i] - gap_des) / (
                self.reaction_time + full_stop_time
            )
            v_des = min([self.v_max, self.v[i] + self.acc_coeff, v_safe])

            # Podjęcie decyzji. Jeśli v_des jest większe od obecnej prędkości to decyzja zostanie podjęta na podstawie Q-table. 
            if v_des > self.v[i]:
                if (
                    self.q_table_accelerate[states[i][0], states[i][1], states[i][2]]
                    > self.q_table_steady[states[i][0], states[i][1], states[i][2]]
                ):
                    actions[i] = "accelerate"                       # Wybieram akcję
                    states_new[i][0] = np.searchsorted(self.v_disc, v_des)           
                    v_new[i] = v_des                            # Ustalam prędkość w kolejnym kroku
                elif (
                    self.q_table_accelerate[states[i][0], states[i][1], states[i][2]]
                    == self.q_table_steady[states[i][0], states[i][1], states[i][2]]
                ):          # Jeśli wartości Q są równe dla obu akcji to losuję akcję
                    if np.random.random() > 0.5:
                        actions[i] = "accelerate"
                        states_new[i][0] = np.searchsorted(self.v_disc, v_des),
                        v_new[i] = v_des
                    else:
                        actions[i] = "steady"
                        states_new[i][0] = np.searchsorted(self.v_disc, self.v[i])
                        v_new[i] = self.v[i]
                else:
                    actions[i] = "steady"
                    states_new[i][0] = np.searchsorted(self.v_disc, self.v[i])
                    v_new[i] = self.v[i]

            else: # W przeciwnym wypadku jesteśmy zmuszeni zwolnić - decyzja jest wymuszona.
                actions[i] = "slowdown"
                v_new[i] = max(0, v_des - np.random.uniform(0, self.eps))

        
        
        # Zapisanie poprzednich prędkości i położeń w historii

        self.history_v[step] = self.v
        self.history_x[step] = self.x % self.road_length

        # Wprowadzenie nowych położeń

        for i in range(n):
            x_new[i] = (self.x[i] + v_new[i]) % self.road_length
            
        # Wyznaczenie nowych odstępów

        gaps_new = np.array([(x_new[(i + 1) % n] - x_new[i]) % self.road_length for i in range(n)])
            
        # Wprowadzenie nowych stanów
        
        for i in range(n):
            states_new[i][1] = np.searchsorted(self.v_prev_disc, v_new[(i - 1) % n])
            states_new[i][2] = np.searchsorted(self.gap_disc, min(gaps_new[(i - 1) % n], 4.99))

        # aktualizacja Q-table

        # Opcja z uwzględnieniem każdej nagrody osobno (moja)
        
        # for i in range(n):
        #     if action == "steady":
        #         reward = v_new[(i - 1) % n] - self.history_v[step][(i - 1) % n]
        #         self.q_table_steady[state[0], state[1], state[2]] = (1 - self.alpha) * self.q_table_steady[state[0], state[1], state[2]] + self.alpha * (reward + self.gamma * self.q_table_steady[state_new[0], state_new[1], state_new[2]])
        #     elif action == "accelerate":
        #         reward = v_new[(i - 1) % n] - self.history_v[step][(i - 1) % n]
        #         self.q_table_accelerate[state[0], state[1], state[2]] = (1 - self.alpha) * self.q_table_accelerate[state[0], state[1], state[2]] + self.alpha * (reward + self.gamma * self.q_table_accelerate[state_new[0], state_new[1], state_new[2]])

        # Opcja z uśrednieniem nagrody (z artykułu)
        reward = np.mean(np.array([v_new[(i - 1) % n] - self.history_v[step][(i - 1) % n] for i in range(n)]))
        for i in range(n):
            if actions[i] == "steady":
                self.q_table_steady[states[i][0], states[i][1], states[i][2]] = (1 - self.alpha) * self.q_table_steady[states[i][0], states[i][1], states[i][2]] + self.alpha * (reward + self.gamma * self.q_table_steady[states_new[i][0], states_new[i][1], states_new[i][2]])
            elif actions[i] == "accelerate":
                self.q_table_accelerate[states[i][0], states[i][1], states[i][2]] = (1 - self.alpha) * self.q_table_accelerate[states[i][0], states[i][1], states[i][2]] + self.alpha * (reward + self.gamma * self.q_table_accelerate[states_new[i][0], states_new[i][1], states_new[i][2]])
        
        return x_new, v_new

    def get_jam_stats(self):
        '''
            Generuje statystyki dotyczące liczby samochodów w korku w każdym kroku na podstawie parametrów:
            - jam_speed_coeff
            - jam_gap_coeff
            - jam_cars_involved_coeff 
            
            Zwraca krok czasowy, w którym wystąpił korek. Jeśli w symulacji nie wystąpił korek zostaje zwrócone None.
        '''
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
        for i in range(self.time):
            jam_velocity_condition = self.history_v[i] < self.jam_speed_coeff * v_hom
            jam_gap_condition = gaps[i] < self.jam_gap_coeff * g_hom
            if np.sum(jam_gap_condition & jam_velocity_condition) > traffic_jam_state and i > 10:
                print(i)
                return i
        return None

    def _plot_space(self):
        '''
            Tworzy wykres pojedynczego kroku i zapisuje w folderze graphs.
        '''
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
            plt.title(f"t = {i}")
            plt.savefig(f'RL/graphs/{str(i+1).rjust(4, "0")}.png')
            plt.clf()

    def _plot_to_gif(self, path):
        '''
            Tworzy gif dla symulacji na podstawie kroków zapisanych w folderze graphs przy użyciu.
        '''
        source = f"{os.getcwd()}/RL/graphs"
        images = [Image.open(source + "/" + file) for file in os.listdir(source)]
        images[0].save(
            f"{path}/simulation_gif.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
        for file in os.listdir(f"{os.getcwd()}/RL/graphs"):
            os.remove(os.path.join(f"{os.getcwd()}/RL/graphs", file))

    def _plot_speed_stats(self, path):
        '''
            Generuje wykres prędkości maksymalnej, minimalnej i średniej.
            
            - path: lokalizacja do której zapisywany jest wykres.
        '''
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
        plt.savefig(f"{path}/speed_statistics.png")
        plt.clf()

    def _plot_traffic_flow_stats(self, path):
        '''
            Generuje wykres przepływu.
            
            - path: lokalizacja do której zapisywany jest wykres.
        '''
        t = np.arange(self.time)
        plt.figure(dpi=1200)
        avg_speed = np.array([np.mean(self.history_v[i]) for i in range(self.time)])
        traffic_flow = avg_speed * self.num_cars / self.road_length
        plt.plot(t, traffic_flow, color="red", label="Przepływ")
        plt.legend()
        plt.xlabel("Czas")
        plt.ylabel("Przepływ")
        plt.savefig(f"{path}/flow.png")
        plt.clf()

    def _plot_jam_stats(self, path):
        '''
            Generuje wykres liczby samochodów stojących w korku.
            
            - path: lokalizacja do której zapisywany jest wykres.
        '''
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
            jam_cars[i] = np.sum(jam_gap_condition & jam_velocity_condition)
        
        plt.figure(dpi=1200)
        plt.plot(t, jam_cars, color="blue", label="Liczba pojazdów")
        plt.axhline(traffic_jam_state, 0, self.time, color="red", label="Stan korku")
        plt.legend()
        plt.xlabel("Czas")
        plt.ylabel("Liczba samochodów stojących w korku")
        plt.savefig(f"{path}/gaps_statistics.png")
        plt.clf()

    def _plot_history(self, path):
        '''
            Generuje wykres położeń i prędkości dla każdego kroku.
            
            - path: lokalizacja do której zapisywany jest wykres.
        '''
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
        plt.savefig(f"{path}/simulation.png")
        plt.clf()
    
    def q_table_data(self, v):
        '''
            Zwraca plik csv zawierający informację, która akcja okazała się korzystniejsza dla każdego ze stanów dla zadanej prędkości obecnego agenta.
        '''
        index = np.searchsorted(self.v_disc, v)
        acc = self.q_table_accelerate[index]
        ste = self.q_table_steady[index]
        chosen_option = np.ones(np.shape(acc))
        chosen_option[acc == ste] = 0
        chosen_option[acc < ste] = -1
        df = pd.DataFrame(chosen_option)
        # df.index = self.v_prev_disc
        df.to_csv(f'RL/simulations/{str(self.num).rjust(2, "0")}/q_table_{v}.csv')        
    
        
def multi_simulation_rl(number_of_simulations, eps, time):
    '''
        Wykonuje wiele symulacji dla zadanego eps i liczby kroków i zwraca odpowiednie statystyki.
    '''
    avg_velocity = 0
    avg_congestion_time = 0 
    for i in range(number_of_simulations):
        sim = QLearningSimulation(i, eps=eps, time=time)
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
    '''
        Zwraca plik csv zawierający informację o średnim momencie wystąpienia korku oraz średniej prędkości dla zadanej listy parametrów eps oraz zadanej liczby kroków.
    '''
    velocity_results = np.empty(len(eps_list))
    congestion_results = np.empty(len(eps_list))
    for i in range(len(eps_list)):
        eps = eps_list[i]
        time = time_list[i]
        velocity_results[i], congestion_results[i] = multi_simulation_rl(sim_num, eps, time)
        print(eps)
    sim_stats = pd.DataFrame()
    sim_stats['eps'] = eps_list
    sim_stats['time'] = time_list
    sim_stats['congestion_time'] = congestion_results
    sim_stats['avg_velocity'] = velocity_results
    sim_stats.to_csv(f'Krauss/stats/congestion_and_velocity_stats.csv')

    ###########################################################

if __name__ == "__main__":
    
    sim = QLearningSimulation(2, time=5000, only_stats=False)
    sim.simulation()
    # sim.q_table_data(0.75)
    # sim.q_table_data(1.5)
    # sim.q_table_data(2)
    # sim.q_table_data(2.5)
    # QLearningSimulation(1, time=1000, only_stats=False, history_plot=True).simulation()
    
