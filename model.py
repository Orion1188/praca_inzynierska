import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

class TrafficSimulation:
    def __init__(
        self,
        model,
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
        reward_independent=True,
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
            - model - model, na podstawie którego zostanie przeprowadzona symulacja ('Krauss' lub 'RL')
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
        self.model = model
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
        self.x = np.linspace(0, self.road_length, self.num_cars, endpoint=False)
        self.v = np.zeros(self.num_cars)
        self.history_x = np.empty((self.time, self.num_cars))
        self.history_v = np.empty((self.time, self.num_cars))
        self.v_disc = np.linspace(0, v_max, v_disc)
        self.v_prev_disc = np.linspace(0, v_max, v_prev_disc)
        # self.gap_disc = np.linspace(0, road_length, gap_disc)
        self.gap_disc = np.linspace(0, 5, gap_disc)
        self.reward_independent = reward_independent
        self.q_table_steady = np.zeros((v_disc, v_prev_disc, gap_disc))
        self.q_table_accelerate = np.zeros((v_disc, v_prev_disc, gap_disc))
        self.jam_speed_coeff = jam_speed_coeff
        self.jam_gap_coeff = jam_gap_coeff
        self.jam_cars_involved_coeff = jam_cars_involved_coeff
        
        
    def simulation(self):
        '''
            Wykonuje symulację dla parametrów zadanych dla instancji klasy.
            Jeśli only_stats==False zostaną dodatkowo wygenerowane wykresy dla symulacji.
            Jeśli create_gif==True zostanie wygenerowany gif całej symulacji (działa pod warunkiem, że only_stats==False).
        '''
        if self.model == 'Krauss':
            self.krauss_simulation()
        elif self.model == 'RL':
            self.rl_simulation()
        else:
            raise ValueError('Wrong model')
        path = f'{os.getcwd()}/{self.model}/simulations/{str(self.num).rjust(2, "0")}'
        if not os.path.isdir(path):
            os.mkdir(path)
        self.velocity_stats(path)
        self.traffic_flow_stats(path)
        self.congestion_stats(path)
    
    
    def krauss_simulation(self):
        '''
            Wykonuje zadaną liczbę kroków symulacji modelu Kraussa.
        '''
        for i in range(self.time):
            self.x, self.v = self.krauss_step(i)
            print(i)
        
        
    def krauss_step(self, step):
        '''
            Pojedynczy krok symulacji modelu Kraussa.
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


    def rl_simulation(self):
        '''
            Wykonuje zadaną liczbę kroków symulacji z użyciem reinforcement learning.
        '''
        for i in range(self.time):
            self.x, self.v = self.rl_step(i)
            print(i)


    def rl_step(self, step):
        '''
            Pojedynczy krok symulacji z użyciem reinforcement learning.
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

        # Opcja z uśrednieniem nagrody (z artykułu)
        
        if self.reward_independent:
            reward = np.mean(np.array([v_new[(i - 1) % n] - self.history_v[step][(i - 1) % n] for i in range(n)]))
            for i in range(n):
                if actions[i] == "steady":
                    self.q_table_steady[states[i][0], states[i][1], states[i][2]] = (1 - self.alpha) * self.q_table_steady[states[i][0], states[i][1], states[i][2]] + self.alpha * (reward + self.gamma * self.q_table_steady[states_new[i][0], states_new[i][1], states_new[i][2]])
                elif actions[i] == "accelerate":
                    self.q_table_accelerate[states[i][0], states[i][1], states[i][2]] = (1 - self.alpha) * self.q_table_accelerate[states[i][0], states[i][1], states[i][2]] + self.alpha * (reward + self.gamma * self.q_table_accelerate[states_new[i][0], states_new[i][1], states_new[i][2]])
        
        # Opcja z uwzględnieniem każdej nagrody osobno (moja)
        else:
            for i in range(n):
                if actions[i] == "steady":
                    reward = v_new[(i - 1) % n] - self.history_v[step][(i - 1) % n]
                    self.q_table_steady[states[i][0], states[i][1], states[i][2]] = (1 - self.alpha) * self.q_table_steady[states[i][0], states[i][1], states[i][2]] + self.alpha * (reward + self.gamma * self.q_table_steady[states_new[i][0], states_new[i][1], states_new[i][2]])
                elif actions[i] == "accelerate":
                    reward = v_new[(i - 1) % n] - self.history_v[step][(i - 1) % n]
                    self.q_table_accelerate[states[i][0], states[i][1], states[i][2]] = (1 - self.alpha) * self.q_table_accelerate[states[i][0], states[i][1], states[i][2]] + self.alpha * (reward + self.gamma * self.q_table_accelerate[states_new[i][0], states_new[i][1], states_new[i][2]])

        return x_new, v_new
    
    
    def velocity_stats(self, path):
        '''
            Generuje wykres prędkości maksymalnej, minimalnej i średniej.
            
            - path: lokalizacja do której zapisywany jest wykres.
        '''
        t = np.arange(self.time)
        avg_speed = [np.mean(self.history_v[i]) for i in range(self.time)]
        df = pd.DataFrame()
        df['t'] = t
        df['avg_velocity'] = avg_speed
        df.to_csv(f"{path}/velocity.csv")
        
        
    def traffic_flow_stats(self, path):
        '''
            Generuje wykres przepływu.
            
            - path: lokalizacja do której zapisywany jest wykres.
        '''
        t = np.arange(self.time)
        avg_speed = np.array([np.mean(self.history_v[i]) for i in range(self.time)])
        traffic_flow = avg_speed * self.num_cars / self.road_length
        df = pd.DataFrame()
        df['t'] = t
        df['traffic_flow'] = traffic_flow
        df.to_csv(f"{path}/flow.csv")
        
        
    def fuel_consumption_stats(self, path):
        '''
            Generuje wykres zużycia paliwa na podstawie wzoru z artykułu.
            
            - path: lokalizacja do której zapisywany jest wykres.
        '''
        t = np.arange(self.time)
        f_cons = lambda v: 2 * v**2 - 2 * v + 2 + 1 / v
        avg_fuel_consumption = np.array([np.mean(f_cons(self.history_v[i])) for i in range(self.time)])
        df = pd.DataFrame()
        df['t'] = t
        df['fuel_consumption'] = avg_fuel_consumption
        df.to_csv(f"{path}/fuel.csv")
        

    def congestion_stats(self, path):
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
        
        jam_cars = np.empty(self.time)
        for i in range(self.time):
            jam_velocity_condition = self.history_v[i] < self.jam_speed_coeff * v_hom
            jam_gap_condition = gaps[i] < self.jam_gap_coeff * g_hom
            jam_cars[i] = np.sum(jam_gap_condition | jam_velocity_condition)
        
        df = pd.DataFrame()
        df['t'] = t
        df['cars_in_congestion'] = jam_cars
        df.to_csv(f"{path}/congestion.csv")
        
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
        
        
    def congestion_stats_multisim(self):
        '''
            Generuje statystyki dotyczące liczby samochodów w korku w każdym kroku na podstawie parametrów:
            - jam_speed_coeff
            - jam_gap_coeff
            - jam_cars_involved_coeff 
            
            Zwraca krok czasowy, w którym wystąpił korek. Jeśli w symulacji nie wystąpił korek zostaje zwrócone None.
        '''
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


def multi_simulation(num, mod, number_of_simulations, eps, time):
    '''
        Wykonuje wiele symulacji dla zadanego modelu, epsilona i liczby kroków i zwraca odpowiednie statystyki.
    '''
    path = f'{os.getcwd()}/{mod}/multisim/{str(num).rjust(2, "0")}'
    if not os.path.isdir(path):
        os.mkdir(path)
        
    avg_velocity = np.empty(number_of_simulations)
    avg_flow = np.empty(number_of_simulations)
    congestion_time = np.empty(number_of_simulations)
    fuel_consumption = np.empty(number_of_simulations)
    f_cons = lambda v: 2 * v**2 - 2 * v + 2 + 1 / v
    
    for i in range(number_of_simulations):
        sim = TrafficSimulation(mod, 99, eps=eps, time=time)
        sim.simulation()
        avg_velocity[i] = np.mean(sim.history_v)
        avg_flow[i] = np.sum(sim.history_v) * sim.num_cars / sim.road_length / time
        congestion_time[i] = sim.congestion_stats_multisim()
        fuel_consumption[i] = np.mean(np.array([np.mean(f_cons(sim.history_v[i])) for i in range(sim.time)]))
    
    df = pd.DataFrame()
    df['velocity'] = avg_velocity
    df['flow'] = avg_flow
    df['congestion'] =  congestion_time
    df['fuel_consumption'] = fuel_consumption
    df.to_csv(f'{mod}/multisim/{str(num).rjust(2, "0")}/stats_eps{eps}.csv')


def jam_emergence_and_average_velocity(num, mod, eps_list, time_list, sim_num=1000):
    '''
        Zwraca plik csv zawierający informację o średnim momencie wystąpienia korku oraz średniej prędkości dla zadanej listy parametrów eps oraz zadanej liczby kroków.
    '''
    velocity_results = np.empty(len(eps_list))
    congestion_results = np.empty(len(eps_list))
    for i in range(len(eps_list)):
        eps = eps_list[i]
        time = time_list[i]
        velocity_results[i], congestion_results[i] = multi_simulation(mod, sim_num, eps, time)
        print(eps)
    sim_stats = pd.DataFrame()
    sim_stats['eps'] = eps_list
    sim_stats['time'] = time_list
    sim_stats['congestion_time'] = congestion_results
    sim_stats['avg_velocity'] = velocity_results
    sim_stats.to_csv(f'Krauss/multisim/{str(num).rjust(2, "0")}/congestion_and_velocity_stats.csv')

