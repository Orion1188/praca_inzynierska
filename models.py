import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

def nagel_schreckenberg_simulation(size=1000, density=0.1, v_max=5, steps=1000, loop=True, random_slow_prob=0.1):
    '''
        Symulacja dla modelu Nagela-Schreckenberga. Parametry:
        size - długość drogi
        density - zagęszczenie ruchu
        v_max - maksymalna prędkość samochodów
        steps - liczba kroków czasowych symulacji
        loop - czy trasa ma być zapętlona
    '''
    road = generate_road(size, density, v_max)
    road_history = np.empty((steps, size))
    for step in range(steps):
        road = nagel_screckenberg_step(step, road, road_history, v_max, size, loop, random_slow_prob)
    print(np.count_nonzero(road_history[0] == -1))
    print(np.count_nonzero(road_history[-1] == 1))
    plot_time_space(road_history[0:])

def generate_road(size, density, v_max):
    '''
        Tworzy jednowymiarową tablicę reprezentującą stan początkowy na drodze. Opis wartości:
            -1 - puste pole
            inna wartość - prędkość samochodu, który znajduje się w danym miejscu
    '''
    road = -np.ones(size)
    cars = int(density * size)
    car_placements = np.random.permutation(size)[:cars]
    car_base_speed = np.random.randint(0, v_max+1, cars)
    road[car_placements] = car_base_speed
    return road.astype(int)

def nagel_screckenberg_step(step, road, road_history, v_max, size, loop, random_slow_prob):
    road_new_state = -np.ones(size)
    car_placements = np.where(road != -1)[0]
    for i in car_placements:
        speed = road[i]
        # 1. Acceleration
        if loop:
            if i+road[i]+2 >= size:
                sight = np.concatenate((road[(i+1):], road[:(i+road[i]+2)%size]))
            else:
                sight = road[(i+1): i+road[i]+2]
        else:
            sight = road[(i+1): min(i+road[i]+2, size)]
        safe_dist_condition = sum(sight) == -len(sight)
        if safe_dist_condition and road[i] < v_max:
            speed += 1
        # 2. Slowing down
        elif not safe_dist_condition and road[i] > 0:
            following = np.where(sight != -1)[0][0]
            speed = max(0, following - 1)
        # 3. Randomization
        if speed > 0 and np.random.random() > random_slow_prob:
            speed -= 1
        road_history[step] = road
        # 4. Update position
        if loop:
            road_new_state[(i + speed) % size] = speed
        else:
            if i + speed < size:
                road_new_state[i + speed] = road[i]
    return road_new_state.astype(int)

def plot_time_space(road_history):
    cmap = colors.ListedColormap(['white', 'black'])
    bounds = [-1.5, -0.5, 5.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.pcolor(road_history, cmap=cmap, norm=norm)
    #sns.heatmap(road_history)
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    nagel_schreckenberg_simulation(size=200, steps=200, density=0.2)
    #size=70, steps=8
        
        
        

        







