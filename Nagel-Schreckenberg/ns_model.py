import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

def nagel_schreckenberg_simulation(size=1000, density=0.1, v_max=5, steps=1000, random_slow_prob=0.5, colored_heatmap=False):
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
        road = nagel_screckenberg_step(step, road, road_history, v_max, size, random_slow_prob)
    plot_time_space(road_history[0:], colored_heatmap)

def generate_road(size, density, v_max):
    '''
        Tworzy jednowymiarową tablicę reprezentującą stan początkowy na drodze. Opis wartości:
            -1 - puste pole
            inna wartość - prędkość samochodu, który znajduje się w danym miejscu
    '''
    road = -np.ones(size).astype(int)
    cars = int(density * size)
    car_placements = np.sort(np.random.permutation(size)[:cars])
    car_base_speed = np.random.randint(0, v_max+1, cars)
    road[car_placements] = car_base_speed
    # road[7] = 4
    # road[24] = 5
    # road[31] = 5
    # road[38] = 5
    # road[47] = 0
    # road[48] = 0
    # road[50] = 1
    # road[58] = 4
    # road[69] = 5

    return road

def nagel_screckenberg_step(step, road, road_history, v_max, size, random_slow_prob):

    car_placements = np.where(road != -1)[0]
    n = len(car_placements)
    distances = np.array([car_placements[(i+1) % n] - car_placements[i] for i in range(n)])
    distances[-1] += size
    for i in range(n):
        x = car_placements[i]
        v = road[x]
        d = distances[i]
        # 1. Acceleration
        if v < v_max and d > v + 1:
            v += 1
        # 2. Slowing down
        elif d <= v and v > 0:
            if d == 0:
                v = 0
            else:
                v = d - 1
        # 3. Randomization
        if v > 0 and np.random.random() < random_slow_prob:
            v -= 1

        road[x] = v

    road_history[step] = road
    road_new_state = -np.ones(size).astype(int)

    for i in range(n):
        x = car_placements[i]
        v = road[x]
        road_new_state[(x + v) % size] = v

    return road_new_state.astype(int)

def plot_time_space(road_history, colored_heatmap):
    if colored_heatmap:
        cmap = colors.ListedColormap(['black', '#9cffe4', '#28fc03', '#9dfc03', '#fce803', '#fc9003', '#fc0303'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    else:
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [-1.5, -0.5, 5.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.pcolor(road_history, cmap=cmap, norm=norm)
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    nagel_schreckenberg_simulation(size=1000, steps=1000)

