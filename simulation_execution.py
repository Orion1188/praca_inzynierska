import model
import plots
import numpy as np

if __name__ == '__main__':
    # sim_kr = model.TrafficSimulation('Krauss', 2, time=5000)
    # sim_rl = model.TrafficSimulation('RL', 1, time=5000)
    # sim_kr.simulation()
    # sim_rl.simulation()
    # print(np.mean(np.array([max(sim_kr.history_v[i]) for i in range(sim_kr.time)])))
    eps = [1, 0.875, 0.75, 0.625, 0.5]
    for e in eps:
        model.multi_simulation(0, 'Krauss', 5, e, 10**3)