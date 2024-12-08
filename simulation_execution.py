import model
import plots

if __name__ == '__main__':
    sim_kr = model.TrafficSimulation('Krauss', 1, time=5000)
    sim_rl = model.TrafficSimulation('RL', 1, time=5000)
    sim_kr.simulation()
    sim_rl.simulation()