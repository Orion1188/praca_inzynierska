import model

if __name__ == '__main__':    
    
    # sim_kr = model.TrafficSimulation('Krauss', 0, time=1000)
    # sim_kr.simulation()
    # sim_rl = model.TrafficSimulation('RL', 0, time=1000)
    # sim_rl.simulation()
    # sim_rl = model.TrafficSimulation('RL', 1, time=1000, reward_independent=False)
    # sim_rl.simulation()
    
    # sim = model.TrafficSimulation('Krauss', 99, time=500, jam_gap_coeff=0.56).simulation()
    
    # time = 15000
    # sim = model.TrafficSimulation('Krauss', 2, time=time).simulation()
    # sim = model.TrafficSimulation('Krauss', 3, time=time, eps=0.75).simulation()
    # sim = model.TrafficSimulation('Krauss', 4, time=time, eps=0.5).simulation()
    # sim = model.TrafficSimulation('RL', 2, time=time).simulation()
    # sim = model.TrafficSimulation('RL', 3, time=time, eps=0.75).simulation()
    # sim = model.TrafficSimulation('RL', 4, time=time, eps=0.5).simulation()
    # sim = model.TrafficSimulation('RL', 5, time=time, reward_independent=False).simulation()
    # sim = model.TrafficSimulation('RL', 6, time=time, eps=0.75, reward_independent=False).simulation()
    # sim = model.TrafficSimulation('RL', 7, time=time, eps=0.5, reward_independent=False).simulation()

    
    '''Pojedyncza symulacja dla Kraussa'''
    # sim_kr = model.TrafficSimulation('Krauss', 0, time=1000)
    # sim_kr.simulation()
    
    '''Pojedyncza symulacja dla rl'''
    
    # t = 10**6
    # timer_start = time.time()
    # sim_rl = model.TrafficSimulation('RL', 97, time=t, reward_independent=False)
    # sim_rl.simulation()
    # sim_rl.q_table_data(0.5)
    # sim_rl.q_table_data(1)
    # sim_rl.q_table_data(1.5)
    # sim_rl.q_table_data(2)
    # sim_rl.q_table_data(2.5)
    # sim_rl.q_table_data(3)
    
    # sim_rl = model.TrafficSimulation('RL', 98, time=t)
    # sim_rl.simulation()
    # sim_rl.q_table_data(0.5)
    # sim_rl.q_table_data(1)
    # sim_rl.q_table_data(1.5)
    # sim_rl.q_table_data(2)
    # sim_rl.q_table_data(2.5)
    # sim_rl.q_table_data(3)
    
    # model.multi_simulation(1, 'Krauss', 50, eps=0.75, time=10000)
    
    '''Wiele symulacji dla zadanych eps Krauss'''
    # eps = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # for e in eps:
    #     model.multi_simulation(0, 'Krauss', 20, e, 1, 100, 15000, rew_ind=True)
    #     model.multi_simulation(2, 'RL', 20, e, 1, 100, 15000, rew_ind=True, cutoff=2000)
    #     model.multi_simulation(3, 'RL', 20, e, 1, 100, 15000, rew_ind=False, cutoff=2000)
    
    # ms = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    # for m in ms:
    #     model.multi_simulation(6, 'Krauss', 20, 0.5, 1, m, 15000, rew_ind=True)
    #     model.multi_simulation(6, 'RL', 20, 0.5, 1, m, 15000, rew_ind=True, cutoff=2000)
    #     model.multi_simulation(7, 'RL', 20, 0.5, 1, m, 15000, rew_ind=False, cutoff=2000)
    
    # dts = [0.25, 0.5, 1, 2, 4]
    dts = [2, 4]
    for dt in dts:
        print('dt =', dt)
        model.multi_simulation(8, 'Krauss', 20, 0.5, dt, 100, 15000, rew_ind=True)
        model.multi_simulation(8, 'RL', 20, 0.5, dt, 100, 15000, rew_ind=True, cutoff=2000)
        model.multi_simulation(9, 'RL', 20, 0.5, dt, 100, 15000, rew_ind=False, cutoff=2000)
    
    '''Wiele symulacji dla zadanych eps RL'''
    # eps = [1, 0.875, 0.75, 0.625, 0.5]
    # for e in eps:
    #     model.multi_simulation(num, 'RL', 10, e, 10**4)