import model

if __name__ == '__main__':
    # mod = 'RL'
    # eps = [1, 0.75, 0.5]
    # for i in range(len(eps)):
    #     sim_kr = model.TrafficSimulation(mod, i+3, time=10**4, eps=eps[i])
    #     sim_kr.simulation()
    
    '''Pojedyncza symulacja dla Kraussa'''
    sim_kr = model.TrafficSimulation('RL', 90, time=1000, dt=0.25, reaction_time=0.25)
    sim_kr.simulation()
    
    '''Pojedyncza symulacja dla rl'''
    # sim_rl = model.TrafficSimulation('RL', 91, time=10**6)
    # sim_rl.simulation()
    # sim_rl.q_table_data(0.5)
    # sim_rl.q_table_data(1)
    # sim_rl.q_table_data(1.5)
    # sim_rl.q_table_data(2)
    # sim_rl.q_table_data(2.5)
    # sim_rl.q_table_data(3)
    
    num = 1
    '''Wiele symulacji dla zadanych eps Krauss'''
    # eps = [1, 0.875, 0.75, 0.625, 0.5]
    # for e in eps:
    #     model.multi_simulation(num, 'Krauss', 10, e, 10**4)
    
    '''Wiele symulacji dla zadanych eps RL'''
    # eps = [1, 0.875, 0.75, 0.625, 0.5]
    # for e in eps:
    #     model.multi_simulation(num, 'RL', 10, e, 10**4)