import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Wykresy do obu modeli dla pojedynczych symulacji
#======================================================
def avg_velocity_plot(model, num):
    
    df = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/velocity.csv')
    plt.figure(dpi=1200)
    plt.plot(df['t'], df['avg_velocity'], label="$v_{śr}$")
    plt.legend()
    plt.xlabel("Czas")
    plt.ylabel("Prędkość")
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/velocity.png')
    plt.clf()
    plt.close()
    
    
def traffic_flow_plot(model, num):
    
    df = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/flow.csv')
    plt.figure(dpi=1200)
    plt.plot(df['t'], df['traffic_flow'], label="Przepływ")
    plt.legend()
    plt.xlabel("Czas")
    plt.ylabel("Przepływ")
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/flow.png')
    plt.clf()
    plt.close()
    
def congestion_plot(model, num):
    
    df = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/congestion.csv')
    plt.figure(dpi=1200)
    plt.plot(df['t'], df['cars_in_congestion'], color="blue", label="Liczba pojazdów")
    plt.axhline(10, 0, max(df['t']), color="red", label="Stan korku")
    plt.legend()
    plt.xlabel("Czas")
    plt.ylabel("Liczba samochodów stojących w korku")
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/congestion.png')
    plt.clf()
    plt.close()

# Wykresy do obu modeli dla wielu symulacji
#======================================================

def multisim_stats_barplot(num, model, eps):
    
    n = len(eps)
    
    velocity = np.empty(n)
    flow = np.empty(n)
    congestion = np.empty(n)
    fuel = np.empty(n)
    
    for i in range(n):
        e = eps[i]
        df = pd.read_csv(f'{model}/multisim/{str(num).rjust(2, "0")}/stats_eps{e}.csv')
        velocity[i] = np.mean(df['velocity'])
        flow[i] = np.mean(df['flow'])
        fuel[i] = np.mean(df['fuel_consumption'])
        if None in df['congestion']:
            congestion[i] = 10**5
        else:
            congestion[i] = np.mean(df['congestion'])
    
    plt.bar(eps, velocity, (max(eps) - min(eps))/len(eps)*0.95)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/velocity_{num}_eps{e}.png')
    plt.clf()
    
    plt.bar(eps, flow, (max(eps) - min(eps))/len(eps)*0.95)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/flow_{num}_eps{e}.png')
    plt.clf()

    plt.bar(eps, congestion, (max(eps) - min(eps))/len(eps)*0.95)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/congestion_{num}_eps{e}.png')
    plt.clf()

    plt.bar(eps, fuel, (max(eps) - min(eps))/len(eps)*0.95)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/fuel_{num}_eps{e}.png')
    plt.clf()
    
    plt.close()
    
# Wykresy porównujące metody
#=====================================================

def flow_comparison(kr_num, rl_num):
    
    df_kr = pd.read_csv(f'Krauss/simulations/{str(kr_num).rjust(2, "0")}/flow.csv')
    df_rl = pd.read_csv(f'RL/simulations/{str(rl_num).rjust(2, "0")}/flow.csv')
    plt.figure(dpi=1200)
    plt.plot(df_kr['t'], df_kr['traffic_flow'], label="Przepływ dla modelu Kraussa")
    plt.plot(df_rl['t'], df_rl['traffic_flow'], label="Przepływ dla modelu z RLearning")
    plt.legend()
    plt.xlabel("Czas")
    plt.ylabel("Przepływ")
    plt.savefig(f'Model_Comparison/kr{str(kr_num).rjust(2, "0")}_rl{str(rl_num).rjust(2, "0")}_flow.png')
    plt.clf()
    plt.close()
    
    
# Wykresy tylko do modelu RL
#=====================================================

def q_table_heatmap(num, v):
    
    df = pd.read_csv(f'RL/simulations/{str(num).rjust(2, "0")}/q_table_{v}.csv').to_numpy()[:,1:]
    
    ticks = np.linspace(0, 21, 6)
    ticks_labels = np.linspace(0, 5, 6)

    ax = sns.heatmap(df, cmap='gray')
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_yticklabels(ticks_labels)
    ax.set_xticklabels(ticks_labels)
    c_bar = ax.collections[0].colorbar
    c_bar.set_ticks([-1, 0, 1])
    c_bar.set_ticklabels(['', 'Medium', 'Aceleeration'])
    plt.xlabel('Odstęp od poprzedniego samochodu')
    plt.ylabel('Prędkość poprzedniego samochodu')
    plt.gca().invert_yaxis()
    plt.savefig(f'RL/simulations/{str(num).rjust(2, "0")}/q_table_{v}.png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    # avg_velocity_plot('Krauss', 1)
    # avg_velocity_plot('RL', 1)
    # traffic_flow_plot('Krauss', 1)
    # traffic_flow_plot('RL', 1)
    # congestion_plot('Krauss', 1)
    # congestion_plot('RL', 1)
    # flow_comparison(1, 1)
    eps = [1, 0.875, 0.75, 0.625, 0.5]
    multisim_stats_barplot(0, 'Krauss', eps)
    
