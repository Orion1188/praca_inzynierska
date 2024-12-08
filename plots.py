import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import model

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

def velocity_and_congestion_barplot(model):
    
    df = pd.read_csv(f'{model}/stats/congestion_and_velocity_stats.csv')
    plt.bar(df['eps'], df['avg_velocity'])
    plt.savefig(f'{model}/stats/velocity_barplot.png')
    plt.clf()
    plt.bar(df['eps'], df['congestion_time'])
    plt.savefig(f'{model}/stats/congestion_barplot.png')
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
    avg_velocity_plot('Krauss', 1)
    avg_velocity_plot('RL', 1)
    traffic_flow_plot('Krauss', 1)
    traffic_flow_plot('RL', 1)
    congestion_plot('Krauss', 1)
    congestion_plot('RL', 1)
    flow_comparison(1, 1)
