import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from PIL import Image
import os

# Wykresy do obu modeli dla pojedynczych symulacji
#======================================================
def avg_velocity_plot(model, num):
    
    df = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/velocity.csv')
    # plt.figure(dpi=1200)
    plt.plot(df['t'], df['avg_velocity'], label="$v_{śr}$")
    plt.legend()
    plt.xlabel("Czas")
    plt.ylabel("Prędkość")
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/velocity.png')
    plt.clf()
    plt.close()
    
    
def traffic_flow_plot(model, num):
    
    df = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/flow.csv')
    # plt.figure(dpi=1200)
    plt.plot(df['t'], df['traffic_flow'], label="Przepływ")
    plt.legend()
    plt.xlabel("Czas")
    plt.ylabel("Przepływ")
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/flow.png')
    plt.clf()
    plt.close()
    
def congestion_plot(model, num):
    
    df = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/congestion.csv')
    # plt.figure(dpi=1200)
    plt.plot(df['t'], df['cars_in_congestion'], color="blue", label="Liczba pojazdów")
    plt.axhline(10, 0, max(df['t']), color="red", label="Stan korku")
    plt.legend()
    plt.xlabel("Czas")
    plt.ylabel("Liczba samochodów stojących w korku")
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/congestion.png')
    plt.clf()
    plt.close()
    
def fuel_plot(model, num):
    
    df = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/fuel.csv')
    # plt.figure(dpi=1200)
    plt.plot(df['t'], df['fuel_consumption'], color="blue", label="Zużycie paliwa")
    plt.legend()
    plt.xlabel("Czas")
    plt.ylabel("Zużycie paliwa")
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/fuel.png')
    plt.clf()
    plt.close()
    
def _plot_space(model, num):
    '''
        Tworzy wykres pojedynczego kroku i zapisuje w folderze graphs.
    '''
    dfv = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/history_v.csv').to_numpy()[:,1:]
    dfx = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/history_x.csv').to_numpy()[:,1:]
    time, n = dfv.shape
    for i in range(time):
        plt.scatter(
            dfx[i],
            [0] * n,
            c=dfv[i],
            cmap="viridis",
            vmax=5,
            vmin=0,
            s=1,
        )
        plt.colorbar()
        plt.xlim((0, 200))
        plt.title(f"t = {i}")
        plt.savefig(f'{model}/graphs/{str(i+1).rjust(4, "0")}.png')
        plt.clf()
        plt.close()

def sim_gif(model, num):
    '''
        Tworzy gif dla symulacji na podstawie kroków zapisanych w folderze graphs przy użyciu.
    '''
    _plot_space(model, num)
    source = f"{os.getcwd()}/{model}/graphs"
    images = [Image.open(source + "/" + file) for file in os.listdir(source)]
    images[0].save(
        f"{model}/simulations/{str(num).rjust(2, "0")}/simulation_gif.gif",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )
    for file in os.listdir(f"{os.getcwd()}/Krauss/graphs"):
        os.remove(os.path.join(f"{os.getcwd()}/Krauss/graphs", file))

def history_plot(model, num):
    '''
        Generuje wykres położeń i prędkości dla każdego kroku.
        
        - path: lokalizacja do której zapisywany jest wykres.
    '''
    dfv = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/history_v.csv').to_numpy()[:,1:]
    dfx = pd.read_csv(f'{model}/simulations/{str(num).rjust(2, "0")}/history_x.csv').to_numpy()[:,1:]
    time, n = dfv.shape
    # plt.figure(dpi=1200)
    for i in range(time):
        plt.scatter(
            dfx[i],
            [i] * n,
            c=dfv[i],
            cmap="viridis",
            vmax=5,
            vmin=0,
            s=0.5,
            edgecolors='none'
        )
    plt.colorbar()
    plt.xlim((0, 200))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(aspect=0.5)
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/history.png')
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
    ticks = np.linspace(1, 0.5, 5)

    
    fig, ax = plt.subplots()
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Średnia prędkość")
    plt.bar(eps, velocity, (max(eps) - min(eps))/len(eps)*0.95)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/velocity.png')
    ax.clear()
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Średni przepływ")
    plt.bar(eps, flow, (max(eps) - min(eps))/len(eps)*0.95)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/flow.png')
    ax.clear()

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Średni moment wystąpienia zatoru")
    plt.bar(eps, congestion, (max(eps) - min(eps))/len(eps)*0.95)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/congestion.png')
    ax.clear()

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Średnie zużycie paliwa")
    plt.bar(eps, fuel, (max(eps) - min(eps))/len(eps)*0.95)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/fuel.png')
    ax.clear()
    
    plt.close()
    
# Wykresy porównujące metody
#=====================================================

def flow_comparison(kr_num, rl_num):
    
    df_kr = pd.read_csv(f'Krauss/simulations/{str(kr_num).rjust(2, "0")}/flow.csv')
    df_rl = pd.read_csv(f'RL/simulations/{str(rl_num).rjust(2, "0")}/flow.csv')
    # plt.figure(dpi=1200)
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
    c_bar.set_ticklabels(['Stała prędkość', 'Brak', 'Przyspieszenie'])
    plt.xlabel('Odstęp od poprzedniego samochodu')
    plt.ylabel('Prędkość poprzedniego samochodu')
    plt.gca().invert_yaxis()
    plt.savefig(f'RL/simulations/{str(num).rjust(2, "0")}/q_table_{v}.png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    num = 90
    num2 = 1
    model = 'RL'
    eps = [1, 0.875, 0.75, 0.625, 0.5]
    '''Wykresy pojedyncza symulacja'''
    avg_velocity_plot(model, num)
    traffic_flow_plot(model, num)
    congestion_plot(model, num)
    fuel_plot(model, num)
    history_plot(model, num)
    # sim_gif(model, num)
    
    '''Wykresy porównujące metody'''
    # flow_comparison(num, num)
    
    '''Wykresy dla wielu symulacji'''
    # multisim_stats_barplot(1, model, eps)
    # q_table_heatmap(num, 0.5)
    # q_table_heatmap(num, 1)
    # q_table_heatmap(num, 1.5)
    # q_table_heatmap(num, 2)
    # q_table_heatmap(num, 2.5)
    # q_table_heatmap(num, 3)
    
