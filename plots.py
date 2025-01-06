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
    plt.figure(figsize=(6.8, 12.8), dpi=120)
    for i in range(time):
        plt.scatter(
            dfx[i],
            [i] * n,
            c=dfv[i],
            cmap="viridis",
            vmax=5,
            vmin=0,
            s=3,
            edgecolors='none'
        )
    plt.colorbar().set_label('Prędkość', fontsize = 20)
    plt.xlim((0, 200))
    plt.ylim((0, 1000))
    plt.xlabel('Droga', fontsize = 20)
    plt.ylabel('Czas', fontsize = 20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.gca().set_aspect(aspect=0.5)
    plt.savefig(f'{model}/simulations/{str(num).rjust(2, "0")}/history.png')
    plt.clf()
    plt.close()


# Wykresy do obu modeli dla wielu symulacji
#======================================================

def multisim_stats_barplot(num, model, eps, param):
    
    n = len(eps)
    
    velocity = np.empty(n)
    flow = np.empty(n)
    congestion = np.empty(n)
    fuel = np.empty(n)
    
    for i in range(n):
        e = eps[i]
        df = pd.read_csv(f'{model}/multisim/{str(num).rjust(2, "0")}/stats_{param}{e}.csv')
        velocity[i] = np.mean(df['velocity'])
        flow[i] = np.mean(df['flow'])
        fuel[i] = np.mean(df['fuel_consumption'])
        if not np.any(df['congestion']):
            congestion[i] = 15000
        else:
            congestion[i] = int(np.mean(df['congestion']))
    ticks = eps
    ticksl = ticks
    ticks = [1, 2, 3, 4, 5]
    offset = 6

    
    fig, ax = plt.subplots()
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticksl)
    plt.title('Prędkość - model Kraussa')
    # plt.title('Prędkość - model z alg. Q-learning niezależnym od nagród')
    # plt.title('Prędkość - model z alg. Q-learning zależnym od nagród')
    # plt.xlabel("$\epsilon$")
    # plt.xlabel('m')
    plt.xlabel('$\\tau$, $\Delta t$')
    plt.ylabel("Średnia prędkość")
    plt.ylim(0, max(velocity)*1.1)
    plt.bar(ticks, velocity, (max(ticks) - min(ticks))/len(ticks)*0.95)
    for i in range(len(velocity)):
        plt.text(ticks[i] - abs(ticks[0] - ticks[1])/offset,round(velocity[i]+0.02*max(velocity), 2),round(velocity[i], 2)) # do 6, 7
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/velocity.png')
    ax.clear()
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticksl)
    plt.title('Przepływ - model Kraussa')
    # plt.title('Przepływ - model z alg. Q-learning niezależnym od nagród')
    # plt.title('Przepływ - model z alg. Q-learning zależnym od nagród')
    # plt.xlabel("$\epsilon$")
    # plt.xlabel('m')
    plt.xlabel('$\\tau$, $\Delta t$')
    plt.ylabel("Średni przepływ")
    plt.ylim(0, max(flow)*1.1)
    plt.bar(ticks, flow, (max(ticks) - min(ticks))/len(ticks)*0.95)
    for i in range(len(flow)):
        plt.text(ticks[i] - abs(ticks[0] - ticks[1])/offset,flow[i]+0.02*max(flow),round(flow[i], 2)) # 6, 7
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/flow.png')
    ax.clear()

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticksl)
    plt.title('Moment wystąpienia korka - model Kraussa')
    # plt.title('Moment wystąpienia korka - model z alg. Q-learning niezależnym od nagród', fontsize=11)
    # plt.title('Moment wystąpienia korka - model z alg. Q-learning zależnym od nagród')
    # plt.xlabel("$\epsilon$")
    # plt.xlabel('m')
    plt.xlabel('$\\tau$, $\Delta t$')
    plt.ylabel("Średni moment wystąpienia korka")
    plt.ylim(0, max(congestion)*1.1)
    plt.bar(ticks, congestion, (max(ticks) - min(ticks))/len(ticks)*0.95)
    for i in range(len(congestion)):
        if congestion[i] == 15000:
            plt.text(ticks[i] - abs(ticks[0] - ticks[1])/offset,congestion[i]+0.02*max(congestion),'>15000', fontsize=6)
        else:
            plt.text(ticks[i] - abs(ticks[0] - ticks[1])/(offset+1),congestion[i]+0.02*max(congestion), int(round(congestion[i], 0)), fontsize=7)
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/congestion.png')
    ax.clear()

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticksl)
    plt.title('Zużycie paliwa - model Kraussa')
    # plt.title('Zużycie paliwa - model z alg. Q-learning niezależnym od nagród')
    # plt.title('Zużycie paliwa - model z alg. Q-learning zależnym od nagród')
    # plt.xlabel("$\epsilon$")
    # plt.xlabel('m')
    plt.xlabel('$\\tau$, $\Delta t$')
    plt.ylabel("Średnie zużycie paliwa")
    plt.ylim(0, max(fuel)*1.1)
    plt.bar(ticks, fuel, (max(ticks) - min(ticks))/len(ticks)*0.95)
    for i in range(len(fuel)):
        plt.text(ticks[i] - abs(ticks[0] - ticks[1])/offset,round(fuel[i]+0.02*max(fuel), 2),round(fuel[i], 2))
    plt.savefig(f'{model}/multisim/{str(num).rjust(2, "0")}/fuel.png')
    ax.clear()
    
    plt.close()
    
# Wykresy porównujące metody
#=====================================================

def flow_comparison(kr_num, rl_num, rl2_num):
    
    df_kr = pd.read_csv(f'Krauss/simulations/{str(kr_num).rjust(2, "0")}/flow.csv')[::100]
    df_rl = pd.read_csv(f'RL/simulations/{str(rl_num).rjust(2, "0")}/flow.csv')[::100]
    df_rl2 = pd.read_csv(f'RL/simulations/{str(rl2_num).rjust(2, "0")}/flow.csv')[::100]
    # plt.figure(dpi=1200)
    plt.plot(df_kr['t'], df_kr['traffic_flow'], label="Model Kraussa", alpha=0.75)
    plt.plot(df_rl['t'], df_rl['traffic_flow'], label="Model z alg. Q-Learning (niezależny od nagród)", alpha=0.75)
    plt.plot(df_rl2['t'], df_rl2['traffic_flow'], label="Model z alg. Q-Learning (zależny od nagród)", alpha=0.75)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Czas")
    plt.ylabel("Przepływ")
    plt.savefig(f'Model_Comparison/kr{str(kr_num).rjust(2, "0")}_rl{str(rl_num).rjust(2, "0")}_rl{str(rl2_num).rjust(2, "0")}_flow.png')
    plt.clf()
    plt.close()

def fuel_comparison(kr_num, rl_num, rl2_num):
    
    df_kr = pd.read_csv(f'Krauss/simulations/{str(kr_num).rjust(2, "0")}/fuel.csv')[::100]
    df_rl = pd.read_csv(f'RL/simulations/{str(rl_num).rjust(2, "0")}/fuel.csv')[::100]
    df_rl2 = pd.read_csv(f'RL/simulations/{str(rl2_num).rjust(2, "0")}/fuel.csv')[::100]
    # plt.figure(dpi=1200)
    plt.plot(df_kr['t'], df_kr['fuel_consumption'], label="Model Kraussa", alpha=0.75)
    plt.plot(df_rl['t'], df_rl['fuel_consumption'], label="Model z alg. Q-Learning (niezależny od nagród)", alpha=0.75)
    plt.plot(df_rl2['t'], df_rl2['fuel_consumption'], label="Model z alg. Q-Learning (zależny od nagród)", alpha=0.75)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Czas")
    plt.ylabel("Zużycie paliwa")
    plt.savefig(f'Model_Comparison/kr{str(kr_num).rjust(2, "0")}_rl{str(rl_num).rjust(2, "0")}_rl{str(rl2_num).rjust(2, "0")}_fuel.png')
    plt.clf()
    plt.close()
    
def congestion_comparison(kr_num, rl_num, rl2_num):
    
    df_kr = pd.read_csv(f'Krauss/simulations/{str(kr_num).rjust(2, "0")}/congestion.csv')[::100]
    df_rl = pd.read_csv(f'RL/simulations/{str(rl_num).rjust(2, "0")}/congestion.csv')[::100]
    df_rl2 = pd.read_csv(f'RL/simulations/{str(rl2_num).rjust(2, "0")}/congestion.csv')[::100]
    # plt.figure(dpi=1200)
    plt.plot(df_kr['t'], df_kr['cars_in_congestion'], label="Model Kraussa", alpha=0.75)
    plt.plot(df_rl['t'], df_rl['cars_in_congestion'], label="Model z alg. Q-Learning (niezależny od nagród)", alpha=0.75)
    plt.plot(df_rl2['t'], df_rl2['cars_in_congestion'], label="Model z alg. Q-Learning (zależny od nagród)", alpha=0.75)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Czas")
    plt.ylabel("Liczba samochodów w korku")
    plt.savefig(f'Model_Comparison/kr{str(kr_num).rjust(2, "0")}_rl{str(rl_num).rjust(2, "0")}_rl{str(rl2_num).rjust(2, "0")}_congestion.png')
    plt.clf()
    plt.close()
    
    
# Wykresy tylko do modelu RL
#=====================================================

def q_table_heatmap(num, v):
    
    df = pd.read_csv(f'RL/simulations/{str(num).rjust(2, "0")}/q_table_{v}.csv').to_numpy()[:,1:]
    
    ticks = np.linspace(0, 21, 6)
    ticks_labels = np.linspace(0, 5, 6)

    ax = sns.heatmap(df, vmin=-1, vmax=1, cmap='gray')
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_yticklabels(ticks_labels)
    ax.set_xticklabels(ticks_labels)
    c_bar = ax.collections[0].colorbar
    c_bar.set_ticks([-1, 0, 1])
    c_bar.set_ticklabels(['Stała prędkość', 'Brak', 'Przyspieszenie'])
    plt.xlabel('Odstęp od kolejnego samochodu')
    plt.ylabel('Prędkość kolejnego samochodu')
    plt.title(f'v = {v}')
    plt.gca().invert_yaxis()
    plt.savefig(f'RL/simulations/{str(num).rjust(2, "0")}/q_table_{v}.png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    # num = 0
    # num2 = 1
    # model = 'Krauss'
    eps = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ms = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    dts = [0.25, 0.5, 1, 2, 4]
    
    '''Wykresy pojedyncza symulacja'''
    # history_plot('RL', 1)
    # history_plot('RL', 0)
    # history_plot('Krauss', 0)
    
    '''Wykresy porównujące metody'''
    # flow_comparison(2, 2, 5)
    # flow_comparison(3, 3, 6)
    # flow_comparison(4, 4, 7)
    # fuel_comparison(2, 2, 5)
    # fuel_comparison(3, 3, 6)
    # fuel_comparison(4, 4, 7)
    # congestion_comparison(2, 2, 5)
    # congestion_comparison(3, 3, 6)
    # congestion_comparison(4, 4, 7)
    
    '''Wykresy dla wielu symulacji'''
    # multisim_stats_barplot(0, 'Krauss', eps, 'eps') # ok
    # multisim_stats_barplot(2, 'RL', eps, 'eps') # ok
    # multisim_stats_barplot(3, 'RL', eps, 'eps') # ok
    
    # multisim_stats_barplot(6, 'Krauss', ms, 'm') # ok
    # multisim_stats_barplot(6, 'RL', ms, 'm') # ok
    # multisim_stats_barplot(7, 'RL', ms, 'm') # ok
    
    multisim_stats_barplot(8, 'Krauss', dts, 'dt')
    # multisim_stats_barplot(8, 'RL', dts, 'dt') # ok
    # multisim_stats_barplot(9, 'RL', dts, 'dt') # ok
    
    '''Wykresy Q-table'''    
    # num = 98
    # q_table_heatmap(num, 0.5)
    # q_table_heatmap(num, 1)
    # q_table_heatmap(num, 1.5)
    # q_table_heatmap(num, 2)
    # q_table_heatmap(num, 2.5)
    # q_table_heatmap(num, 3)
    
