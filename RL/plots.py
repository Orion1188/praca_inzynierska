import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def velocity_and_congestion_barplot():
    
    df = pd.read_csv(f'RL/stats/congestion_and_velocity_stats.csv')
    
    plt.figure(dpi=1200)
    plt.bar(df['eps'], df['avg_velocity'])
    plt.savefig(f'RL/stats/velocity_barplot.png')
    plt.clf()
    
    plt.bar(df['eps'], df['congestion_time'])
    plt.savefig(f'RL/stats/congestion_barplot.png')
    plt.clf()
    
def q_table_heatmap(num, v):
    
    df = pd.read_csv(f'RL/simulations/{str(num).rjust(2, "0")}/q_table_{v}.csv')
    
    plt.figure(dpi=1200)
    sns.heatmap(df)
    plt.savefig(f'RL/simulations/{str(num).rjust(2, "0")}/q_table_{v}.png')
    plt.clf()

if __name__ == '__main__':
    # velocity_and_congestion_barplot()
    for v in [0.75, 1.5, 2.5, 2]:
        q_table_heatmap(13, v)