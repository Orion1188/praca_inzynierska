import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def velocity_and_congestion_barplot():
    
    df = pd.read_csv(f'Krauss/stats/congestion_and_velocity_stats.csv')
    
    plt.bar(df['eps'], df['avg_velocity'])
    plt.savefig(f'Krauss/stats/velocity_barplot.png')
    plt.clf()
    
    plt.bar(df['eps'], df['congestion_time'])
    plt.savefig(f'Krauss/stats/congestion_barplot.png')
    plt.clf()
    


if __name__ == '__main__':
    velocity_and_congestion_barplot()