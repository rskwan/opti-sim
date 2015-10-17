from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import csv

def scenario_plot(results, attitudes, horizons, num_episodes, fname):
    """Plots the scenario using the data in RESULTS, ATTITUDES, and HORIZONS,
    saving the plot to FNAME."""
    margin_prop = 0.05
    # initialize colors
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    tableau20 = [(r/255, g/255, b/255) for (r, g, b) in tableau20]
    # set up plot
    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(111)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xlim(margined(np.array(horizons), margin_prop))
    plt.ylim(margined_pm(results[:, :, 1], results[:, :, 2], margin_prop))
    plt.xticks(horizons)
    plt.xlabel('Horizon')
    plt.ylabel('Average Earnings')
    plt.title('Performance for {0} Episodes'.format(num_episodes))
    lines = []
    for attitude_idx, attitude in enumerate(attitudes):
        data = results[attitude_idx]
        avgs = data[:,1]
        stderrs = data[:,2]
        color = tableau20[attitude_idx]
        plt.errorbar(horizons, avgs, stderrs, color=color, elinewidth=1, lw=1.5)
    plt.legend(attitudes, loc=4, fontsize=12)
    plt.savefig(fname)
    plt.close()

def make_existing_plot(origname, destname):
    """Reads the results file in ORIGNAME and saves a plot of it to DESTNAME."""
    # pass 1: populate attitudes and horizons
    reader = csv.reader(open(origname, 'rb'))
    attitudes = {}
    horizons = set()
    first = True
    for row in reader:
        if first:
            first = False
            continue
        horizons.add(int(row[0]))
        attitudes[int(row[3])] = row[4]
    attitudes = [attitudes[i] for i in range(len(attitudes))]
    horizons = sorted(horizons)
    # pass 2: populate results
    reader = csv.reader(open(origname, 'rb'))
    results = np.zeros((len(attitudes), len(horizons), 3))
    first = True
    for row in reader:
        if first:
            first = False
            continue
        horizon_idx = horizons.index(int(row[0]))
        attitude_idx = int(row[3])
        results[int(row[3])][horizon_idx] = row[:3]
    scenario_plot(results, attitudes, horizons, destname)


