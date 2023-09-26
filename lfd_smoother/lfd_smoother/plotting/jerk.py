
import math
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.collections as mcoll
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.default'] = 'regular'

class JerkAnalysis:

    def __init__(self) -> None:
        pass


    def plot_with_high_jerk(self, ts, ys, jerks, bounds = None):
        colors = [
            "darkslateblue", 
            "teal",             
            "limegreen",
            "darkorange",
            "darkorange",
            "darkorange",
            "red",
            "red",
            "red",
            "red",
            "red",
            "red",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
            "darkred"
        ]
        cmap_name = "custom_div_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        # Take the absolute value of jerks
        if bounds is None:
            bounds = np.abs(jerks)
        else:
            bounds = np.abs(bounds)

        abs_jerks = np.abs(jerks)

        # Create line segments and corresponding colors
        points = np.array([ts, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, bounds.max())
        colors = cm(norm(abs_jerks))  # Using the custom colormap
        
        # Plot trajectory with color-coded jerk
        fig, ax = plt.subplots(figsize=(5, 3))
        lc = mcoll.LineCollection(segments, colors=colors, linewidth=4)
        ax.add_collection(lc)
        ax.autoscale()
        
        fig.subplots_adjust(left=0.1)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)  # Using the custom colormap
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, location='left', use_gridspec=True, pad=0.12)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        # Set three rounded ticks for both x and y axes (beginning, middle, and end)
        ax.set_xticks([round(ts.min(), 1), round(np.mean(ts), 1), round(ts.max(), 1)])
        ax.set_yticks([round(ys.min(), 1), round(np.mean(ys), 1), round(ys.max(), 1)])

        # Set three rounded ticks for the colorbar (beginning, middle, and end)
        rounded_bound_max = math.floor(bounds.max() / 1000) * 1000
        cbar.set_ticks([0, rounded_bound_max / 2, rounded_bound_max])

        plt.show()


    def plot_with_low_jerk(self, ts, ys, jerks, bounds = None):
        colors = [
            "darkslateblue", 
            "teal",             
            "limegreen"
        ]
        # colors = ["black", "blue", "cyan", "green", "yellow", "orange", "red", "magenta"]
        cmap_name = "custom_div_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        # Take the absolute value of jerks
        if bounds is None:
            bounds = np.abs(jerks)
        else:
            bounds = np.abs(bounds)

        abs_jerks = np.abs(jerks)

        # Create line segments and corresponding colors
        points = np.array([ts, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, bounds.max())
        colors = cm(norm(abs_jerks))  # Using the custom colormap
        
        # Plot trajectory with color-coded jerk
        fig, ax = plt.subplots(figsize=(5, 3))
        lc = mcoll.LineCollection(segments, colors=colors, linewidth=4)
        ax.add_collection(lc)
        ax.autoscale()
        
        fig.subplots_adjust(left=0.1)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)  # Using the custom colormap
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, location='left', use_gridspec=True, pad=0.12)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        # Set three rounded ticks for both x and y axes (beginning, middle, and end)
        ax.set_xticks([round(ts.min(), 1), round(np.mean(ts), 1), round(ts.max(), 1)])
        ax.set_yticks([round(ys.min(), 1), round(np.mean(ys), 1), round(ys.max(), 1)])

        # Set three rounded ticks for the colorbar (beginning, middle, and end)
        rounded_bound_max = math.floor(bounds.max() / 10) * 10
        cbar.set_ticks([0, rounded_bound_max / 2, rounded_bound_max])

        plt.show()

