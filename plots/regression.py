import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


    
def plot_predictions(
        y_pred, 
        y_true,
        s=10,
        p_value=None,
        bins=20
):

        val_max = max(y_true.max(), y_pred.max())
        val_min = min(y_true.min(), y_pred.min())
        x = np.linspace(0, 13, 100)

        fig = plt.figure(figsize=(15, 15), dpi=1000)

        gs = GridSpec(
        4, 4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.5],   # thinner right histogram
        height_ratios=[0.5, 1, 1, 1],  # shorter top histogram
        wspace=0,
        hspace=0
    )

        ax_main = fig.add_subplot(gs[1:4, 0:3])

        ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        

        ax_main.scatter(y_pred, y_true, c=NATURE_COLORS['red_2'], edgecolors='black', linewidths=0.3, alpha=0.8, zorder=3, s=s)
        ax_main.plot(x, x, c=NATURE_COLORS['grey_1'], zorder=2)
        ax_main.fill_between(x, x+1, x-1, color=NATURE_COLORS['grey_2'], alpha=0.2,edgecolor='none', zorder=1, label='Error < 1')
        ax_main.fill_between(x, x+2, x-2, color=NATURE_COLORS['grey_3'], alpha=0.2,edgecolor='none', zorder=1, label='Error < 2')

        ax_top.hist(y_pred, bins=bins, color=NATURE_COLORS['red_2'], alpha=0.7, edgecolor='black')
        ax_top.axis('off')


        ax_right.hist(y_true, bins=bins, orientation='horizontal', color=NATURE_COLORS['red_2'], alpha=0.7, edgecolor='black')
        ax_right.axis('off')

        ax_main.axis('on')


        ax_top.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="y", labelleft=False)

        ax_main.set_xlabel("Predicted $pK$", fontsize=26)
        ax_main.set_ylabel("Experimental $pK$", fontsize=26)

        #plt.tight_layout()
        ax_main.legend(loc='lower right', fontsize=24)

        stat_text = calc_stats(y_true, y_pred, p_value_model=p_value)

        ax_main.tick_params(
        top=True,
        right=True,
        bottom=True,
        left=True,
        direction='in',
        length=6)


        ax_main.text(0.02, 0.98, stat_text, transform=ax_main.transAxes, ha='left', va='top', fontsize=22)
        ax_main.set_xlim((x.min(), x.max()))
        ax_main.set_ylim((x.min(), max(x.max(), y_pred.max(), y_true.max())))
        ax_main.tick_params(labelsize=22)
        plt.show()