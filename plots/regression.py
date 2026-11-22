import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from plots.colors import NATURE_COLORS

from scipy.stats import pearsonr, spearmanr

from typing import TypedDict

from sklearn.metrics import (
        r2_score,
        root_mean_squared_error as rmse,
        mean_absolute_error as mae
)


def rp_calc(y_test: np.ndarray, y_hat: np.ndarray):
    return pearsonr(y_test, y_hat)[0]


def rs_calc(y_test: np.ndarray, y_hat: np.ndarray):
    return spearmanr(y_test, y_hat)[0]


class Metrics(TypedDict):
        rp: float
        rs: float
        r2: float
        rmse: float
        mae: float


def assess_prediction(
        y_true: np.ndarray,
        y_hat: np.ndarray
) -> Metrics:      
        rp = rp_calc(y_true, y_hat)
        rs = rs_calc(y_true, y_hat)
        r2 = r2_score(y_true, y_hat)
        rmse_val = rmse(y_true, y_hat)
        mae_val = mae(y_true, y_hat)

        return Metrics(rp=rp, rs=rs, r2=r2, rmse=rmse_val, mae=mae_val)



    
def plot_predictions(
        y_pred, 
        y_true,
        title,
        save_path,
        s=10,
        bins=20
):

        evaluation = assess_prediction(y_true, y_pred)
        evaluation_text = (f"$Rp = {evaluation['rp']:.3f}$, "
                   f"$Rs = {evaluation['rs']:.3f}$, "
                   f"$R^2 = {evaluation['r2']:.3f}$, "
                   f"$RMSE = {evaluation['rmse']:.3f}$, "
                   f"$MAE = {evaluation['mae']:.3f}$"
        )

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

        ax_top.set_title(
              f"{title}\n"
              f"Performance Metrics: {evaluation_text}\n\n",
              fontsize=20
        )


        ax_right.hist(y_true, bins=bins, orientation='horizontal', color=NATURE_COLORS['red_2'], alpha=0.7, edgecolor='black')
        ax_right.axis('off')

        ax_main.axis('on')

        ax_top.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="y", labelleft=False)

        ax_main.set_xlabel("Predicted $pK$", fontsize=26)
        ax_main.set_ylabel("Experimental $pK$", fontsize=26)

        ax_main.legend(loc='lower right', fontsize=24)

        ax_main.tick_params(
        top=True,
        right=True,
        bottom=True,
        left=True,
        direction='in',
        length=6)

        ax_main.set_xlim((x.min(), x.max()))
        ax_main.set_ylim((x.min(), max(x.max(), y_pred.max(), y_true.max())))
        ax_main.tick_params(labelsize=22)
        plt.savefig(save_path)