from matplotlib.pyplot import rcParams
from plots import colors


class NatureStyle():

    def __init__(self,
                 rc_params):
        

        rc_params.update(
            {
                "font.family": "Arial",
                "font.size": 7,              
                "axes.linewidth": 0.6,      
                "xtick.major.width": 0.6,
                "ytick.major.width": 0.6,
                "xtick.major.size": 3,
                "ytick.major.size": 3,
                "lines.linewidth": 1.2,
                "axes.labelsize": 8,
                "axes.titlesize": 8,
                "axes.spines.right": False,
                "axes.spines.top": False
            }
        )

        self.colors = colors.NATURE_COLORS