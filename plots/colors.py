import matplotlib.pyplot as plt


NATURE_COLORS = {
    'red_1': '#6e1416',
    'red_2': '#982b2b',
    'red_3': '#c74546',
    'red_4': '#db6968',
    'red_5': '#ea9c9d',
    'red_6': '#fbc9c4',
    'blue_1': '#0074b3',
    'blue_2': '#4d97cd',
    'blue_3': '#88c4e8',
    'blue_4': '#b9e5fa',
    'grey_1': '#3d4e6a',
    'grey_2': '#606f8a',
    'grey_3': '#8b96ad',
    'grey_4': '#bdc3d2',
    'yellow': '#e8c559',
    'cyan': '#46c1be',
    'orange': '#f8984e',
    'purple': '#b379b4'
}


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,              # Nature prefers 5-7 pt
    "axes.linewidth": 0.6,       # Thin but visible
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    'axes.spines.right': False,
    'axes.spines.top': False
    
})