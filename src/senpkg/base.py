from contextlib import nullcontext
from glob import glob

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import datetime
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.decomposition import PCA
import tensorflow as tf
tf.config.optimizer.set_jit(True)
import pandas as pd
import seaborn as sns
import plotly.express as px
from time import sleep
from rich.console import Console

console = Console()


np.seterr(divide='ignore', invalid='ignore')


class sen:
    colors = ['tomato', 'navy', 'MediumSpringGreen', 'lightblue', 'orange', 'blue',
              'maroon', 'purple', 'yellow', 'olive', 'brown', 'cyan']
    S_sentinel_bands = None
    arr_st = None

    def __init__(self, Path=None):
        self.Path = Path
        
        sen.S_sentinel_bands = glob(self.Path+"/*B?*.tiff")
        sen.S_sentinel_bands.sort()
        l = []
        for i in sen.S_sentinel_bands:
          with rio.open(i, 'r') as f:
            l.append(f.read(1))
        sen.arr_st = np.stack(l)
        
    def dispdata(self):
        
        print(*sen.S_sentinel_bands, sep = "\n")
        print(f'Height: {sen.arr_st.shape[1]}\nWidth: {sen.arr_st.shape[2]}\nBands: {sen.arr_st.shape[0]}')

    def visualizedata(self):
        ep.plot_bands(sen.arr_st, cmap='gist_earth',
                      figsize=(20, 12), cols=6, cbar=False)
        plt.show()

    def RGB(self):
        ep.plot_rgb(sen.arr_st,
                    rgb=(3, 2, 1),
                    figsize=(8, 10),
                    title='RGB Image'
                    )

        plt.show()

    def RGBs(self):
        ep.plot_rgb(sen.arr_st,
                    rgb=(3, 2, 1),
                    figsize=(8, 10),
                    title='RGB Composite Image'
                    )

        plt.show()
        ep.plot_rgb(sen.arr_st,
                    rgb=(3, 2, 1),
                    stretch=True,
                    str_clip=0.2,
                    figsize=(8, 10),
                    title='RGB Composite Image with Stretch Applied'
                    )

        plt.show()

    def hist(self):
        ep.hist(sen.arr_st,
                colors=sen.colors,
                title=[f'Band-{i}' for i in range(1, 13)],
                cols=3,
                alpha=0.5,
                figsize=(12, 10)
                )

        plt.show()
