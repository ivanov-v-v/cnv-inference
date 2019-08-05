import os
import pickle
import seaborn as sns
import sys

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pickle_dump(obj, path):
    with open(path, "wb") as outfile:
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.SparseDataFrame):
            obj.reset_index(drop=True, inplace=True)
        pickle.dump(obj, outfile)

        
def pickle_load(path):
    with open(path, "rb") as infile:
        return pickle.load(infile)
    
    
def nan_fraction(df, axis_list=[], 
                 verbose=False, 
                 show_plots=False, 
                 suptitle="",
                 **plot_kwargs):
    
    assert np.setdiff1d(axis_list, [0,1]).size == 0
    
    ostream = sys.stdout if verbose else open(os.devnull, "w")
    result_dict = {}
    
    for axis in axis_list:
        axis_name = "column" if axis == 0 else "row"
        result_dict[axis_name] = df.isna().mean(axis=axis)
        print(f"Per {axis_name}:", file=ostream)
        print(result_dict[axis_name], file=ostream)
        
    if show_plots:
        assert len(axis_list) > 0
        sns.set(font_scale=1.2)
        fig, axes = np.atleast_1d(
            plt.subplots(1, len(axis_list), 
                         figsize=(8 * len(axis_list), 8))
        )
        fig.suptitle(suptitle)
        for i, axis in enumerate(axis_list):
            axis_name = "column" if axis == 0 else "row"
            ax = axes[i]
            ax.set_title(f"Fraction of NaNs in a {axis_name}")
            sns.distplot(result_dict[axis_name].values, 
                         ax=ax, **plot_kwargs)
            
    result_dict["total"] = df.isna().sum().sum() / df.size
    print("In total: ", result_dict["total"], file=ostream)
    return result_dict


def filter_by_isin(df, target_col, keep_list):
    return df[df[target_col].isin(keep_list)]


def is_sorted(iterable: Iterable):
    return np.all(iterable == sorted(iterable))