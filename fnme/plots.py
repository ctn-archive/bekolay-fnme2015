import json
import os
from collections import OrderedDict
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

root = os.path.dirname(__file__)
plots_dir = os.path.realpath(os.path.join(root, os.pardir, 'plots'))
results_dir = os.path.realpath(os.path.join(root, os.pardir, 'results'))


def setup():
    sns.set_style("white")
    sns.set_style("ticks")
    plt.rcParams['figure.figsize'] = (6.0, 3.0)
    plt.clf()


def prettify(despine=None, tight_layout=None):
    despine = {} if despine is None else despine
    tight_layout = {} if tight_layout is None else tight_layout
    sns.despine(**despine)
    plt.tight_layout(**tight_layout)


def compliance():
    backends = ['nengo', 'nengo_ocl', 'nengo_distilled', 'nengo_brainstorm']
    # Just copied for now
    compl = [212, 187, 129, 107]

    setup()
    plt.bar(np.arange(len(compl)), compl, align="center")
    plt.xticks(np.arange(len(compl)), backends)
    plt.ylabel("Tests passed")
    prettify()

    plt.savefig(os.path.join(plots_dir, 'compliance.svg'), transparent=True)


def get_data(task, key):
    data = OrderedDict()
    dirs = [d for d in os.listdir(results_dir)
            if '.sim' in d]
    sims = [os.path.basename(d).split('.')[0] for d in dirs]

    for dir_, sim in zip(dirs, sims):
        data[sim] = []
        paths = glob(os.path.join(results_dir, dir_, "test_%s*.txt" % task))
        for path in paths:
            with open(path, 'r') as fp:
                data[sim].append(json.load(fp)[key])
    return pd.DataFrame(data)


def plot_bar(task, key):
    setup()
    data = get_data(task, key)
    plt.figure()
    means = data.mean()
    errors = data.std()
    means.plot(yerr=errors, kind='bar')
    plt.gca().set_xticklabels(data.columns, rotation=0)


def save_bar(fname):
    prettify()
    plt.savefig(os.path.join(plots_dir, fname), transparent=True)
    plt.close('all')


def accuracy():
    plot_bar("cchannelchain", "rmse")
    plt.title("Communication channel chain RSME (lower is better)")
    plt.ylabel("RMSE")
    save_bar("accuracy-1.svg")

    plot_bar("product", "rmse")
    plt.title("Product RSME (lower is better)")
    plt.ylabel("RMSE")
    save_bar("accuracy-2.svg")

    plot_bar("controlledoscillator", "score")
    plt.title("Controlled oscillator frequency similarity (higher is better)")
    plt.ylabel("Similarity")
    save_bar("accuracy-3.svg")

    plot_bar("sequencememory", "prob_60000")
    plt.title("Probability of decoding correct term (higher is better)")
    plt.ylabel("Probability")
    save_bar("accuracy-4.svg")

    plot_bar("sequencememory", "prob_1000000000")
    plt.title("Probability of decoding correct term (higher is better)")
    plt.ylabel("Probability")
    save_bar("accuracy-5.svg")


def speed():
    plot_bar("cchannelchain", "buildtime")
    plt.title("Communication channel chain build time")
    plt.ylabel("Time (seconds)")
    save_bar("build-1.svg")

    plot_bar("product", "buildtime")
    plt.title("Product build time")
    plt.ylabel("Time (seconds)")
    save_bar("build-2.svg")

    plot_bar("controlledoscillator", "buildtime")
    plt.title("Controlled oscillator build time")
    plt.ylabel("Time (seconds)")
    save_bar("build-3.svg")

    plot_bar("sequencememory", "buildtime")
    plt.title("SPA sequence memory build time")
    plt.ylabel("Time (seconds)")
    save_bar("build-4.svg")

    plot_bar("cchannelchain", "runtime")
    plt.title("Communication channel chain run time")
    plt.ylabel("Time (seconds)")
    save_bar("run-1.svg")

    plot_bar("product", "runtime")
    plt.title("Product run time")
    plt.ylabel("Time (seconds)")
    save_bar("run-2.svg")

    plot_bar("controlledoscillator", "runtime")
    plt.title("Controlled oscillator run time")
    plt.ylabel("Time (seconds)")
    save_bar("run-3.svg")

    plot_bar("sequencememory", "runtime")
    plt.title("SPA sequence memory run time")
    plt.ylabel("Time (seconds)")
    save_bar("run-4.svg")
