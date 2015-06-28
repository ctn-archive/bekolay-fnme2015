import json
import os
import subprocess
from collections import OrderedDict
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import svgutils.transform as sg
from scipy import stats

root = os.path.dirname(__file__)
plots_dir = os.path.realpath(os.path.join(root, os.pardir, 'plots'))
results_dir = os.path.realpath(os.path.join(root, os.pardir, 'results'))
noprobes_dir = os.path.join(results_dir, 'noprobes')
results_dir = os.path.join(results_dir, 'probes')
fig_dir = os.path.realpath(os.path.join(root, os.pardir, 'figures'))
# Change if not on Mac OS X
inkscape = os.path.expanduser("~/Applications/Inkscape.app/Contents"
                              "/Resources/bin/inkscape")

onecolumn = 3.34646  # inches
twocolumn = 7.08661  # inches


class RectElement(sg.FigureElement):
    def __init__(self, x, y):
        s = 18
        rect = sg.etree.Element(sg.SVG+"rect",
                                {"x": str(x), "y": str(y - s),
                                 "width": str(s), "height": str(s),
                                 "style": "fill:white;"})
        sg.FigureElement.__init__(self, rect)


def el(char, path, x, y, scale=1, offset=(10, 30)):
    toret = []
    if char is not None:
        toret.append(RectElement(x + offset[0], y + offset[1]))
        toret.append(sg.TextElement(x + offset[0],
                                    y + offset[1],
                                    char,
                                    size=24,
                                    weight='bold',
                                    font='Arial'))
    if path.endswith(".svg"):
        svg = sg.fromfile(path)
        svg = svg.getroot()
        svg.moveto(str(x), str(y), scale)
        return [svg] + toret


def svgfig(w, h):
    w = str(w)
    h = str(h)
    return sg.SVGFigure(w, h)


def savefig(fig, out):
    svg_path = os.path.join(fig_dir, '%s.svg' % out)
    pdf_path = os.path.join(fig_dir, '%s.pdf' % out)
    # eps_path = os.path.join(fig_dir, '%s.eps' % out)
    fig.save(svg_path)
    subprocess.call([inkscape, '--export-pdf=%s' % pdf_path, svg_path])
    # subprocess.call([inkscape, '--export-text-to-path',
    #                  '--export-eps=%s' % eps_path, svg_path])


def setup(figsize=None):
    plt.close('all')
    sns.set_style("white")
    sns.set_style("ticks")
    plt.figure(figsize=figsize)


def prettify(despine=None, tight_layout=None):
    despine = {} if despine is None else despine
    tight_layout = {} if tight_layout is None else tight_layout
    sns.despine(**despine)
    plt.tight_layout(**tight_layout)


def save(fname, ext="svg", probes=True, despine=None, tight_layout=None):
    prettify(despine, tight_layout)
    fname = fname if probes else "%s-np" % fname
    plt.savefig(os.path.join(plots_dir, "%s.%s" % (fname, ext)))


def get_data(task, key, separate_cols=True, probes=True):
    data_dir = results_dir if probes else noprobes_dir

    if separate_cols:
        data = OrderedDict()
    else:
        data = []
    dirs = [d for d in os.listdir(data_dir)
            if '.sim' in d]
    sims = [os.path.basename(d).split('.')[0] for d in dirs]

    for dir_, sim in zip(dirs, sims):
        if separate_cols:
            data[sim] = []

        paths = glob(os.path.join(data_dir, dir_, "test_%s*.txt" % task))
        for path in paths:
            with open(path, 'r') as fp:
                try:
                    if separate_cols:
                        data[sim].append(json.load(fp)[key])
                    else:
                        data.append((sim, json.load(fp)[key]))
                except:
                    print(path)
                    raise
    return pd.DataFrame(data, columns=['Backend', key])


def plot_summary(task, key, probes=True, figsize=None, rotation=0):
    setup(figsize=figsize)
    data = get_data(task, key, probes=probes)
    sns.boxplot(data, showfliers=False)
    plt.gca().set_xticklabels(data.columns, rotation=rotation)


def accuracy():
    plot_summary("cchannelchain", "rmse", figsize=(onecolumn * 2, 4.0))
    plt.ylabel("RMSE")
    save("accuracy-1")

    plot_summary("product", "rmse", figsize=(onecolumn * 2, 4.0))
    plt.ylabel("RMSE")
    save("accuracy-2")

    plot_summary("controlledoscillator", "score",
                 figsize=(twocolumn * 0.66, 3.0), rotation=12)
    plt.ylabel("FFT similarity")
    save("accuracy-3")

    plot_summary("sequence", "timing_mean", figsize=(onecolumn * 2, 3.0))
    plt.ylabel("Mean transition time (s)")
    save("accuracy-4")


def speed(probes=True):
    d_args = {'separate_cols': False, 'probes': probes}
    t1 = get_data("cchannelchain", "buildtime", **d_args)
    t1['Model'] = "Chained channels"
    t2 = get_data("product", "buildtime", **d_args)
    t2['Model'] = "Product"
    t3 = get_data("controlledoscillator", "buildtime", **d_args)
    t3['Model'] = "Oscillator"
    t4 = get_data("sequence", "buildtime", **d_args)
    t4['Model'] = "BG sequence"
    build = pd.concat((t1, t2, t3, t4))

    setup()
    plt.subplot(1, 2, 1)
    sns.factorplot('Model', 'buildtime', 'Backend', data=build,
                   legend_out=False,
                   x_order=["Chained channels",
                            "Product",
                            "Oscillator",
                            "BG sequence"])
    plt.gcf().set_size_inches(onecolumn * 2, 3.0)
    plt.ylabel("Build time (s)")
    plt.xlabel("")
    save("fig5", probes=probes, ext='pdf')

    d_args = {'separate_cols': False, 'probes': probes}
    t1 = get_data("cchannelchain", "runtime", **d_args)
    t1['runtime'] /= 0.5  # => times real time
    t1['Model'] = "Chained channels"
    t2 = get_data("product", "runtime", **d_args)
    t2['runtime'] /= 5.5  # => times real time
    t2['Model'] = "Product"
    t3 = get_data("controlledoscillator", "runtime", **d_args)
    t3['runtime'] /= 10.0  # => times real time
    t3['Model'] = "Oscillator"
    t4 = get_data("sequence", "runtime", **d_args)
    t4['runtime'] /= 4.0  # => times real time
    t4['Model'] = "BG sequence"
    run = pd.concat((t1, t2, t3, t4))

    setup()
    sns.factorplot('Model', 'runtime', 'Backend', data=run,
                   legend_out=False, x_order=["Chained channels",
                                              "Product",
                                              "Oscillator",
                                              "BG sequence"])
    plt.gcf().set_size_inches(onecolumn * 2, 3.0)
    plt.axhline(1.0, lw=1, c='k', ls=':')
    plt.ylim(top=8.0)
    plt.ylabel("Real time / Run time")
    plt.xlabel("")
    save("fig6", probes=probes, ext='pdf')


def fig1():
    w = onecolumn * 2 * 72
    h = 3.9 * 72

    fig = svgfig(w, h * 2)
    fig.append(el(None, 'plots/results-1.svg', 0, 0))
    fig.append(el(None, 'plots/accuracy-1.svg', 0, h))
    savefig(fig, 'fig1')


def fig2():
    w = onecolumn * 2 * 72
    h = 3.9 * 72

    fig = svgfig(w, h * 2)
    fig.append(el(None, 'plots/results-2.svg', 0, 0))
    fig.append(el(None, 'plots/accuracy-2.svg', 0, h))
    savefig(fig, 'fig2')


def fig3():
    w = twocolumn * 2 * 72
    h = 3.0 * 72

    fig = svgfig(w, h)
    fig.append(el(None, 'plots/results-3.svg', 0, 0))
    fig.append(el(None, 'plots/accuracy-3.svg', twocolumn * 1.33 * 72, 0))
    savefig(fig, 'fig3')


def fig4():
    w = onecolumn * 2 * 72
    h = 2.9 * 72

    fig = svgfig(w, h * 2)
    fig.append(el(None, 'plots/results-4.svg', 0, 0))
    fig.append(el(None, 'plots/accuracy-4.svg', 0, h))
    savefig(fig, 'fig4')
