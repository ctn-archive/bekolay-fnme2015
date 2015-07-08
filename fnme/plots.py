import json
import os
import subprocess
from collections import OrderedDict
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import svgutils.transform as sg
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

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


def el(char, path, x, y, scale=1, offset=(4, 24)):
    toret = []
    if char is not None:
        toret.append(RectElement(x + offset[0], y + offset[1]))
        toret.append(sg.TextElement(x + offset[0],
                                    y + offset[1],
                                    char,
                                    size=24,
                                    weight='bold',
                                    font='Arial'))
    if path is not None and path.endswith(".svg"):
        svg = sg.fromfile(path)
        svg = svg.getroot()
        svg.moveto(str(x), str(y), scale)
        toret = [svg] + toret
    return toret


def svgfig(w, h):
    w = str(w)
    h = str(h)
    return sg.SVGFigure(w, h)


def savefig(fig, out):
    svg_path = os.path.join(fig_dir, '%s.svg' % out)
    fig.save(svg_path)
    svg2other(out)


def svg2other(out):
    svg_path = os.path.join(fig_dir, '%s.svg' % out)
    pdf_path = os.path.join(fig_dir, '%s.pdf' % out)
    eps_path = os.path.join(fig_dir, '%s.eps' % out)
    subprocess.call([inkscape, '--export-pdf=%s' % pdf_path, svg_path])
    subprocess.call([inkscape, '--export-text-to-path',
                     '--export-eps=%s' % eps_path, svg_path])


def setup(figsize=None, palette_args=None):
    plt.close('all')
    sns.set_style("white")
    sns.set_style("ticks")
    if palette_args is None:
        palette_args = {"palette": "cubehelix_r", "n_colors": 6}
    sns.set_palette(**palette_args)
    plt.figure(figsize=figsize)


def save(fname, fig=False):
    sns.despine()
    plt.tight_layout()
    if fig:
        plt.savefig(os.path.join(fig_dir, "%s.svg" % fname))
    else:
        plt.savefig(os.path.join(plots_dir, "%s.svg" % fname))


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
    if separate_cols:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame(data, columns=['Backend', key])


def plot_summary(task, key, probes=True, figsize=None, rotation=0):
    setup(figsize=figsize)
    data = get_data(task, key, probes=probes)
    sns.boxplot(data=data, showfliers=False)
    plt.gca().set_xticklabels(data.columns, rotation=rotation)


def accuracy():
    plot_summary("cchannelchain", "rmse", figsize=(onecolumn * 2, 4.0))
    plt.ylabel("RMSE")
    save("accuracy-1")

    plot_summary("product", "rmse", figsize=(onecolumn * 2, 4.0))
    plt.ylabel("RMSE")
    save("accuracy-2")

    plot_summary("controlledoscillator", "score", figsize=(onecolumn * 2, 4.0))
    plt.ylabel("FFT similarity")
    save("accuracy-3")

    plot_summary("sequence", "timing_mean", figsize=(onecolumn * 2, 3.0))
    plt.ylabel("Mean transition time (s)")
    save("accuracy-4")

    plot_summary("sequence_pruned", "timing_mean", figsize=(onecolumn * 2, 3.0))
    plt.ylabel("Mean transition time (s)")
    save("accuracy-4-pruned")


def speed(probes=True):
    # Only get build speed for probed data
    if probes:
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

        setup(figsize=(onecolumn * 2, 3.0))
        ax = plt.subplot(1, 1, 1)
        sns.barplot(x='Model', y='buildtime', hue='Backend', data=build, ax=ax,
                    order=["Chained channels",
                           "Product",
                           "Oscillator",
                           "BG sequence"])
        plt.ylabel("Build time (s)")
        plt.xlabel("")
        save("fig5", fig=True)
        svg2other("fig5")

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

    setup(figsize=(onecolumn * 2, 3.0))
    ax = plt.subplot(1, 1, 1)
    sns.barplot(x='Model', y='runtime', hue='Backend', data=run, ax=ax,
                order=["Chained channels",
                       "Product",
                       "Oscillator",
                       "BG sequence"])
    plt.axhline(1.0, lw=1, c='k', ls=':')
    plt.ylim(top=8.0)
    plt.ylabel("Simulation time + overhead / Run time")
    plt.xlabel("")

    inset = zoomed_inset_axes(plt.gca(), 0.22,
                              bbox_to_anchor=(0.76, 0.98),
                              bbox_transform=plt.gcf().transFigure)
    sns.barplot(x='Model', y='runtime', hue='Backend', data=run, ax=inset,
                order=["Chained channels",
                       "Product",
                       "Oscillator",
                       "BG sequence"])
    plt.legend([])
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(())
    plt.xlim(left=2.5)
    pp, _, _ = mark_inset(ax, inset, loc1=3, loc2=4, fc="none", ec="0.5")
    pp.set_visible(False)
    sns.despine(bottom=True, ax=inset)

    fname = "fig6" if probes else "fig7"
    save(fname, fig=True)
    svg2other(fname)


def fig1():
    w = onecolumn * 2 * 72
    h = 3.9 * 72

    fig = svgfig(w, h * 2)
    fig.append(el("A", 'plots/results-1.svg', 0, 0))
    fig.append(el("B", 'plots/accuracy-1.svg', 0, h))
    savefig(fig, 'fig1')


def fig2():
    w = onecolumn * 2 * 72
    h = 3.9 * 72

    fig = svgfig(w, h * 2)
    fig.append(el("A", 'plots/results-2.svg', 0, 0))
    fig.append(el("B", None, 0, h * 0.42))
    fig.append(el("C", 'plots/accuracy-2.svg', 0, h))
    savefig(fig, 'fig2')


def fig3():
    w = onecolumn * 2 * 72
    h = 3.9 * 72

    fig = svgfig(w, h * 2)
    fig.append(el("A", 'plots/results-3.svg', 0, 0))
    fig.append(el("B", 'plots/accuracy-3.svg', 0, h))
    savefig(fig, 'fig3')


def fig4():
    w = onecolumn * 2 * 72
    h = 2.9 * 72

    fig = svgfig(w, h * 2)
    fig.append(el("A", 'plots/results-4.svg', 0, 0))
    fig.append(el("B", 'plots/accuracy-4.svg', 0, h))
    savefig(fig, 'fig4')
