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
results_dir = os.path.join(results_dir, 'probes')
fig_dir = os.path.realpath(os.path.join(root, os.pardir, 'figures'))
inkscape = "inkscape"
# Uncomment if on Mac OS X
# inkscape = os.path.expanduser("~/Applications/Inkscape.app/Contents"
#                               "/Resources/bin/inkscape")

# All in inches
# onecolumn = 3.34646  # from initial submission
onecolumn = 4.685044  # after author proof
twocolumn = 7.08661  # never used
horizontal = False


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


def get_data(task, key, separate_cols=True):
    data_dir = results_dir

    if separate_cols:
        data = OrderedDict()
    else:
        data = []
    dirs = [d for d in sorted(os.listdir(data_dir))
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


def plot_summary(task, key, figsize=None):
    setup(figsize=figsize)
    data = get_data(task, key)
    sns.boxplot(data=data)
    rotation = 15 if horizontal else 0
    plt.gca().set_xticklabels(data.columns, rotation=rotation)


def accuracy():
    figsize = (onecolumn, 4.0) if horizontal else (onecolumn * 2, 4.0)
    plot_summary("cchannelchain", "rmse", figsize=figsize)
    plt.ylabel("RMSE")
    save("accuracy-1")

    plot_summary("product", "rmse", figsize=figsize)
    plt.ylabel("RMSE")
    save("accuracy-2")

    plot_summary("controlledoscillator", "score", figsize=figsize)
    plt.ylabel("FFT similarity")
    save("accuracy-3")

    figsize = (onecolumn, 4.0) if horizontal else (onecolumn * 2, 3.0)
    plot_summary("sequence", "timing_mean", figsize=figsize)
    plt.ylabel("Mean transition time (s)")
    plt.ylim(0.04, 0.06)
    save("accuracy-4")

    plot_summary("sequence_pruned", "timing_mean", figsize=figsize)
    plt.ylabel("Mean transition time (s)")
    plt.ylim(0.04, 0.06)
    save("accuracy-4-pruned")


def speed():
    def get_all_data(key, d_args, scale_to_realtime=False):
        t1 = get_data("cchannelchain", key, **d_args)
        t1['Model'] = "Chained channels"
        t2 = get_data("product", key, **d_args)
        t2['Model'] = "Product"
        t3 = get_data("controlledoscillator", key, **d_args)
        t3['Model'] = "Oscillator"
        t4 = get_data("sequence", key, **d_args)
        t4['Model'] = "BG sequence"
        t5 = get_data("sequence_pruned", key, **d_args)
        t5['Model'] = "BG sequence *"
        if scale_to_realtime:
            t1[key] /= 0.5
            t2[key] /= 5.5
            t3[key] /= 10.0
            t4[key] /= 4.0
            t5[key] /= 4.0
        return pd.concat((t1, t2, t3, t4, t5))

    model_order = ["Chained channels",
                   "Product",
                   "Oscillator",
                   "BG sequence",
                   "BG sequence *"]

    # Only get build speed for probed data
    build = get_all_data("buildtime",
                         {'separate_cols': False})

    figsize = (onecolumn * 2, 3.0)
    setup(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    sns.barplot(x='Model', y='buildtime', hue='Backend',
                data=build, ax=ax, order=model_order)
    plt.ylabel("Build time (s)")
    plt.xlabel("")
    save("fig5", fig=True)
    svg2other("fig5")

    run = get_all_data("runtime",
                       {'separate_cols': False},
                       scale_to_realtime=True)

    setup(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    sns.barplot(x='Model', y='runtime', hue='Backend',
                data=run, ax=ax, order=model_order)
    plt.axhline(1.0, lw=1, c='k', ls=':')
    plt.ylim(top=7.0)
    plt.ylabel("Simulation time + overhead / Run time")
    plt.xlabel("")

    inset = zoomed_inset_axes(plt.gca(), 0.38,
                              bbox_to_anchor=(0.6, 0.98),
                              bbox_transform=plt.gcf().transFigure)

    # Have to remove non-BG models before plotting inset
    run = run[run['Model'] != "Chained channels"]
    run = run[run['Model'] != "Product"]
    run = run[run['Model'] != "Oscillator"]
    sns.barplot(x='Model', y='runtime', hue='Backend',
                data=run, ax=inset, order=model_order)
    plt.axhline(1.0, lw=1, c='k', ls=':')
    plt.legend([])
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(())
    plt.xlim(left=2.5)
    pp, _, _ = mark_inset(ax, inset, loc1=3, loc2=4, fc="none", ec="0.5")
    pp.set_visible(False)
    sns.despine(bottom=True, ax=inset)

    fname = "fig6"
    save(fname, fig=True)
    svg2other(fname)


def fig1():
    w = (onecolumn * 72) if horizontal else (onecolumn * 2 * 72)
    h = 3.9 * 72

    if horizontal:
        fig = svgfig(w * 2, h)
        fig.append(el("A", 'plots/results-1.svg', 0, 0))
        fig.append(el("B", 'plots/accuracy-1.svg', w, 0))
    else:
        fig = svgfig(w, h * 2)
        fig.append(el("A", 'plots/results-1.svg', 0, 0))
        fig.append(el("B", 'plots/accuracy-1.svg', 0, h))
    savefig(fig, 'fig1')


def fig2():
    w = (onecolumn * 72) if horizontal else (onecolumn * 2 * 72)
    h = 3.9 * 72

    if horizontal:
        fig = svgfig(w * 2, h)
        fig.append(el("A", 'plots/results-2.svg', 0, 0))
        fig.append(el("B", None, 0, h * 0.42))
        fig.append(el("C", 'plots/accuracy-2.svg', w, 0))
    else:
        fig = svgfig(w, h * 2)
        fig.append(el("A", 'plots/results-2.svg', 0, 0))
        fig.append(el("B", None, 0, h * 0.42))
        fig.append(el("C", 'plots/accuracy-2.svg', 0, h))
    savefig(fig, 'fig2')


def fig3():
    w = (onecolumn * 72) if horizontal else (onecolumn * 2 * 72)
    h = 3.9 * 72

    if horizontal:
        fig = svgfig(w * 2, h)
        fig.append(el("A", 'plots/results-3.svg', 0, 0))
        fig.append(el("B", 'plots/accuracy-3.svg', w, 0))
    else:
        fig = svgfig(w, h * 2)
        fig.append(el("A", 'plots/results-3.svg', 0, 0))
        fig.append(el("B", 'plots/accuracy-3.svg', 0, h))
    savefig(fig, 'fig3')


def fig4():
    w = (onecolumn * 72) if horizontal else (onecolumn * 2 * 72)
    h = 3.9 * 72 if horizontal else 2.9 * 72

    if horizontal:
        fig = svgfig(w * 3, h)
        fig.append(el("A", 'plots/results-4.svg', 0, 0))
        fig.append(el("B", 'plots/accuracy-4.svg', w, 0))
        fig.append(el("C", 'plots/accuracy-4-pruned.svg', w * 2, 0))
    else:
        fig = svgfig(w, h * 3)
        fig.append(el("A", 'plots/results-4.svg', 0, 0))
        fig.append(el("B", 'plots/accuracy-4.svg', 0, h))
        fig.append(el("C", 'plots/accuracy-4-pruned.svg', 0, h * 2))
    savefig(fig, 'fig4')
