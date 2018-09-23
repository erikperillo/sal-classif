"""
Visualization utility functions to be frequently used during data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

DEF_SCATTER_ARGS = {
    "color": "b",
    "marker": "o",
    "linewidths": 1,
    "label": "pontos"
}

DEF_LINE_ARGS = {
    "color": "b",
    "linewidths": 1,
    "label": "linha"
}

DEF_ERRORBAR_ARGS = {
    "color": "g",
    "linestyle": "None",
    "label": "erros"
}

DEF_REGLINE_ARGS = {
    "color": "r",
    "label": "regressão"
}

DEF_LEGEND_ARGS = {
    "loc": 2
}

def plot(x, y, style="scatter", plot_args={},
        title="", x_label="", y_label="",
        x_err=None, y_err=None, err_plot_args={},
        x_reg=None, y_reg=None, reg_plot_args={},
        legend=True, legend_args={},
        x_limits=None, y_limits=None,
        colorbar=False,
        new_fig=True, show=True):
    """pyplot plot wrapper"""

    if new_fig:
        plt.figure()

    #main plot
    if style == "scatter":
        _plot_args = dict(DEF_SCATTER_ARGS)
        _plot_args.update(plot_args)
        plt.scatter(x=x, y=y, **_plot_args)
    elif style == "line":
        _plot_args = dict(DEF_LINE_ARGS)
        _plot_args.update(plot_args)
        plt.plot(x=x, y=y, **_plot_args)
    else:
        raise ValueError("unknown plot style '%s'" % style)

    #error bars
    if x_err is not None or y_err is not None:
        _err_plot_args = dict(DEF_ERRORBAR_ARGS)
        _err_plot_args.update(err_plot_args)
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, **_err_plot_args)

    #regression line
    if x_reg is not None and y_reg is not None:
        _reg_plot_args = dict(DEF_REGLINE_ARGS)
        _reg_plot_args.update(reg_plot_args)
        plt.plot(x_reg, y_reg, **_reg_plot_args)

    #labels, title and legends
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legend:
        _legend_args = dict(DEF_LEGEND_ARGS)
        _legend_args.update(legend_args)
        plt.legend(**_legend_args)

    #setting plot range
    if x_limits is not None:
        plt.xlim(xmin=x_limits[0], xmax=x_limits[1])
    if y_limits is not None:
        plt.ylim(ymin=y_limits[0], ymax=y_limits[1])

    if colorbar:
        plt.colorbar()

    #displaying if required
    if show:
        plt.show()

def show_imgs(imgs, n_cols=None, show=True, title=None, cmap="RdYlGn"):
    """
    Displays a sequence of images.
    imgs is a list that can contain images (np.array) and dicts in format:
        img: np.array image
        title: title of image
        cmap: color map of image
    """
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]

    #converting images to dicts
    imgs_dcts = []
    for img in imgs:
        if not isinstance(img, dict):
            img = {"img": img}
        imgs_dcts.append(img)

    #number of rows/columns
    if n_cols is None:
        n_cols = np.ceil(np.sqrt(len(imgs_dcts)))
    n_rows = np.ceil(len(imgs_dcts)/n_cols)

    plt.figure()

    #setting title
    if title is not None:
        plt.suptitle(title)

    #plotting figures
    for i, dct in enumerate(imgs_dcts):
        ax = plt.subplot(n_rows, n_cols, i+1)
        #setting of scale/axis
        plt.axis("off")
        #setting title if required
        if "title" in dct:
            ax.set_title(dct["title"])
        #displaying image
        plt.imshow(dct["img"], cmap=dct.get("cmap", cmap))

    #this sets a better-looking layout
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    #show images
    if show:
        plt.show()

