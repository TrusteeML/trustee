import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib import rcParams

FONT_NAME = "Roboto"
FONT_WEIGHT = "light"

rcParams["font.family"] = "serif"
rcParams["font.serif"] = [FONT_NAME]
rcParams["font.weight"] = FONT_WEIGHT


def plot_heatmap(matrix, labels=[], path=None):
    """Util function to plot confusion matrix"""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#d75d5b", "#a7c3cd"])
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.matshow(matrix, cmap=cmap, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=f"{matrix[i, j]:.2f}",
                va="center",
                ha="center",
                # size="xx-large",
            )

    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_lines(x, y, xlim=None, ylim=None, labels=[], title=None, xlabel=None, ylabel=None, size=(), path=None):
    """Util function to plot lines"""
    plt.figure(figsize=size if size else (4, 1.5))  # width:20, height:3
    # plt.figure(figsize=size if size else (3, 2))  # width:20, height:3
    markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        "8",
        "s",
        "p",
        "P",
        "*",
        "h",
        "H",
        "+",
        "x",
        "X",
        "D",
        "d",
        "|",
        "_",
    ]
    colors = [
        "#a7c3cd",
        "#8a4444",
        "#524a47",
        "#d75d5b",
        "#c8c5c3",
        "#f5f0ed",
        "#edeef0",
    ]

    if np.shape(x)[0] == 1:
        x = np.ravel(x)

    if np.shape(x) and np.shape(x)[0] > 1 and not isinstance(x[0], str) and np.shape(x[0]) and np.shape(x[0])[0] >= 1:
        for idx, (x_values, y_values) in enumerate(zip(x, y)):
            plt.plot(
                x_values,
                y_values,
                color=colors[idx] if idx < len(colors) else None,
                # marker=markers[idx] if idx < len(markers) else None,
                label=labels[idx] if idx < len(labels) else "",
            )
    else:
        for idx, values in enumerate(y):
            plt.plot(
                x,
                values,
                color=colors[idx] if idx < len(colors) else None,
                # marker=markers[idx] if idx < len(markers) else None,
                label=labels[idx] if idx < len(labels) else "",
            )

    if len(labels) > 1:
        plt.legend(loc="lower right", ncol=2)

    if xlabel:
        plt.xlabel(xlabel, fontname=FONT_NAME, fontweight=FONT_WEIGHT)

    if ylabel:
        plt.ylabel(ylabel, fontname=FONT_NAME, fontweight=FONT_WEIGHT)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)

    _, end = plt.xlim()
    end = int(end)
    plt.xticks(np.arange(0, end + 1, max(1, int(end / 10))))

    plt.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_bars(x, y, ylim=None, xlabel=None, ylabel=None, labels=[], title=None, path=None):
    """Util function to plot bars"""
    plt.figure(figsize=(30, 3))  # width:20, height:3
    width = 0.4
    fig, ax = plt.subplots()
    locs = np.arange(len(x))  # the label locations
    colors = [
        "#d75d5b",
        "#a7c3cd",
        "#524a47",
        "#8a4444",
        "#c8c5c3",
        "#524a47",
        "#edeef0",
    ]

    for idx, values in enumerate(y):
        ax.bar(
            locs - (width / 2) if idx % 2 == 0 else locs + (width / 2),
            values,
            width,
            color=colors[idx] if idx <= len(colors) else None,
            label=labels[idx] if idx < len(labels) else "",
        )

    ax.set_xticks(locs)
    ax.set_xticklabels(x, rotation=60)
    if labels:
        ax.legend()

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if ylim:
        ax.set_ylim(ylim)

    if title:
        plt.title(title)

    fig.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_lines_and_bars(
    x,
    lines,
    bars,
    ylim=None,
    xlabel=None,
    ylabel=None,
    second_x_axis=None,
    second_x_axis_label=None,
    labels=[],
    legend=[],
    colors_by_x=[],
    title=None,
    path=None,
):
    """Util function to plot lines"""
    plt.figure(figsize=(40, 3))  # width:20, height:3

    width = 0.4
    fig, ax = plt.subplots()
    locs = np.arange(len(x))  # the label locations
    colors = [
        "#d75d5b",
        "#a7c3cd",
        "#f5f0ed",
        "#524a47",
        "#8a4444",
        "#edeef0",
        "#c8c5c3",
    ]

    for idx, values in enumerate(lines):
        ax.plot(
            x,
            values,
            color=colors[idx] if idx < len(colors) else None,
            label=labels[idx] if idx < len(labels) else "",
        )

    for idx, values in enumerate(bars):
        if colors_by_x:
            ax.bar(
                locs,
                values,
                width if len(bars) > 1 else 1,
                color=colors_by_x,
            )
        else:
            ax.bar(
                locs - (width / 2) if idx % 2 == 0 else locs + (width / 2),
                values,
                width if len(bars) > 1 else 1,
                color=colors[len(colors) - idx - 1] if len(colors) - idx - 1 >= 0 else None,
                label=labels[idx] if idx < len(labels) else "",
            )

    patches = []
    if legend:
        for label, color in legend.items():
            patches.append(mpatches.Patch(color=color, label=label))

    ax.set_xticks(locs)
    ax.set_xticklabels(x, rotation=60)

    if second_x_axis is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(locs)
        ax2.set_xticklabels(second_x_axis, rotation=60)

        if second_x_axis_label:
            ax2.set_xlabel(second_x_axis_label)

    if patches:
        plt.legend(handles=patches)
    elif labels:
        plt.legend()

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)

    plt.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_stacked_bars(x, y, y_placeholder=None, ylim=None, xlabel=None, ylabel=None, labels=[], title=None, path=None):
    plt.figure(figsize=(50, 10))  # width:20, height:3
    """Util function to plot stacker bars"""
    fig, ax = plt.subplots()
    width = 0.8
    colors = [
        "#a7c3cd",
        "#8a4444",
        "#c8c5c3",
        "#f5f0ed",
        "#d75d5b",
    ]
    # hatches = ["/", "-", "+", ".", "*"]

    y_placeholder = np.sort(y_placeholder, axis=0)[::-1] if y_placeholder else None
    labels = [label for _, label in (sorted(zip(y, labels), key=lambda pair: np.sum(pair[0]))[::-1])]
    y = np.sort(y, axis=0)[::-1]

    if y_placeholder is not None:
        previous_stack = 0
        for i, stack in enumerate(y_placeholder):
            if i > 0:
                previous_stack += y_placeholder[i - 1]

            rects = ax.bar(
                x,
                stack,
                width,
                color="#edeef0",
                edgecolor="#524a47",
                linewidth=0.25,
                bottom=previous_stack,
            )
        sum_y = [sum(val) for val in zip(*y)]
        # ax.bar_label(rects, labels=[f"{val:.2f}" for val in sum_y], padding=1)

    bottom_by_y = {}
    if y_placeholder is not None:
        for i, stack in enumerate(y_placeholder):
            if i == 0:
                bottom_by_y[i] = stack
            else:
                bottom_by_y[i] = stack + bottom_by_y[i - 1]
    else:
        for i, stack in enumerate(y):
            if i == 0:
                bottom_by_y[i] = stack
            else:
                bottom_by_y[i] = stack + bottom_by_y[i - 1]

    for i, values in enumerate(y):
        rects = ax.bar(
            x,
            values,
            width,
            bottom=bottom_by_y[i - 1] if i > 0 and bottom_by_y else 0,
            # hatch=hatches[i] if i < len(hatches) else None,
            color=colors[i] if i < len(colors) else None,
            label=labels[i] if labels else "",
        )
        # ax.bar_label(rects, label_type="center", fmt="%.2f", padding=5)

    if labels:
        ax.legend()

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if ylim:
        ax.set_ylim(ylim)

    plt.xticks(rotation=60)
    if title:
        plt.title(title)

    fig.tight_layout()
    plt.tight_layout()

    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_stacked_bars_split(
    x, y_a, y_b, y_placeholder=None, ylim=None, xlabel=None, ylabel=None, labels=[], title=None, path=None
):
    """Util function to plot stacker bars"""
    plt.figure(figsize=(50, 3))  # width:50, height:3
    fig, ax = plt.subplots()
    width = 0.8
    colors = [
        "#a7c3cd",
        "#8a4444",
        "#c8c5c3",
        "#f5f0ed",
        "#d75d5b",
    ]
    # hatches = ["/", "-", "+", ".", "*"]

    labels = [label for _, label in sorted(zip(y_a, labels), key=lambda pair: np.sum(pair[0]))[::-1]]
    y_a = np.sort(y_a, axis=0)[::-1]
    y_b = np.sort(y_b, axis=0)[::-1]
    x = np.sort(x, axis=0)[::-1]
    y_placeholder = np.sort(y_placeholder, axis=0)[::-1] if y_placeholder else None

    locs = np.arange(len(x))  # the label locations
    new_locs = np.array([2 * i for i in locs])
    if y_placeholder is not None:
        previous_stack = 0
        for i, stack in enumerate(y_placeholder):
            if i > 0:
                previous_stack += y_placeholder[i - 1]

            rects1 = ax.bar(
                new_locs - (width / 2),
                stack,
                width,
                color="#edeef0",
                edgecolor="#524a47",
                linewidth=0.25,
                bottom=previous_stack,
            )
            rects2 = ax.bar(
                new_locs + (width / 2),
                stack,
                width,
                color="#edeef0",
                edgecolor="#524a47",
                linewidth=0.25,
                bottom=previous_stack,
            )
        sum_y_a = [sum(val) for val in zip(*y_a)]
        sum_y_b = [sum(val) for val in zip(*y_b)]
        # ax.bar_label(rects1, labels=[f"{val:.2f}" for val in sum_y_a], padding=1, rotation=60)
        # ax.bar_label(rects2, labels=[f"{val:.2f}" for val in sum_y_b], padding=1, rotation=60)

    bottom_by_y = {}
    bottom_by_y_a = {}
    bottom_by_y_b = {}
    if y_placeholder is not None:
        for i, stack in enumerate(y_placeholder):
            if i == 0:
                bottom_by_y[i] = stack
            else:
                bottom_by_y[i] = stack + bottom_by_y[i - 1]
    else:
        for i, (values_a, values_b) in enumerate(zip(y_a, y_b)):
            if i == 0:
                bottom_by_y_a[i] = np.array(values_a)
                bottom_by_y_b[i] = np.array(values_b)
            else:
                bottom_by_y_a[i] = np.array(values_a) + bottom_by_y_a[i - 1]
                bottom_by_y_b[i] = np.array(values_b) + bottom_by_y_b[i - 1]

    for i, (values_a, values_b) in enumerate(zip(y_a, y_b)):
        rects1 = ax.bar(
            new_locs - (width / 2),
            values_a,
            width,
            bottom=bottom_by_y[i - 1]
            if i > 0 and bottom_by_y
            else bottom_by_y_a[i - 1]
            if i > 0 and bottom_by_y_a
            else 0,
            # hatch=hatches[i] if i < len(hatches) else None,
            color=colors[i] if i < len(colors) else None,
            label=labels[i] if labels and i < len(labels) else None,
        )
        rects2 = ax.bar(
            new_locs + (width / 2),
            values_b,
            width,
            bottom=bottom_by_y[i - 1]
            if i > 0 and bottom_by_y
            else bottom_by_y_b[i - 1]
            if i > 0 and bottom_by_y_b
            else 0,
            # hatch=hatches[i] if i < len(hatches) else None,
            color=colors[i] if i < len(colors) else None,
            # label=labels[i] if labels else "",
        )
        # ax.bar_label(rects1, label_type="center", fmt="%.2f", padding=5)
        # ax.bar_label(rects2, label_type="center", fmt="%.2f", padding=5)

    ax.set_xticks(new_locs)
    ax.set_xticklabels(x, rotation=60)
    if labels:
        ax.legend()

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if ylim:
        ax.set_ylim(ylim)

    if title:
        plt.title(title)

    fig.tight_layout()
    plt.tight_layout()

    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
