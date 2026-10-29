"""Adjust publication fonts after layout, preserving axes positions and ticks."""

from matplotlib.text import Text
from matplotlib.ticker import FixedLocator


def paper_fonts(fig, size, *, tick_size=None, label_size=None, legend_size=None):
    """Sizes are in drawing points, accounting for the intended paper width.

    Call after tight_layout/subplots_adjust and before saving. Resolve automatic
    ticks at the original font size so larger text cannot change their spacing.
    This function never resizes the figure or moves its axes or legends.
    """
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
            axis.set_minor_locator(FixedLocator(axis.get_minorticklocs()))
    for text in fig.findobj(Text):
        text.set_fontsize(size)
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=tick_size or size)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(label_size or size)
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set_fontsize(tick_size or size)
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes]
    for legend in filter(None, legends):
        for text in [*legend.get_texts(), legend.get_title()]:
            text.set_fontsize(legend_size or size)
