import matplotlib.colors as mcolors
import numpy as np

flair_cmap = {
    0: "#db0e9a",
    1: "#938e7b",
    2: "#f80c00",
    3: "#a97101",
    4: "#1553ae",
    5: "#194a26",
    6: "#46e483",
    7: "#f3a60d",
    8: "#660082",
    9: "#55ff00",
    10: "#fff30d",
    11: "#e4df7c",
    12: "#000000",
}

msk_to_name = {
    0: "building",
    1: "pervious surface",
    2: "impervious surface",
    3: "bare soil",
    4: "water",
    5: "coniferous",
    6: "deciduous",
    7: "brushwood",
    8: "vineyard",
    9: "herbaceous vegetation",
    10: "agricultural land",
    11: "plowed land",
    12: "other",
}

train_data_percentages = (
    8.14,
    8.25,
    13.72,
    3.47,
    4.88,
    2.74,
    15.38,
    6.95,
    3.13,
    17.84,
    10.98,
    3.88,
    0.65,
)

test_data_percentages = (
    3.26,
    3.82,
    5.87,
    1.60,
    3.17,
    10.24,
    24.79,
    3.81,
    2.55,
    19.76,
    18.19,
    1.81,
    1.14,
)


def get_custom_colormap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """
    Creates and returns a custom matplotlib colormap and normalization based on predefined flair
    color mappings.

    Returns:
        tuple: A tuple containing the custom ListedColormap and its corresponding BoundaryNorm.
    """
    colors_list = [flair_cmap[key] for key in flair_cmap]
    cmap = mcolors.ListedColormap(colors_list, name="custom_lut_map")
    norm = mcolors.BoundaryNorm(np.arange(len(colors_list) + 1) - 0.5, len(colors_list))
    return cmap, norm
