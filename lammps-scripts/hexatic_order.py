#!/usr/bin/env python
# coding: utf-8

"""
Module for calculating and visualizing hexatic order parameters and Voronoi diagrams
from particle position data.
"""

import ast
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import NearestNeighbors


def get_points(txt_file):
    """Load particle positions from a text file."""
    times = np.loadtxt(txt_file, skiprows=1)
    return times[:, 1:3]


def draw_voronoi(points):
    """Draw and display a Voronoi diagram colored by the number of polygon sides."""
    vor = Voronoi(points)

    fig = voronoi_plot_2d(
        vor, point_size=1.5, line_width=0.5, line_colors="black", show_vertices=False
    )
    # fig.set_size_inches(5,5)
    fig.set_dpi(300)
    for region in vor.regions:
        if region and -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            if len(region) == 4:
                plt.fill(*zip(*polygon), facecolor="#4B0082")
            elif len(region) == 5:
                plt.fill(*zip(*polygon), facecolor="#00FFFF")
            elif len(region) == 6:
                plt.fill(*zip(*polygon), facecolor="#0000FF")
            elif len(region) == 7:
                plt.fill(*zip(*polygon), facecolor="#C1F80A")
            elif len(region) == 8:
                plt.fill(*zip(*polygon), facecolor="#ED0DD9")
            elif len(region) == 9:
                plt.fill(*zip(*polygon), facecolor="#808080")
            else:
                plt.fill(*zip(*polygon), facecolor="#15B01A")

    plt.title("Voronoi Diagram")

    colors = ["#4B0082", "#00FFFF", "#0000FF", "#C1F80A", "#ED0DD9", "#808080", "#15B01A"]
    labels = [
        "4-sided polygon",
        "5-sided polygon",
        "6-sided polygon",
        "7-sided polygon",
        "8-sided polygon",
        "9-sided polygon",
        "10+ sided polygon",
    ]

    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()


def hexatic_order(points):
    """Calculate the hexatic order parameter for each particle."""
    neighbors_model = NearestNeighbors(n_neighbors=7, algorithm="ball_tree").fit(points)
    _, indices = neighbors_model.kneighbors(points)

    angle_array = []
    for i in range(len(indices)):
        angle_array_row = []
        for j in range(7):
            index_neighbor = indices[i][j]
            neighbor_x = points[index_neighbor, 0]
            neighbor_y = points[index_neighbor, 1]
            original_x = points[i, 0]
            original_y = points[i, 1]

            delta_x = neighbor_x - original_x
            delta_y = neighbor_y - original_y
            if delta_x != 0:
                angle = math.atan(delta_y / delta_x)
                angle_array_row.append(angle)
        angle_array.append(angle_array_row)

    hexatic_order_params = []
    for i in range(len(angle_array)):
        hex_sum = 0
        for j in range(len(angle_array[i])):
            hex_sum += np.exp(complex(0, 6 * angle_array[i][j]))

        hexatic_order_params.append(abs(hex_sum) / 6)
    return hexatic_order_params


def draw_histogram(points):
    """Draw a histogram of hexatic order parameters."""
    params = hexatic_order(points)
    plt.hist(params, bins=10, color="magenta", alpha=0.9)

    plt.title("Hexatic Order Parameters")
    plt.xlabel("Hexatic Order Parameter")
    plt.ylabel("Frequency")
    plt.show()


def print_hex_info(points):
    """Print statistical information about hexatic order parameters."""
    params = hexatic_order(points)
    hist, edges = np.histogram(params, bins=np.linspace(0.0, 1.0, 11))

    for i in range(len(hist)):
        bin_label = "Hexatic Order: {:.1f} - {:.1f}; Quantity: ".format(edges[i], edges[i + 1])
        print(bin_label + str(hist[i]))

    try:
        if os.path.exists("hexatic_order.txt"):
            with open("hexatic_order.txt", "r", encoding="utf-8") as order_file:
                saved_params = ast.literal_eval(order_file.read())
            mean = np.mean(saved_params)
            print(f"Mean of saved hexatic order: {mean}")
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing hexatic_order.txt: {e}")


def get_files(directory):
    """Get sorted list of files from a directory."""
    files = os.listdir(directory)

    def sort_alg(filename):
        try:
            if "_" in filename:
                return int(filename[:-6])
            return int(filename[:-4])
        except (ValueError, IndexError):
            return 0

    files.sort(key=sort_alg)
    return files


def plot_list(dir_list):
    """Plot the mean hexatic order over time for a list of directories."""
    markers = ["o", "*", ".", "x", "X", "+", "P", "s", "D", "d", "p", "H", "h", "v", "^", "<", ">"]
    mark = 0
    for directory in dir_list:
        if not os.path.isdir(directory):
            continue
        x_axis = []
        y_axis = []
        files = get_files(directory)
        for filename in files:
            if filename.endswith(".txt"):
                try:
                    points = get_points(os.path.join(directory, filename))
                    params = hexatic_order(points)
                    y_val = np.mean(params)
                    y_axis.append(y_val)
                    if "_" in filename:
                        x_val = int(filename[:-6])
                    else:
                        x_val = int(filename[:-4])
                    x_axis.append(x_val)
                except (ValueError, IndexError, np.linalg.LinAlgError):
                    continue
        plt.plot(x_axis, y_axis, marker=markers[mark % len(markers)], label=directory)
        mark += 1
    plt.xlabel("Time (seconds)")
    plt.ylabel("mean hexatic order")
    plt.title("Hexatic order over time")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()


def plot_single(directory):
    """Plot the mean hexatic order for a single directory."""
    plot_list([directory])


def plot_all():
    """Plot the mean hexatic order for all directories in the current folder."""
    plot_list([d for d in os.listdir() if os.path.isdir(d)])


def main():
    """Main execution block."""
    plot_all()

    # Process a specific test file
    test_file = "100%/600.txt"
    if os.path.exists(test_file):
        points = get_points(test_file)
        draw_voronoi(points)
        draw_histogram(points)

    # Process all directories and their files
    for entry in os.listdir():
        if os.path.isdir(entry):
            plot_single(entry)
            files = os.listdir(entry)
            for filename in files:
                filepath = os.path.join(entry, filename)
                print(filepath)
                if filename.endswith(".txt"):
                    try:
                        points = get_points(filepath)
                        draw_voronoi(points)
                    except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                        print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    main()
