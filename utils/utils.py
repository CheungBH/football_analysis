from collections import defaultdict


def merge_points_in_fixed_area(points, area_bounds, window_size, keep_same_team=False):
    """
    Merges points within a non-overlapping sliding window inside a given area.

    :param points: List of tuples [(x1, y1), (x2, y2), ...] representing the points in the plane.
    :param area_bounds: Tuple (min_x, min_y, max_x, max_y) defining the bounding box of the area.
    :param window_size: The fixed size of the sliding window.
    :return: A list of merged points [(x', y'), ...] within the given area.
    """
    min_x, min_y, max_x, max_y = area_bounds
    grid = defaultdict(list)

    # Categorize points into grid cells within the specified area
    for x, y, team, color in points:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            grid_x = (x - min_x) // window_size  # Shift to start from min_x
            grid_y = (y - min_y) // window_size  # Shift to start from min_y
            grid[(grid_x, grid_y)].append((x, y, team, color))

    # Compute centroids for each grid cell
    merged_points = []
    for (grid_x, grid_y), cell_points in grid.items():
        if cell_points:
            avg_x = sum(p[0] for p in cell_points) / len(cell_points)
            avg_y = sum(p[1] for p in cell_points) / len(cell_points)
            t = cell_points[0][2]
            color = cell_points[0][3]
            merged_points.append((avg_x, avg_y, t, color))

    return merged_points