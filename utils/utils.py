from collections import defaultdict

def merge_points_same_team(points, area_bounds, window_size):
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
        teams = set(p[2] for p in cell_points)
        if len(teams) == 1 and cell_points:
            avg_x = sum(p[0] for p in cell_points) / len(cell_points)
            avg_y = sum(p[1] for p in cell_points) / len(cell_points)
            t = cell_points[0][2]
            color = cell_points[0][3]
            merged_points.append((avg_x, avg_y, t, color))
        else:
            # If there are multiple teams in the same grid cell, ignore the cell
            for cell_point in cell_points:
                merged_points.append(cell_point)

    return merged_points

def merge_points_in_fixed_area(points, area_bounds, window_size):
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
