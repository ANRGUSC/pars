
import numpy as np

def gen_neighborhoods_uniform_distribution(n_users=250, grid_size=10):

    # Mean and standard deviation for the normal distribution
    mean = [grid_size / 2, grid_size / 2]
    std_dev = grid_size / 3  # This ensures that most points fall within the grid

    # Ensure there's at least one point in each cell
    if n_users != grid_size * grid_size:
        raise ValueError("Number of users must be at least as large as the number of cells in the grid.")

    # Start by placing one user in each cell
    cell_coords = np.array([(i // grid_size, i % grid_size) for i in range(grid_size**2)])

    # Generate the remaining random 2D points based on the normal distribution
    remaining_users = n_users - grid_size**2
    if remaining_users > 0:
        additional_points = np.random.normal(mean, std_dev, (remaining_users, 2))
        additional_coords = np.clip(np.round(additional_points).astype(int), 0, grid_size - 1)
        cell_coords = np.vstack((cell_coords, additional_coords))

    # Create an empty grid to store user counts
    user_map = np.zeros((grid_size, grid_size), dtype=int)

    # Count the number of users in each cell
    for coord in cell_coords:
        user_map[coord[0], coord[1]] += 1

    return cell_coords, user_map

def gen_neighborhoods_normal_distribution(pnrg, n_users=250, grid_size=10):
    mean = grid_size / 2
    std_dev = grid_size / 3
    unique_coords = set()

    while len(unique_coords) < n_users:
        x_coord = pnrg.normal(mean, std_dev, 1)[0]  
        y_coord = pnrg.normal(mean, std_dev, 1)[0]  

        x_coord = int(np.clip(x_coord, 0, grid_size - 1))
        y_coord = int(np.clip(y_coord, 0, grid_size - 1))

        unique_coords.add((x_coord, y_coord))

    cell_coords = np.array(list(unique_coords))

    # Create an empty grid to store user counts
    user_map = np.zeros((grid_size, grid_size), dtype=int)

    # Count the number of users in each cell
    for coord in cell_coords:
        user_map[coord[0], coord[1]] += 1

    return cell_coords, user_map

