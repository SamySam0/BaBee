import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Plot euclidean distance to hover point over time
def plot_distance_to_hover(save_path, title, errors, std, reference=None, show_plot=False):
    # Generate time (X) values
    x_values = list(x/120 for x in range(1, len(errors) + 1)) # 120Hz

    # Set the seaborn style background
    sns.set(style="darkgrid")

    # Plotting mean distances
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, errors, marker=None, linestyle='-', color='b', label="Mean distance")

    # Adding standard deviation as shaded area
    plt.fill_between(x_values, [e - s for e, s in zip(errors, std)], [e + s for e, s in zip(errors, std)], 
                     color='b', alpha=0.2, label="Standard deviation")

    # Add reference (nominal only, i.e. expected outcome) if available
    if reference is not None:
        (errors_ref, std_ref) = reference
        plt.plot(x_values, errors_ref, marker=None, linestyle='dashed', color='darkcyan', label="Expected distance (nominal only)")

    plt.title(title)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Euclidean distance')
    plt.ylim(ymin=0)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()


# Plot position of the drone in 3D space over time
def plot_position_in_3d_space(xs, ys, zs, save_path, hover_point, graph_type, reference=None, show_plot=False):
    goal_x, goal_y, goal_z = hover_point

    # Creating the figure and 3D axis
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Set plans color
    ax.xaxis.pane.set_facecolor('#dadeea')
    ax.yaxis.pane.set_facecolor('#dadeea')
    ax.zaxis.pane.set_facecolor('#dadeea')

    # Deicide on graph type
    if graph_type == 'ZOOM':
        ax.set_xlim([goal_x-0.25, goal_x+0.25])
        ax.set_ylim([goal_y-0.25, goal_y+0.25])
        ax.set_zlim([goal_z-0.25, goal_z+0.25])

        # Target dot size (proportional to space boundaries size)
        target_dot_size = 115

        # Get intersection of goal with XYZ plans
        inter_x, inter_y, inter_z = goal_x-0.25, goal_y+0.25, goal_z-0.25

        # Cut down out of bound points
        i_xs = [i for i in range(len(xs)) if not goal_x-0.25 <= xs[i] <= goal_x+0.25]
        i_ys = [i for i in range(len(ys)) if not goal_y-0.25 <= ys[i] <= goal_y+0.25]
        i_zs = [i for i in range(len(zs)) if not goal_z-0.25 <= zs[i] <= goal_z+0.25]
        i = i_xs + i_ys + i_zs
        xs = [xs[e] for e in range(len(xs)) if e not in i]
        ys = [ys[e] for e in range(len(ys)) if e not in i]
        zs = [zs[e] for e in range(len(zs)) if e not in i]

        # Set the view orientation
        ax.view_init(elev=18, azim=-68)

    elif graph_type == 'FIT_PATH':
        # Target dot size (proportional to space boundaries size)
        target_dot_size = 70

        # Get intersection of goal with XYZ plans
        inter_x, inter_y, inter_z = min(xs), max(ys), min(zs)

        # Set the view orientation
        ax.view_init(elev=20, azim=-45)

    elif graph_type == 'FULL_WORLD':
        # Fit to world size/limit from simulation
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])

        # Target dot size (proportional to space boundaries size)
        target_dot_size = 30

        # Get intersection of goal with XYZ plans
        inter_x, inter_y, inter_z = -1, 1, 0

        # Set the view orientation
        ax.view_init(elev=20, azim=-45)

    else:
        assert False, 'Graph type is invalid.'


    # Adding lines perpendicular to the X, Y, and Z planes from the goal location
    ax.plot([goal_x, goal_x], [goal_y, goal_y], [inter_z, goal_z], color='lightgrey', linestyle='--')
    ax.plot([goal_x, goal_x], [inter_y, goal_y], [goal_z, goal_z], color='lightgrey', linestyle='--')
    ax.plot([inter_x, goal_x], [goal_y, goal_y], [goal_z, goal_z], color='lightgrey', linestyle='--')

    # Plotting the path as a continuous line
    ax.plot(xs, ys, zs, color='r', marker=None, linestyle='-', linewidth=1, label='Drone trajectory')

    # Add reference (nominal only, i.e. expected outcome) if available
    if (reference is not None) and (graph_type != 'ZOOM'):
        (xs_ref, ys_ref, zs_ref) = reference
        ax.plot(xs_ref, ys_ref, zs_ref, color='darkcyan', marker=None, linestyle='dashed', linewidth=1, label='Expected trajectory (nominal only)')

    # Add target dot
    ax.scatter(goal_x, goal_y, goal_z, color='g', alpha=0.5, s=target_dot_size, label='Goal location')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Adding labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Title and Legend
    ax.set_title('Drone Trajectory in 3D Space')
    ax.legend()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    if show_plot:
        plt.show()

