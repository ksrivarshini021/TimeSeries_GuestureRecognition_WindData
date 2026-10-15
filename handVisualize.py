import numpy as np
import matplotlib.pyplot as plt

# -------- Helper functions --------
def euler_to_vector(pitch, yaw, roll):
    """
    Convert Unreal-style P/Y/R (degrees) to a direction vector.
    """
    p = np.radians(pitch)
    y = np.radians(yaw)
    r = np.radians(roll)

    # Basic forward vector (Z forward assumption)
    x = np.cos(p) * np.cos(y)
    yv = np.cos(p) * np.sin(y)
    z = np.sin(p)

    return np.array([x, yv, z])


def build_finger(base_pos, rotations, segment_length=1.0):
    """
    Build finger joint positions from rotations
    """
    points = [base_pos]
    current_pos = base_pos.copy()

    for rot in rotations:
        direction = euler_to_vector(*rot)
        current_pos = current_pos + direction * segment_length
        points.append(current_pos)

    return np.array(points)


# -------- Data Definition --------
wrist_origin = np.array([0, 0, 0])

thumb_rotations = [(-3.05, 128.94, 139.44), (2.44, 141.48, 171.68), (1.77, 165.79, 162.98), (9.49, 171.83, 172.86)]
index_rotations = [(8.72, 165.91, 84.26), (44.00, 159.83, 78.27), (58.20, 159.09, 75.74)]
middle_rotations = [(3.91, 171.78, 76.67), (39.75, 160.83, 70.84), (54.37, 148.92, 55.39)]
ring_rotations = [(-2.54, -179.47, 69.12), (37.66, 158.36, 56.87), (53.49, 129.95, 34.47)]
pinky_rotations = [(-13.25, -171.22, 49.21), (-6.86, -167.45, 58.88), (38.80, 155.73, 37.62)]

# Generate point arrays
thumb = build_finger(wrist_origin, thumb_rotations, 0.8)
index = build_finger(wrist_origin, index_rotations, 1.0)
middle = build_finger(wrist_origin, middle_rotations, 1.1)
ring = build_finger(wrist_origin, ring_rotations, 1.0)
pinky = build_finger(wrist_origin, pinky_rotations, 0.9)

# -------- Plotting --------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

def draw_finger(points, label):
    # Plot the skeleton line
    line, = ax.plot(points[:, 0], points[:, 1], points[:, 2], marker='o', markersize=6, label=label)
    color = line.get_color()
    
    # Label each joint point (0 = wrist/base, up to the tip)
    for i, (x, y, z) in enumerate(points):
        # We add a tiny offset to the text so it doesn't overlap the marker
        ax.text(x + 0.02, y + 0.02, z + 0.02, f"{label}_{i}", size=8, color=color, alpha=0.8)

# Execute drawing
draw_finger(thumb, "Thumb")
draw_finger(index, "Index")
draw_finger(middle, "Middle")
draw_finger(ring, "Ring")
draw_finger(pinky, "Pinky")

# Highlight the common Wrist point
ax.scatter(0, 0, 0, color='black', s=100, label='Wrist Center')

# Formatting the axes and labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title("3D Hand Pose Reconstruction", pad=20)

# Move the legend outside the plot to the right
ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), borderaxespad=0.)

# Use tight_layout to make room for the external legend
plt.tight_layout()

# Save and Show
plt.savefig("hand_visualization_annotated.png", dpi=300, bbox_inches='tight')
plt.show()