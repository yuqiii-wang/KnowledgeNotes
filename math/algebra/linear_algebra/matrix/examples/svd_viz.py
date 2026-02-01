import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# 1) Define two vectors
x\_i = np.array([3, 3])
x_j = np.array([3, 1])

# 2) Define matrix A, compute its SVD, get V, and build rotation matrix R
A = np.array([[-1, 1], [0, 1]])
U, S, Vt = np.linalg.svd(A)
S = np.diag(S)

# 2.2) Compute for reflection/rotation for V and U
def compute_vec_change_ang(mat):
    mat_det = np.linalg.det(mat)
    print(f"det({mat})={mat_det}")
    if mat_det == 1.0:
        change_ang = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        eigvals, eigvecs = np.linalg.eig(mat)
        axis_idx = np.where(np.isclose(eigvals, 1))[0][0]
        n = eigvecs[:, axis_idx]
        n = n / np.linalg.norm(n)  # Ensure unit vector
        I = np.eye(2)
        change_ang = I - 2 * np.outer(n, n)
    return change_ang
change_ang_v = compute_vec_change_ang(Vt)
change_ang_u = compute_vec_change_ang(U)

# 3) Transform by SVD
x_i_v = Vt @ x\_i
x_j_v = Vt @ x_j
x_i_s = S @ Vt @ x\_i
x_j_s = S @ Vt @ x_j
x_i_u = U @ S @ Vt @ x\_i
x_j_u = U @ S @ Vt @ x_j


# Find max/min to plot
x_min, y_min = -3, -1
x_max, y_max = 4, 5

# Add some padding to the limits
padding = 0.5
x_min -= padding
x_max += padding
y_min -= padding
y_max += padding

# 4) Compute absolute angles for drawing arcs
ang_i = np.degrees(np.arctan2(x\_i[1], x\_i[0]))
ang_j = np.degrees(np.arctan2(x_j[1], x_j[0]))
ang_i_v = np.degrees(np.arctan2(x_i_v[1], x_i_v[0]))
ang_j_v = np.degrees(np.arctan2(x_j_v[1], x_j_v[0]))
ang_i_s = np.degrees(np.arctan2(x_i_s[1], x_i_s[0]))
ang_j_s = np.degrees(np.arctan2(x_j_s[1], x_j_s[0]))
ang_i_u = np.degrees(np.arctan2(x_i_u[1], x_i_u[0]))
ang_j_u = np.degrees(np.arctan2(x_j_u[1], x_j_u[0]))

# 5) Compute preserved angle θ' (should equal θ)
theta_original = np.arccos(
    np.dot(x\_i, x_j) /
    (np.linalg.norm(x\_i) * np.linalg.norm(x_j))
)
theta_v = np.arccos(
    np.dot(x_i_v, x_j_v) /
    (np.linalg.norm(x_i_v) * np.linalg.norm(x_j_v))
)
theta_s = np.arccos(
    np.dot(x_i_s, x_j_s) /
    (np.linalg.norm(x_i_s) * np.linalg.norm(x_j_s))
)
theta_u = np.arccos(
    np.dot(x_i_u, x_j_u) /
    (np.linalg.norm(x_i_u) * np.linalg.norm(x_j_u))
)

# 6) Plot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))

def plot_ax(ax, x\_i, x_j, x_i_new, x_j_new,
            ang_i, ang_j, ang_i_new, ang_j_new,
            change_ang,
            flag='v'):

    # Original vectors (dashed gray)
    ax.arrow(0, 0, *x\_i, head_width=0.1, linestyle='--', color='gray', alpha=0.7)
    ax.arrow(0, 0, *x_j, head_width=0.1, linestyle='--', color='gray', alpha=0.7)

    # Rotated vectors x' (solid dark gray)
    ax.arrow(0, 0, *x_i_new, head_width=0.1, color='dimgray', alpha=1.0)
    ax.arrow(0, 0, *x_j_new, head_width=0.1, color='dimgray', alpha=1.0)

    # Labels
    ax.text(x\_i[0]+0.1, x\_i[1]+0.1, '$x\_i$', fontsize=12)
    ax.text(x_j[0]+0.1, x_j[1]+0.1, '$x_j$', fontsize=12)
    ax.text(x_i_new[0]+0.1, x_i_new[1]+0.1, "$x\_i$'", fontsize=12)
    ax.text(x_j_new[0]+0.1, x_j_new[1]+0.1, "$x_j$'", fontsize=12)

    if flag == 'v' or flag == 'u':
        if flag == 'v':
            ax.set_title(f'Reflection by {flag}')
            theta_p = theta_v
            theta = theta_original
        else:
            ax.set_title(f'Rotation by {flag}')
            theta_p = theta_u
            theta = theta_s

        if ang_i > ang_j:
            ang_i, ang_j = ang_j, ang_i
        if ang_i_new > ang_j_new:
            ang_i_new, ang_j_new = ang_j_new, ang_i_new

        # Arcs for θ and θ' (blue)
        r_theta = 0.6
        arc_theta = Arc((0, 0), 2*r_theta, 2*r_theta,
                        angle=0, theta1=ang_i, theta2=ang_j,
                        color='blue', lw=2)
        ax.add_patch(arc_theta)

        r_theta2 = 0.9
        arc_theta2 = Arc((0, 0), 2*r_theta2, 2*r_theta2,
                        angle=0, theta1=ang_i_new, theta2=ang_j_new,
                        color='blue', lw=2)
        ax.add_patch(arc_theta2)

        if not isinstance(change_ang, np.ndarray):
            change_ang_i = change_ang
            change_ang_j = change_ang
            r_phi = 2.2
            start = (r_phi * np.cos(np.radians(ang_i)),
                    r_phi * np.sin(np.radians(ang_i)))
            end = (r_phi * np.cos(np.radians(ang_i + np.degrees(change_ang_i))),
                r_phi * np.sin(np.radians(ang_i + np.degrees(change_ang_i))))
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                connectionstyle="arc3,rad=0.3")
            )
            r_phi = 1.8
            start = (r_phi * np.cos(np.radians(ang_j)),
                    r_phi * np.sin(np.radians(ang_j)))
            end = (r_phi * np.cos(np.radians(ang_j + np.degrees(change_ang_j))),
                r_phi * np.sin(np.radians(ang_j + np.degrees(change_ang_j))))
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                connectionstyle="arc3,rad=0.3")
            )
        else:
            # Compute the outer product component
            diff_matrix = (np.eye(2) - change_ang) / 2

            # Extract n from the first non-zero column
            n_column = diff_matrix[:, 0] if np.any(diff_matrix[:, 0]) else diff_matrix[:, 1]
            n = n_column / np.linalg.norm(n_column)
            angle_normal = np.arctan2(n[1], n[0])  # Angle of the normal vector

            # Ensure the angle is within [0, 2π)
            reflection_angle = (2* angle_normal) % (2 * np.pi)

            ax.plot([0, n[0] * (x_max+y_max)/2 ], [0, n[1] * (x_max+y_max)/2 ], color='red', 
                    linewidth=1, alpha=0.5, zorder=2)

            r_phi = 2.6
            start = (r_phi * np.cos(np.radians(ang_i)),
                    r_phi * np.sin(np.radians(ang_i)))
            end = (r_phi * np.cos(np.radians(np.degrees(reflection_angle)-ang_i)),
                r_phi * np.sin(np.radians(np.degrees(reflection_angle)-ang_i)))
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                connectionstyle="arc3,rad=0.3")
            )
            r_phi = 3.5
            start = (r_phi * np.cos(np.radians(ang_j)),
                    r_phi * np.sin(np.radians(ang_j)))
            end = (r_phi * np.cos(np.radians(np.degrees(reflection_angle)-ang_j)),
                r_phi * np.sin(np.radians(np.degrees(reflection_angle)-ang_j)))
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                connectionstyle="arc3,rad=0.3")
            )

        # Labels on arcs
        mid_theta = ang_i + (ang_j - ang_i)/2
        ax.text(r_theta * np.cos(np.radians(mid_theta)) * 1.3,
                r_theta * np.sin(np.radians(mid_theta)) * 1.3,
                f'θ={np.degrees(theta):.1f}°',
                color='blue', fontsize=12)
        
        mid_theta2 = ang_i_new + np.degrees(theta_p)/2
        ax.text(r_theta2 * np.cos(np.radians(mid_theta2)+1) * 1.3,
                r_theta2 * np.sin(np.radians(mid_theta2)-0.5) * 1.3,
                f"θ'=θ",
                color='blue', fontsize=12)

    elif flag == 's':
        ax.set_title(f'Scale by {flag}')

        # Add bold lines along x and y axes indicating scaling factors
        sigma_x = S[0, 0]
        sigma_y = S[1, 1]

        x_i_0_sig = sigma_x * x\_i[0]
        x_i_1_sig = sigma_y * x\_i[1]
        x_j_0_sig = sigma_x * x_j[0]
        x_j_1_sig = sigma_y * x_j[1]

        # X-axis line (red)
        ax.plot([x\_i[0], x_i_0_sig], [0, 0], color='red', linewidth=3, alpha=0.5, zorder=2)
        ax.plot([x_j[0], x_j_0_sig], [0, 0], color='red', linewidth=3, alpha=0.5, zorder=2)
        ax.plot([0, x\_i[0]], [0, 0], color='red', linewidth=1, alpha=0.5, zorder=2)
        ax.plot([0, x_j[0]], [0, 0], color='red', linewidth=1, alpha=0.5, zorder=2)
        # Y-axis line (blue)
        ax.plot([0, 0], [x\_i[1], x_i_1_sig], color='blue', linewidth=3, alpha=0.5, zorder=2)
        ax.plot([0, 0], [x_j[1], x_j_1_sig], color='blue', linewidth=3, alpha=0.5, zorder=2)
        ax.plot([0, 0], [0, x\_i[1]], color='blue', linewidth=1, alpha=0.5, zorder=2)
        ax.plot([0, 0], [0, x_j[1]], color='blue', linewidth=1, alpha=0.5, zorder=2)
        # Align Y-axis line (blue)
        ax.plot([x\_i[0], 0], [x\_i[1], x\_i[1]], color='blue', linewidth=1, alpha=0.2, zorder=2)
        ax.plot([x_i_new[0], 0], [x_i_new[1], x_i_new[1]], color='blue', linewidth=1, alpha=0.2, zorder=2)
        ax.plot([x_j[0], 0], [x_j[1], x_j[1]], color='blue', linewidth=1, alpha=0.2, zorder=2)
        ax.plot([x_j_new[0], 0], [x_j_new[1], x_j_new[1]], color='blue', linewidth=1, alpha=0.2, zorder=2)
        # Align X-axis line (red)
        ax.plot([x\_i[0], x\_i[0]], [0, x\_i[1]], color='red', linewidth=1, alpha=0.2, zorder=2)
        ax.plot([x_i_new[0], x_i_new[0]], [0, x_i_new[1]], color='red', linewidth=1, alpha=0.2, zorder=2)
        ax.plot([x_j[0], x_j[0]], [0, x_j[1]], color='red', linewidth=1, alpha=0.2, zorder=2)
        ax.plot([x_j_new[0], x_j_new[0]], [0, x_j_new[1]], color='red', linewidth=1, alpha=0.2, zorder=2)

        # Labels for sigma values
        ax.text((x\_i[0]+x_i_0_sig)/2-1.0, -0.2, f'$σ^1={sigma_x:.2f}$', color='red', ha='center', va='top', fontsize=12, zorder=4)
        ax.text(-0.3, (x\_i[1]+x_i_1_sig)/2, f'$σ^2={sigma_y:.2f}$', color='blue', ha='right', va='center', fontsize=12, zorder=4)

    ax.set_aspect('equal')
    ax.grid(True)

plot_ax(axs[0], x\_i, x_j, x_i_v, x_j_v,
        ang_i, ang_j, ang_i_v, ang_j_v,
        change_ang_v,
        flag='v')
plot_ax(axs[1], x_i_v, x_j_v, x_i_s, x_j_s,
        ang_i_v, ang_j_v, ang_i_s, ang_j_s,
        change_ang_v,
        flag='s')
plot_ax(axs[2], x_i_s, x_j_s, x_i_u, x_j_u,
        ang_i_s, ang_j_s, ang_i_u, ang_j_u,
        change_ang_u,
        flag='u')

# Set the same limits for all axes
for ax in [axs[0], axs[1], axs[2]]:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

axs[0].text(0.0, -0.2, 
            f"$\\mathbf{{x}}\_i = [{x\_i[0]}, {x\_i[1]}]$", 
            ha='left', va='top', transform=axs[0].transAxes)
axs[0].text(0.0, -0.3, 
            f"$\\mathbf{{x}}_j = [{x_j[0]}, {x_j[1]}]$", 
            ha='left', va='top', transform=axs[0].transAxes)
axs[0].text(0.5, -0.2, 
            f"$A=[[{A[0, 0]}, {A[0, 1]}],$", 
            ha='left', va='top', transform=axs[0].transAxes)
axs[0].text(0.5, -0.3, 
            f"       $[{A[1, 0]}, {A[1, 1]}]]$", 
            ha='left', va='top', transform=axs[0].transAxes)

axs[2].text(2.4, -0.2, 
            f"$\\mathbf{{x}}'_i=A\\mathbf{{x}}\_i = [{x_i_u[0]:.2f}, {x_i_u[1]:.2f}]$", 
            ha='left', va='top', transform=axs[0].transAxes)
axs[2].text(2.4, -0.3, 
            f"$\\mathbf{{x}}'_j=A\\mathbf{{x}}_j = [{x_j_u[0]:.2f}, {x_j_u[1]:.2f}]]$", 
            ha='left', va='top', transform=axs[0].transAxes)

print(x_i_u)
print(x_j_u)

plt.show()
