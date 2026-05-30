import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# load trajectory
# os.makedirs("results", exist_ok=True)
traj = torch.load("lift_block_traj_1.pt", map_location="cpu")


# extract params
ee = torch.cat([x["ee_pos"] for x in traj], dim=0).numpy()
object = torch.cat([x["object_pos"] for x in traj], dim=0).numpy()
steps = [x["step"] for x in traj]
rewards = [x["reward"].item() for x in traj]
terminated = [x["terminated"].item() for x in traj]
truncated = [x["truncated"].item() for x in traj]
success = [x["object_lifted"] for x in traj]

# 2D top view
plt.figure(figsize=(6, 6))

plt.plot(ee[:, 0], ee[:, 1], label="EE Trajectory")
plt.plot(object[:, 0], object[:,1 ], label="object Trajectory")

plt.scatter(ee[0, 0], ee[0, 1], marker="o", label="EE Start")
plt.scatter(ee[-1, 0], ee[-1, 1], marker="x", label="EE End")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Top View Trajectory")
plt.legend()
plt.axis("equal")

plt.savefig("results/top_view_1.png", dpi=300, bbox_inches="tight")
plt.close()


# 3D trajectory
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    ee[:,0],
    ee[:,1],
    ee[:,2],
    label="EE"
)

ax.plot(
    object[:,0],
    object[:,1],
    object[:,2],
    label="object"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("3D Trajectory")
ax.legend()

plt.savefig("results/trajectory_3d_1.png", dpi=300, bbox_inches="tight")
plt.close()


# object lifting curve
plt.figure(figsize=(8,4))

plt.plot(object[:,2])

plt.xlabel("Episode (per 5 steps)")
plt.ylabel("object Height (Z)")
plt.title("object Lifting")

plt.savefig("results/object_height_1.png", dpi=300, bbox_inches="tight")
plt.close()


# reward curve
plt.figure(figsize=(8,4))

plt.plot(steps, rewards)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Curve")

plt.savefig("results/reward_1.png", dpi=300, bbox_inches="tight")
plt.close()


# terminated
plt.figure(figsize=(8,4))

plt.plot(steps, terminated)
plt.xlabel("Step")
plt.ylabel("Terminated")
plt.title("Terminated Curve")

plt.savefig("results/terminated_1.png", dpi=300, bbox_inches="tight")
plt.close()

# truncated
plt.figure(figsize=(8,4))

plt.plot(steps, truncated)
plt.xlabel("Step")
plt.ylabel("Truncated")
plt.title("Truncated Curve")

plt.savefig("results/truncated_1.png", dpi=300, bbox_inches="tight")
plt.close()

# success
plt.figure(figsize=(8,4))

plt.plot(steps, success)
plt.xlabel("Step")
plt.ylabel("Success")
plt.title("Success Curve")

plt.savefig("results/success_1.png", dpi=300, bbox_inches="tight")
plt.close()

# interactive 3D trajectory
fig_html = go.Figure()

fig_html.add_trace(
    go.Scatter3d(
        x=ee[:, 0],
        y=ee[:, 1],
        z=ee[:, 2],
        mode="lines+markers",
        name="EE"
    )
)

fig_html.add_trace(
    go.Scatter3d(
        x=object[:, 0],
        y=object[:, 1],
        z=object[:, 2],
        mode="lines+markers",
        name="object"
    )
)

fig_html.update_layout(
    title="Interactive 3D Trajectory",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    )
)

fig_html.write_html("results/traj_3d_1.html")

print("Saved:")
print(" results/top_view.png")
print(" results/object_height.png")
print(" results/reward.png")
print(" results/traj_3d.html")