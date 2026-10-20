import time
from main import *

world = build_world(gui=True)
ctrl = CartesianImpedanceController(world.arm)
g = GAINS_PEN_UP

# Camera ko robot ke paas zoom karo
p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=45, cameraPitch=-30,
                             cameraTargetPosition=[0.5, 0.0, 0.35], physicsClientId=world.cid)

# Hover point ke around 5 cm ka square: har corner pe 1.5 s
c = world.pre_contact_pos
corners = [c, c + [0.05, 0, 0], c + [0.05, 0.05, 0], c + [0, 0.05, 0], c]
drawer = TrajectoryDrawer(world.cid, color=(1, 0, 0))

time.sleep(3)

for k in range(len(corners) - 1):
    traj = MinJerkTrajectory(corners[k], corners[k + 1], 1.5)
    for i in range(1500):
        target, _ = traj.eval(i * DT)
        tau, info = ctrl.step(target, world.nominal_quat, g["K_trans"], g["K_rot"], g["damping_ratio"])
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()
        pos, _, _, _ = world.arm.get_tool_pose()
        if i % 10 == 0:
            drawer.update(pos)          # red line: tool ka actual path
        time.sleep(DT)                  # real-time speed
    print(f"corner {k+1}: error = {np.linalg.norm(pos - corners[k+1])*1000:.2f} mm")

time.sleep(5)
world.disconnect()