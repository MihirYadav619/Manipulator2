#!/usr/bin/env python3
"""
AR523: Robot Manipulators -- Assignment 2
Force-Controlled Writing & Haptic Material Identification
"""

import argparse
import os
import time

import numpy as np
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import *


# ==============================================================================
# PART 1: Trajectory Generation
# ==============================================================================

def _resample_by_arclength(pts, spacing):
    """Dense curve ko equal arc-length spacing pe resample karta hai."""
    pts = np.asarray(pts, dtype=float)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)     # har chhote segment ki length
    s = np.concatenate([[0.0], np.cumsum(seg_len)])            # cumulative arc length
    n = max(int(np.ceil(s[-1] / spacing)), 2) + 1              # kitne points chahiye
    s_new = np.linspace(0.0, s[-1], n)                         # equally spaced arc-length values
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([x, y])


def generate_pattern(shape, scale=0.1, spacing=None):
    """
    TODO 1: Returns a list of strokes (each stroke is a list of (x,y) tuples).
    `scale` = largest dimension of the shape in meters; shape is centered at origin.
    """
    if spacing is None:
        spacing = STROKE_SPEED * 0.15        # ~6 mm -> har segment ~0.15 s (min duration)

    t = np.linspace(0.0, 2.0 * np.pi, 4000)  # dense parameter sweep

    if shape == "circle":
        x, y = np.cos(t), np.sin(t)
    elif shape == "spiral":
        turns = 3
        theta = np.linspace(0.0, 2.0 * np.pi * turns, 6000)
        r = theta / theta[-1]                # radius 0 se 1 tak linearly
        x, y = r * np.cos(theta), r * np.sin(theta)
    elif shape == "infinity":
        d = 1.0 + np.sin(t) ** 2
        x, y = np.cos(t) / d, np.sin(t) * np.cos(t) / d
    elif shape == "heart":
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    else:
        raise ValueError(f"Unknown shape '{shape}'. Use circle, spiral, infinity, heart.")

    pts = np.column_stack([x, y])

    # Normalize: bounding box ka center origin pe, sabse badi side = scale
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    pts = pts - center
    pts = pts * (scale / np.ptp(pts, axis=0).max())

    pts = _resample_by_arclength(pts, spacing)
    return [[(float(px), float(py)) for px, py in pts]]   # ek hi continuous stroke


# ==============================================================================
# PART 2: Cartesian Impedance Control Law
# ==============================================================================

class CartesianImpedanceController:

    def __init__(self, arm, k_null=10.0, d_null=3.0, v_des_max=1.0, a_des_max=5.0):
        self.arm = arm
        # Null-space posture: jis joint config pe robot start hua (elbow yahin rahe)
        self.q_nominal, _ = arm.get_joint_state()
        self.k_null = k_null
        self.d_null = d_null
        # Desired velocity / acceleration from successive setpoints (finite differences)
        self.prev_pos_des = None
        self.prev_v_des = np.zeros(3)
        self.v_des_max = v_des_max   # faster "velocity" = setpoint jump -> ignore
        self.a_des_max = a_des_max   # same idea for acceleration

    def step(self, pos_des, quat_des, K_trans, K_rot, damping_ratio=1.0):
        # TODO 2: Cartesian impedance torque law
        pos_des = np.asarray(pos_des, dtype=float)
        q, qdot = self.arm.get_joint_state()
        J = self.arm.jacobian(q, qdot)                   # (6,7) tool-tip Jacobian
        pos, quat, _, _ = self.arm.get_tool_pose()

        xdot = J @ qdot                                  # [v; w] tool twist
        v, w = xdot[:3], xdot[3:]

        # Desired velocity & acceleration from setpoint changes (zero on jumps)
        jump = self.prev_pos_des is None
        v_des = np.zeros(3)
        if not jump:
            v_des = (pos_des - self.prev_pos_des) / DT
            if np.linalg.norm(v_des) > self.v_des_max:
                v_des = np.zeros(3)
                jump = True
        a_des = np.zeros(3) if jump else (v_des - self.prev_v_des) / DT
        if np.linalg.norm(a_des) > self.a_des_max:
            a_des = np.zeros(3)
        self.prev_pos_des = pos_des.copy()
        self.prev_v_des = v_des.copy()

        # Errors (convention: current - desired)
        e_pos = pos - pos_des
        e_rot = quat_to_axis_angle_error(quat, quat_des)

        # Stiffness & damping: D = 2*zeta*sqrt(K*m)
        K_t = np.asarray(K_trans, dtype=float)
        K_r = np.asarray(K_rot, dtype=float)
        D_t = 2.0 * damping_ratio * np.sqrt(K_t * REFLECTED_MASS_TRANS)
        D_r = 2.0 * damping_ratio * np.sqrt(K_r * REFLECTED_INERTIA_ROT)

        # Virtual spring-damper wrench (damping acts on velocity ERROR)
        F = -K_t * e_pos - D_t * (v - v_des)
        Mo = -K_r * e_rot - D_r * w

        # Task-space inertia: Lambda = (J M^-1 J^T)^-1
        Mq = np.array(p.calculateMassMatrix(self.arm.robot_id, list(q), physicsClientId=self.arm.cid))
        Minv_JT = np.linalg.solve(Mq, J.T)
        Lam = np.linalg.inv(J @ Minv_JT)

        # Acceleration feedforward: the force needed to accelerate the arm's
        # (coupled) inertia along the desired path, so lateral motion does
        # not leak into the normal (z) contact force.
        wrench = np.concatenate([F, Mo]) + Lam @ np.concatenate([a_des, np.zeros(3)])

        # Task torque + gravity compensation
        tau = J.T @ wrench + self.arm.gravity_compensation(q)

        # Dynamically-consistent null-space posture torque (does not disturb the tool)
        J_bar = Minv_JT @ Lam
        N = np.eye(len(q)) - J.T @ J_bar.T
        tau += N @ (-self.k_null * (q - self.q_nominal) - self.d_null * qdot)

        info = dict(e_pos=e_pos, e_rot=e_rot, wrench=wrench, v_des=v_des, a_des=a_des)
        return tau, info



# ==============================================================================
# PART 3: Admittance Control Outer Loop
# ==============================================================================

class AdmittanceController:
    def __init__(self, mass=ADMITTANCE_MASS, damping=ADMITTANCE_DAMPING):
        self.M = mass
        self.B = damping
        self.xc = 0.0
        self.xc_dot = 0.0

    def reset(self):
        self.xc = 0.0
        self.xc_dot = 0.0

    def step(self, f_measured, f_desired, dt, xc_limit=0.03, xc_dot_limit=0.08):
        """
        TODO 3: Admittance mass-damper outer loop step.
            M * xc_ddot + B * xc_dot = f_measured - f_desired
        Positive xc raises the setpoint (too much force -> back off).
        Returns: float (self.xc)
        """
        e_f = f_measured - f_desired                       # force error
        xc_ddot = (e_f - self.B * self.xc_dot) / self.M    # virtual dynamics

        # Semi-implicit Euler: velocity first, then position with the NEW velocity
        self.xc_dot += xc_ddot * dt
        self.xc_dot = float(np.clip(self.xc_dot, -xc_dot_limit, xc_dot_limit))
        self.xc += self.xc_dot * dt

        # Position limit + anti-windup (don't keep pushing into the limit)
        if self.xc > xc_limit:
            self.xc = xc_limit
            self.xc_dot = min(self.xc_dot, 0.0)
        elif self.xc < -xc_limit:
            self.xc = -xc_limit
            self.xc_dot = max(self.xc_dot, 0.0)

        return float(self.xc)

ADMITTANCE_NOMINAL_PENETRATION = 0.015
# Critically damped virtual damping (Part A (d)): B = 2*sqrt(M * K_eff), where
# K_eff is the inner position loop (600 N/m) in SERIES with the contact (750 N/m)
# -- the inner loop is not perfectly stiff, so K_eff = 333 N/m (not K_e = 750).
K_EFF_ADMITTANCE = 1.0 / (1.0 / 600.0 + 1.0 / 750.0)
ADMITTANCE_DAMPING_CRIT = 2.0 * np.sqrt(ADMITTANCE_MASS * K_EFF_ADMITTANCE)   # ~44.7 N.s/m
# Effective contact stiffness of tool-on-surface (Bullet combines the two
# bodies' contactStiffness in series: 1/(1/1500 + 1/1500) = 750 N/m)
CONTACT_STIFFNESS_EFF = 750.0

# Smooth pen-down for impedance writing: descend to 2 mm above the surface
# (APP_T1), ramp the setpoint into the surface (APP_T2), then hold so the
# contact force settles at F_des before the stroke starts (APP_HOLD).
APP_T1, APP_T2, APP_HOLD = 0.30, 0.40, 0.50

# Shift the writing 5 cm away from the robot base. At x < ~0.40 m the KUKA
# elbow (joint 3) hits its -2.094 rad limit and the joint-limit constraint
# absorbs part of the commanded force (measured: 4.34 N instead of 5.0 N).
WRITE_X_OFFSET = 0.05
# Writing impedance: stiff in the plane (accurate tracking against friction),
# compliant only along the surface normal (z) to regulate contact force.
# damping_ratio per axis: z uses ~2.5 so that the CONTACT system
# (robot 150 + surface 750 = 900 N/m) is ~critically damped.
GAINS_WRITE_IMPEDANCE = dict(
    K_trans=np.array([800., 800., 150.]), K_rot=np.array([30., 30., 30.]),
    damping_ratio=np.array([1.5, 1.5, 2.5]))


def impedance_contact_z(world, f_desired, K_z):
    """Virtual setpoint below the surface that yields f_desired with
    robot spring (K_z) and contact spring (CONTACT_STIFFNESS_EFF) in series."""
    return world.surface_top_z + TOOL_RADIUS - f_desired * (1.0 / K_z + 1.0 / CONTACT_STIFFNESS_EFF)

# ==============================================================================
# LIVE VISUALIZATION (TODO 8)
# ==============================================================================

class LiveDashboard:
    def __init__(self, mode="writing"):
        """
        TODO 7: Implement a real-time matplotlib dashboard using plt.ion().
        """
        raise NotImplementedError("TODO 7: Create a live plotting window using matplotlib.")

    def update_writing(self, t, pos_act, pos_des, f_act, f_des):
        self.t_data.append(t)
        if f_des > 0:
            self.x_act.append(pos_act[0]); self.y_act.append(pos_act[1])
            self.x_des.append(pos_des[0]); self.y_des.append(pos_des[1])
        else:
            self.x_act.append(np.nan); self.y_act.append(np.nan)
            self.x_des.append(np.nan); self.y_des.append(np.nan)
        self.f_act.append(f_act[2])
        self.f_des.append(f_des)
        
        now = time.time()
        if now - self.last_update > 0.05:
            self.line_xy_act.set_data(self.x_act, self.y_act)
            self.line_xy_des.set_data(self.x_des, self.y_des)
            self.ax_xy.relim()
            self.ax_xy.autoscale_view()
            
            self.line_f_act.set_data(self.t_data, self.f_act)
            self.line_f_des.set_data(self.t_data, self.f_des)
            self.ax_f.relim()
            self.ax_f.autoscale_view()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.last_update = now

    def update_material(self, depth, f_act, k_hat=None):
        self.depths.append(depth)
        self.f_act.append(f_act)
        
        now = time.time()
        if now - self.last_update > 0.05:
            self.line_fd.set_data(self.depths, self.f_act)
            if k_hat is not None and len(self.depths) > 0:
                d_max = max(self.depths)
                self.line_fit.set_data([0, d_max], [0, k_hat * d_max])
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.last_update = now
            
    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ==============================================================================
# WRITING & PROBING RUNNERS
# ==============================================================================

def build_stroke_plan(world, letter_origin_xy, word=None, pattern=None):
    strokes = []
    if word:
        strokes.extend(get_word_strokes(word))
    if pattern:
        pattern_scale = 0.06                       # 6 cm, fits comfortably on the surface
        pattern_strokes = generate_pattern(pattern, scale=pattern_scale)
        # Place pattern centred BELOW the word (the original +0.5 m x-offset
        # pushed it off the 0.4 m-wide writing surface)
        y_off = -(pattern_scale / 2 + 0.02) if word else 0.0
        pattern_strokes = [[(px, py + y_off) for (px, py) in seg] for seg in pattern_strokes]
        strokes.extend(pattern_strokes)
        
    plan = []
    for stroke in strokes:
        pts = [np.array(letter_origin_xy) + np.array(pt) for pt in stroke]
        for i in range(len(pts)):
            if i == 0:
                if plan: plan.append(("up", plan[-1][2], pts[0]))
                else:    plan.append(("up", world.pre_contact_pos[:2], pts[0]))
            if i < len(pts) - 1:
                plan.append(("down", pts[i], pts[i+1]))
    return plan


def run_writing_task(world, ctrl, controller_mode, f_desired, gui=False, save_dir="outputs", tag="writing", word="Robot", pattern=None, live=False):
    plan = build_stroke_plan(world, world.surface_center_xy + np.array([WRITE_X_OFFSET, 0.0]), word=word, pattern=pattern)
    adm = AdmittanceController(damping=ADMITTANCE_DAMPING_CRIT)

    drawer = TrajectoryDrawer(world.cid, color=(0, 0, 0)) if gui else None
    dashboard = LiveDashboard(mode="writing") if live else None
    logger = TrialLogger()
    logger.start_timer()

    first_xy = plan[0][1]
    lift_pos = np.array([first_xy[0], first_xy[1], world.surface_top_z + PEN_LIFT_HEIGHT])
    traj_in = MinJerkTrajectory(world.pre_contact_pos, lift_pos, APPROACH_DURATION)

    t = 0.0
    for step in range(int(APPROACH_DURATION / DT)):
        pos_des, _ = traj_in.eval(t)
        tau, _ = ctrl.step(pos_des, world.nominal_quat, GAINS_PEN_UP["K_trans"], GAINS_PEN_UP["K_rot"], GAINS_PEN_UP["damping_ratio"])
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()
        tool_pos, _, _, _ = world.arm.get_tool_pose()
        logger.record(t, tool_pos, pos_des, force=np.zeros(3), force_des=0.0)
        t += DT

    prev_kind = None
    for kind, xy_from, xy_to in plan:
        if kind == "up":
            p_from = np.array([xy_from[0], xy_from[1], world.surface_top_z + PEN_LIFT_HEIGHT])
            p_to = np.array([xy_to[0], xy_to[1], world.surface_top_z + PEN_LIFT_HEIGHT])
            traj = MinJerkTrajectory(p_from, p_to, PEN_UP_DURATION)
            for step in range(int(PEN_UP_DURATION / DT)):
                pos_des, _ = traj.eval(step * DT)
                tau, _ = ctrl.step(pos_des, world.nominal_quat, GAINS_PEN_UP["K_trans"], GAINS_PEN_UP["K_rot"], GAINS_PEN_UP["damping_ratio"])
                world.arm.apply_joint_torques(tau)
                p.stepSimulation(physicsClientId=world.cid)
                world.arm.clamp_joint_velocities()
                tool_pos, _, _, _ = world.arm.get_tool_pose()
                logger.record(t, tool_pos, pos_des, force=np.zeros(3), force_des=0.0)
                t += DT
            adm.reset()
        else:
            if prev_kind != "down":
                adm.reset()
                z_lift = world.surface_top_z + PEN_LIFT_HEIGHT
                z_touch = world.surface_top_z + TOOL_RADIUS
                # Smooth two-phase pen-down (both controllers): descend to 2 mm above
                # the surface, ramp the setpoint to its nominal contact depth, hold.
                if controller_mode == "admittance":
                    g = GAINS_INNER_STIFF
                    z_contact = z_touch - ADMITTANCE_NOMINAL_PENETRATION   # ~5 N with xc = 0
                else:
                    g = GAINS_WRITE_IMPEDANCE
                    z_contact = impedance_contact_z(world, f_desired, g["K_trans"][2])
                phases = [(z_lift, z_touch + 0.002, APP_T1, 0.0, g),
                          (z_touch + 0.002, z_contact, APP_T2, APP_HOLD, g)]
                for z_a, z_b, T_move, T_hold, g_app in phases:
                    traj_app = MinJerkTrajectory(np.array([xy_from[0], xy_from[1], z_a]),
                                                 np.array([xy_from[0], xy_from[1], z_b]), T_move)
                    for step in range(int((T_move + T_hold) / DT)):
                        pos_des, _ = traj_app.eval(step * DT)   # eval clips at end -> hold
                        tau, _ = ctrl.step(pos_des, world.nominal_quat,
                                           g_app["K_trans"], g_app["K_rot"], g_app["damping_ratio"])
                        world.arm.apply_joint_torques(tau)
                        p.stepSimulation(physicsClientId=world.cid)
                        world.arm.clamp_joint_velocities()
                
            stroke_len = np.linalg.norm(xy_to - xy_from)
            duration = max(stroke_len / STROKE_SPEED, 0.15)
            xy_traj = MinJerkTrajectory(np.append(xy_from, 0), np.append(xy_to, 0), duration)
            for step in range(int(duration / DT)):
                xy_pos, _ = xy_traj.eval(step * DT)
                tool_pos, _, _, _ = world.arm.get_tool_pose()
                force, _, _ = get_contact_wrench(world.cid, world.arm.tool_id, world.static_ids, tool_pos)
                f_meas_z = force[2]

                if controller_mode == "admittance":
                    xc = adm.step(f_meas_z, f_desired, DT)
                    z_des = world.surface_top_z + TOOL_RADIUS - ADMITTANCE_NOMINAL_PENETRATION + xc
                    gains = GAINS_INNER_STIFF
                else:
                    # TODO 2 (continued): Impedance Writing
                    # Robot spring (K_z) and contact spring in series:
                    #   F = d / (1/K_z + 1/K_contact)  ->  d = F * (1/K_z + 1/K_contact)
                    gains = GAINS_WRITE_IMPEDANCE
                    z_des = impedance_contact_z(world, f_desired, gains["K_trans"][2])

                pos_des = np.array([xy_pos[0], xy_pos[1], z_des])
                tau, _ = ctrl.step(pos_des, world.nominal_quat, gains["K_trans"], gains["K_rot"], gains["damping_ratio"])
                world.arm.apply_joint_torques(tau)
                p.stepSimulation(physicsClientId=world.cid)
                world.arm.clamp_joint_velocities()

                tool_pos, _, _, _ = world.arm.get_tool_pose()
                force, torque, _ = get_contact_wrench(world.cid, world.arm.tool_id, world.static_ids, tool_pos)
                
                logger.record(t, tool_pos, pos_des, force=force, force_des=f_desired, torque=torque)
                if dashboard:
                    dashboard.update_writing(t, tool_pos, pos_des, force, f_desired)
                if drawer is not None and step % 3 == 0:
                    drawer.update(tool_pos)
                t += DT
        prev_kind = kind
    
    logger.stop_timer()
    if dashboard: dashboard.close()

    os.makedirs(save_dir, exist_ok=True)
    arr = logger.arrays()
    
    # --------------------------------------------------------------------------
    # TODO 4: Compute Metrics
    # --------------------------------------------------------------------------
    f_meas = arr["force"][:, 2]
    f_des_arr = arr["force_des"]
    contact = f_des_arr > 0

    # Force tracking RMSE (pen-down samples only)
    if contact.any():
        rmse = float(np.sqrt(np.mean((f_meas[contact] - f_des_arr[contact]) ** 2)))
    else:
        rmse = 0.0

    # Mean squared jerk, computed separately on each continuous pen-down run
    # (avoids fake jerk spikes at the unlogged approach gaps between strokes)
    edges = np.diff(np.concatenate([[0], contact.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    jerk_sq = []
    for s0, s1 in zip(starts, ends):
        seg = arr["pos"][s0:s1]
        if len(seg) < 4:
            continue
        vel = np.gradient(seg, DT, axis=0)
        acc = np.gradient(vel, DT, axis=0)
        jerk = np.gradient(acc, DT, axis=0)
        jerk_sq.append(np.sum(jerk ** 2, axis=1))
    smoothness = float(np.mean(np.concatenate(jerk_sq))) if jerk_sq else 0.0
    
    plot_force_tracking(arr, os.path.join(save_dir, f"{tag}_force_tracking.png"), title=f"{tag}: Force Tracking While Writing")
    plot_xy_path(arr, os.path.join(save_dir, f"{tag}_xy_path.png"), title=f"{tag}: Traced Path (top-down)")
    
    summary = dict(force_tracking_rmse_N=rmse, path_smoothness_jerk2=smoothness, wall_clock_time_s=logger.wall_clock_time(), sim_time_s=float(arr["t"][-1]))
    save_json(os.path.join(save_dir, f"{tag}_summary.json"), summary)
    print(f"[{tag}] force RMSE = {rmse:.3f} N")
    return summary, arr


def probe_material(world, ctrl, pad_id, pad_top_z, xy, save_dir, tag, fixed_penetration=0.010, settle_duration=3.0, live=False):
    gains = dict(K_trans=np.array([600., 600., 600.]), K_rot=np.array([30., 30., 30.]), damping_ratio=1.8)
    just_touching_z = pad_top_z + TOOL_RADIUS
    target = np.array([xy[0], xy[1], just_touching_z - fixed_penetration])

    hover = np.array([xy[0], xy[1], just_touching_z + 0.05])
    q_hover = world.arm.inverse_kinematics(hover, world.nominal_quat)
    world.arm.set_joint_positions(q_hover)
    world.arm.sync_tool_pose()

    traj = MinJerkTrajectory(hover, target, APPROACH_DURATION)
    dashboard = LiveDashboard(mode="material") if live else None
    logger = TrialLogger()
    t = 0.0
    for step in range(int((APPROACH_DURATION + settle_duration) / DT)):
        pos_des, _ = traj.eval(t)
        tau, _ = ctrl.step(pos_des, world.nominal_quat, gains["K_trans"], gains["K_rot"], gains["damping_ratio"])
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()

        tool_pos, _, _, _ = world.arm.get_tool_pose()
        force, torque, _ = get_contact_wrench(world.cid, world.arm.tool_id, [pad_id], tool_pos)
        
        depth = max(0.0, just_touching_z - tool_pos[2])
        logger.record(t, tool_pos, pos_des, force=force, force_des=0.0, torque=torque)
        if dashboard:
            dashboard.update_material(depth, force[2])
        t += DT

    arr = logger.arrays()
    forces = arr["force"][:, 2]
    depths = np.maximum(0.0, just_touching_z - arr["pos_des"][:, 2])

    # --------------------------------------------------------------------------
    # TODO 5: Estimate Contact Stiffness
    # Estimate contact stiffness K_hat via least-squares fit (F = K * depth).
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 5: Calculate the stiffness by finding the slope of the force vs. depth data.")
    K_hat = 0.0

    if dashboard:
        dashboard.update_material(depths[-1], forces[-1], k_hat=K_hat)
        time.sleep(1)
        dashboard.close()

    os.makedirs(save_dir, exist_ok=True)
    steady_force = float(np.percentile(forces[-max(int(2.0/DT), 5):], 75))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(arr["t"], forces, color=RED, lw=1.8)
    ax.axhline(steady_force, color=GREY, ls="--", lw=1.2, label=f"steady state: F={steady_force:.1f} N, K̂={K_hat:.0f} N/m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contact force (N)")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Material Probe -- {tag}", fontsize=12, weight="bold", color=NAVY)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}_loading_curve.png"), dpi=180, facecolor="white")
    plt.close(fig)
    return K_hat


# ==============================================================================
# MAIN EXECUTIONS
# ==============================================================================

def run_part1(save_dir="outputs/part1"):
    print("--- Part 1: Trajectory Generation ---")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    shapes = ["circle", "spiral", "infinity", "heart"]
    for i, shape in enumerate(shapes):
        strokes = generate_pattern(shape, scale=0.1)
        ax = axes[i]
        for s in strokes:
            pts = np.array(s)
            ax.plot(pts[:, 0], pts[:, 1], 'b-', lw=2)
        ax.set_aspect("equal")
        ax.set_title(shape.capitalize())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shapes.png"), dpi=150)
    plt.close()

def run_part2_writing(gui=False, save_dir="outputs/part3", f_desired=F_DESIRED, word="Robot", pattern="circle", live=False):
    print("--- Part 2 (continued): Impedance Writing ---")
    world = build_world(gui=gui)
    ctrl = CartesianImpedanceController(world.arm)
    s, _ = run_writing_task(world, ctrl, "impedance", f_desired, gui=gui, save_dir=save_dir, tag="part3_impedance", word=word, pattern=pattern, live=live)
    world.disconnect()
    return s

def run_part2_writing_admittance(gui=False, save_dir="outputs/part4", f_desired=F_DESIRED, word="Robot", pattern="circle", live=False):
    print("--- Part 3: Admittance Writing ---")
    world = build_world(gui=gui)
    ctrl = CartesianImpedanceController(world.arm)
    s, _ = run_writing_task(world, ctrl, "admittance", f_desired, gui=gui, save_dir=save_dir, tag="part4_admittance", word=word, pattern=pattern, live=live)
    world.disconnect()
    return s

def run_part3(save_dir="outputs/part5"):
    print("--- Part 4: Analysis and Comparison ---")
    os.makedirs(save_dir, exist_ok=True)
    s3 = run_part2_writing(gui=False)
    s4 = run_part3_admittance(gui=False)
    
    dynamic_cmp = {"Impedance (Part 2)": s3, "Admittance (Part 3)": s4}
    bar_compare(["Force RMSE (N)"], {k: [v["force_tracking_rmse_N"]] for k, v in dynamic_cmp.items()}, "RMSE (N)", os.path.join(save_dir, "part5_writing_force_rmse.png"), "Writing: Force Tracking RMSE")
    bar_compare(["Path Smoothness (mean sq. jerk)"], {k: [v["path_smoothness_jerk2"]] for k, v in dynamic_cmp.items()}, "Mean squared jerk", os.path.join(save_dir, "part5_writing_smoothness.png"), "Writing: Motion Smoothness")

def run_part4(gui=False, save_dir="outputs/part6", live=False):
    print("--- Part 5: Haptic Material Identification ---")
    os.makedirs(save_dir, exist_ok=True)
    world = build_world(gui=gui, with_material_samples=True)
    ctrl = CartesianImpedanceController(world.arm)
    results = {}
    for name, (pad_id, pad_top_z, true_label) in world.material_ids.items():
        spec = MATERIAL_SAMPLES[name]
        xy = world.surface_center_xy + np.array(spec["offset_xy"])
        K_hat = probe_material(world, ctrl, pad_id, pad_top_z, xy, save_dir, name, live=live)
        
        # ----------------------------------------------------------------------
        # TODO 6: Classify Material
        # Classify material as 'soft', 'medium', or 'hard' based on K_hat and MATERIAL_CLASS_THRESHOLDS.
        # ----------------------------------------------------------------------
        raise NotImplementedError("TODO 6: Use the stiffness to classify the material as \'soft\', \'medium\', or \'hard\'.")
        pred_label = "unknown"
            
        results[name] = dict(true_stiffness_N_per_m=spec["contact_stiffness"], estimated_stiffness_N_per_m=K_hat, true_label=true_label, predicted_label=pred_label, correct=(pred_label == true_label))
        print(f"[Part 5] {name}: estimated K̂={K_hat:.0f} N/m -> predicted={pred_label}")
    n_correct = sum(1 for r in results.values() if r["correct"])
    save_json(os.path.join(save_dir, "part6_results.json"), dict(accuracy=n_correct/3.0, results=results))
    world.disconnect()
    return results

def main():
    parser = argparse.ArgumentParser(description="Force-Controlled Writing")
    parser.add_argument("--part", choices=["1", "2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--word", type=str, default="Robot", help="Word to write")
    parser.add_argument("--pattern", type=str, default="circle", help="Pattern to draw")
    parser.add_argument("--live", action="store_true", help="Enable live dashboard visualization")
    args = parser.parse_args()
    
    pattern = None if args.pattern == "none" else args.pattern
    
    if args.part in ["1", "all"]: run_part1()
    if args.part in ["2", "all"]: run_part2_writing(gui=args.gui, word=args.word, pattern=pattern, live=args.live)
    if args.part in ["3", "all"]: run_part3(gui=args.gui, word=args.word, pattern=pattern, live=args.live)
    if args.part in ["4", "all"]: run_part4()
    if args.part in ["5", "all"]: run_part5(gui=args.gui, live=args.live)

if __name__ == '__main__':
    main()