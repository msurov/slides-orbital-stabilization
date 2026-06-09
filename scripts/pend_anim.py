import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation
from scipy.integrate import solve_ivp
import numpy as np
from bisect import bisect_left
from typing import Tuple

class PendView:
  rod_width = 0.03
  rod_length = 1.0
  joint_radius = 0.04
  mass_radius = 0.06
  joint_color = '#202020'
  mass_color = '#2020A0'
  rod_color = '#2020A0'

  def __init__(self, ax, color=None):
    if color is not None:
      self.mass_color = self.rod_color = color
    circle_pos, rod_pos, rod_angle = self.get_objs_pos(0.)
    self.mass = Circle(circle_pos, self.mass_radius, color=self.mass_color)
    self.joint = Circle([0, 0], self.joint_radius, color=self.joint_color)
    self.rod = Rectangle(xy=rod_pos, width=self.rod_width, height=self.rod_length, angle=rod_angle, color=self.rod_color)
    for p in self.patches:
      ax.add_patch(p)

  def move(self, theta):
    circle_pos, rod_pos, rod_angle = self.get_objs_pos(theta)
    self.mass.set_center(circle_pos)
    self.rod.set_xy(rod_pos)
    self.rod.set_angle(rod_angle)

  def get_objs_pos(self, theta):
    s = np.sin(theta)
    c = np.cos(theta)
    R = np.array([
      [c, -s],
      [s, c]
    ])
    rod_pos = R @ np.array([-self.rod_width/2, -self.rod_length])
    circle_pos = R @ np.array([0, -self.rod_length])
    return circle_pos, rod_pos, np.rad2deg(theta)

  @property
  def patches(self):
    return self.rod, self.joint, self.mass

def add_annotation(text : str, textpos : Tuple[int, int], fontsize=16):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 1
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': fontsize
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def calc_pend_traj(initial_angle, interval, step):
  def rhs(t, st):
    theta, dtheta = st
    # ddtheta = -10 * np.sin(theta)
    ddtheta = -1 * np.sin(theta)
    return [dtheta, ddtheta]

  t = np.arange(0, interval, step)
  sol = solve_ivp(rhs, [0, interval], [initial_angle, 0.], max_step=1e-2, t_eval=t)
  return sol.t, sol.y.T

def get_dist(p, arr):
  d = np.linalg.norm(arr - p, axis=1)
  return np.min(d)

def make_anim():
  fps = 60
  simtime = 80
  time_hist = 20
  speedup = 2

  t, traj1 = calc_pend_traj(1, simtime, 5e-3)
  _, traj2 = calc_pend_traj(1.1, simtime, 5e-3)

  fig, axes = plt.subplots(3, 2, figsize=(6,6))

  ax_pend1 = axes[0,0]
  ax_pend2 = axes[0,1]

  ax_pend1.set_aspect('equal')
  ax_pend2.set_aspect('equal')

  ax_pend1.set_xlim(-1.0, 1.0)
  ax_pend1.set_ylim(-1.1, 0.1)
  ax_pend1.set_xticks([])
  ax_pend1.set_yticks([])

  ax_pend2.set_xlim(-1.0, 1.0)
  ax_pend2.set_ylim(-1.1, 0.1)
  ax_pend2.set_xticks([])
  ax_pend2.set_yticks([])

  pend1 = PendView(ax_pend1, color='orange')
  pend2 = PendView(ax_pend2, color='blue')

  ax_trajs = axes[1,0]
  plot_traj1, = ax_trajs.plot([], [], color='orange')
  plot_traj2, = ax_trajs.plot([], [], color='blue')
  ax_trajs.grid(True)
  ax_trajs.set_xlim(0, time_hist)
  ax_trajs.set_ylim(-1.2, 1.2)
  ax_trajs.legend([R'$x_\star$', R'$x_s$'], fontsize=12, loc='upper right')
  ax_trajs.tick_params(axis='x', colors='#00000000')

  ax_phase = axes[1,1]
  plt.sca(ax_phase)
  plot_phase_curve, = ax_phase.plot([], [], color='blue', lw=1)
  plot_phase_pt1, = ax_phase.plot([], [], 'o', color='orange')
  plot_phase_pt2, = ax_phase.plot([], [], 'o', color='blue')
  ax_phase.set_aspect('equal')
  ax_phase.plot(traj1[:,0], traj1[:,1], color='orange', lw=1)
  ax_phase.set_xlim(-1.7, 1.7)
  ax_phase.set_ylim(-1.3, 1.3)
  ax_phase.grid(True)
  plot_perp, = ax_phase.plot([], [], color='black', lw=1)
  add_annotation(R'$\dot\theta$', [0.05, 0.83], fontsize=16)
  add_annotation(R'$\theta$', [0.90, 0.08], fontsize=16)

  ax_dif = axes[2,0]
  plot_dif, = ax_dif.plot([], [])
  traj_dif = np.linalg.norm(traj1 - traj2, axis=1)
  ax_dif.set_ylim(-0.2, 1.1 * np.max(traj_dif))
  ax_dif.set_xlim(0, time_hist)
  ax_dif.grid(True)
  ax_dif.legend([R'$\Vert x_s(t) - x_\star(t) \Vert$'], loc='lower right', fontsize=14)

  ax_dist = axes[2,1]
  plot_dist, = ax_dist.plot([], [])
  dist = np.array([get_dist(p, traj1) for p in traj2])
  ax_dist.set_ylim(0, 1.1 * np.max(dist))
  ax_dist.set_xlim(0, time_hist)
  ax_dist.legend([R'$\mathrm{dist}(x_s(t), \mathrm{orb} \, x_\star)$'], loc='lower right', fontsize=14)
  ax_dist.grid(True)

  def drawframe(iframe):
    now = iframe * speedup / fps
    ipoint = bisect_left(t, now) - 1

    pend1.move(traj1[ipoint,0])
    pend2.move(traj2[ipoint,0])

    mask = (t <= now) & (t > now - time_hist)

    plot_dif.set_data(t[mask], traj_dif[mask])
    plot_dist.set_data(t[mask], dist[mask])
    plot_phase_curve.set_data(traj2[mask,0], traj2[mask,1])
    plot_traj1.set_data(t[mask], traj1[mask, 0])
    plot_traj2.set_data(t[mask], traj2[mask, 0])

    plot_phase_pt1.set_data(traj1[ipoint:ipoint+1,0], traj1[ipoint:ipoint+1,1])
    plot_phase_pt2.set_data(traj2[ipoint:ipoint+1,0], traj2[ipoint:ipoint+1,1])

    v = traj2[ipoint, 0:2]
    v = v / np.linalg.norm(v)
    p1 = 0.7 * v
    p2 = 1.3 * v

    plot_perp.set_data([p1[0], p2[0]], [p1[1], p2[1]])

    t1 = max(now - time_hist + 0.1, 0)
    t2 = t1 + time_hist
    ax_dif.set_xlim(t1, t2)
    ax_dist.set_xlim(t1, t2)
    ax_trajs.set_xlim(t1, t2)

    return pend1.patches + pend2.patches + \
      (plot_dif, plot_dist, plot_phase_curve, plot_perp, 
       plot_traj1, plot_traj2, plot_phase_pt1, plot_phase_pt2)

  plt.tight_layout(h_pad=0.1, pad=0.5, w_pad=0.1)
  nframes = int(simtime * fps / speedup - 1)
  anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=1000/fps, blit=True)
  anim.save('../media/pend_anim.mp4', fps=60, dpi=200)
  # plt.show()

if __name__ == '__main__':
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  make_anim()
