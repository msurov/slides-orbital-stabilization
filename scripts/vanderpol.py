import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation
from scipy.integrate import solve_ivp
import numpy as np
from bisect import bisect_left
from typing import Tuple
from scipy.interpolate import make_interp_spline


def get_dist(p, arr):
  d = np.linalg.norm(arr - p, axis=1)
  return np.min(d)

def find_closest_point(p, arr):
  d = np.linalg.norm(arr - p, axis=1)
  return np.argmin(d)

def add_annotation(text : str, textpos : Tuple[int, int], fontsize=16, xycoords='axes fraction', bgalpha=1):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': bgalpha
  }
  annotate_par = {
    'xycoords': xycoords,
    'font': {
      'size': fontsize
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def calc_oscillator_traj(state0, interval, step, mu=2):

  def rhs(t, st):
    x, dx = st
    ddx = mu * (1 - x**2) * dx - x
    return [dx, ddx]

  t = np.arange(0, interval, step)
  sol = solve_ivp(rhs, [0, interval], state0, max_step=1e-2, t_eval=t)
  return sol.t, sol.y.T

def plot_phase():
  plot_par = {
    'color': '#20209070',
    'lw': 1
  }

  x0 = 0
  for dx0 in np.linspace(4, 8, 5):
    _, x_sample = calc_oscillator_traj([x0, dx0], 2.5, 1e-2)
    plt.plot(x_sample[:,0], x_sample[:,1], **plot_par)

    _, x_sample = calc_oscillator_traj([x0, -dx0], 2.5, 1e-2)
    plt.plot(x_sample[:,0], x_sample[:,1], **plot_par)

  x0 = 2.5
  for dx0 in np.linspace(0, -5, 4):
    _, x_sample = calc_oscillator_traj([x0, dx0], 2.5, 1e-2)
    plt.plot(x_sample[:,0], x_sample[:,1], **plot_par)

    _, x_sample = calc_oscillator_traj([-x0, -dx0], 2.5, 1e-2)
    plt.plot(x_sample[:,0], x_sample[:,1], **plot_par)
    
  dx0 = -5
  for x0 in np.linspace(1.8, 3, 7):
    _, x_sample = calc_oscillator_traj([x0, dx0], 2.5, 1e-2)
    plt.plot(x_sample[:,0], x_sample[:,1], **plot_par)

    _, x_sample = calc_oscillator_traj([-x0, -dx0], 2.5, 1e-2)
    plt.plot(x_sample[:,0], x_sample[:,1], **plot_par)

  for x0 in [0.01, 0.1, 0.3, 0.6]:
    _, x_sample = calc_oscillator_traj([x0, 0.0], 20, 1e-2)
    plt.plot(x_sample[:,0], x_sample[:,1], **plot_par)

    _, x_sample = calc_oscillator_traj([-x0, 0.0], 20, 1e-2)
    plt.plot(x_sample[:,0], x_sample[:,1], **plot_par)

  _, x_ref = calc_oscillator_traj([2.02, 0], 8, 1e-2)
  plt.plot(x_ref[:,0], x_ref[:,1], lw=3, color='orange')

  plt.xlim(-2.5, 2.5)
  plt.ylim(-4, 4)
  plt.grid(True)
  plt.show()


def plot_sample():
  plot_par = {
    'color': '#20A0A0A0',
    'lw': 1
  }

  _, x_sample = calc_oscillator_traj([2, -4], 8, 1e-3)
  t, x_ref = calc_oscillator_traj([2.02, 0], 8, 1e-3)

  dist = np.array([get_dist(p, x_ref) for p in x_sample])

  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  plt.sca(axes[0])
  plt.axvline(0, ls='--', lw=1, color='#202020B0')
  plt.axhline(0, ls='--', lw=1, color='#202020B0')
  plot_sample, = plt.plot(x_sample[:,0], x_sample[:,1], lw=2, color='blue', label=R'$x_s$')
  plot_ref, = plt.plot(x_ref[:,0], x_ref[:,1], lw=3, color='orange', label=R'$x_\star$')

  for x0 in np.arange(0.01, 1, 0.3):
    _, x1 = calc_oscillator_traj([x0, 0], 8, 1e-2)
    plt.plot(x1[:,0], x1[:,1], **plot_par, zorder=-1)
    plt.plot(-x1[:,0], -x1[:,1], **plot_par, zorder=-1)

  for dx0 in np.arange(0.5, 20, 2):
    x0 = 3
    _, x1 = calc_oscillator_traj([x0, -dx0], 8, 1e-2)
    plt.plot(x1[:,0], x1[:,1], **plot_par, zorder=-1)
    plt.plot(-x1[:,0], -x1[:,1], **plot_par, zorder=-1)

  plt.xlim(-2.5, 2.5)
  plt.ylim(-5, 4)
  add_annotation(R'$x_1$', [0.55, -0.08], fontsize=18)
  add_annotation(R'$x_2$', [-0.12, 0.54], fontsize=18)
  plt.legend(fontsize=18, loc='upper left')

  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(t, dist, lw=2, color='blue')
  plt.legend([R'$\mathrm{dist}(x_s(t), \mathrm{orb}\, x_\star)$'], fontsize=18)
  add_annotation(R'$t$', [0.55, -0.08], fontsize=18)

  plt.tight_layout(pad=1, h_pad=1)
  plt.savefig('../media/vanderpol_oscillator.svg')
  plt.show()

def plot_orbit_deviation():
  _, x_sample = calc_oscillator_traj([2, -4], 10, 1e-3)
  t, x_ref = calc_oscillator_traj([2.02, 0], 10, 1e-3)
  dif = np.linalg.norm(x_sample - x_ref, axis=1)
  dist = np.array([get_dist(p, x_ref) for p in x_sample])

  plt.figure('dif', figsize=(6, 4))
  plt.plot(t, dif)
  plt.grid(True)
  plt.legend([R'$\Vert x(t) - x_\star(t) \Vert$'], fontsize=16)
  plt.xlabel('time, sec', fontsize=14)
  plt.tight_layout(pad=0.1)
  plt.savefig('../media/vanderpol-dif.svg')

  plt.figure('dist', figsize=(6, 4))
  plt.plot(t, dist)
  plt.grid(True)
  plt.legend([R'$\mathrm{dist}(x(t), \mathrm{orb}\, x_\star)$'], fontsize=16)
  plt.xlabel('time, sec', fontsize=14)
  plt.tight_layout(pad=0.1)
  plt.savefig('../media/vanderpol-orb-dist.svg')

  plt.show()

def plot_line_seg(p1, p2, **kwargs):
  x1, y1 = p1
  x2, y2 = p2
  return plt.plot([x1, x2], [y1, y2], **kwargs)

def plot_dot(p, **kwargs):
  plt.plot(p[0], p[1], 'o', **kwargs)

def plot_section_axis(p, tan, axis_length=1, tick_length=0.1, nticks=7, center_size=6):
  S = np.array([
    [0, -1],
    [1, 0]
  ])
  nor = S @ tan
  p1 = p - axis_length * nor / 2
  p2 = p + axis_length * nor / 2
  plot_line_seg(p1, p2, color='#202020', lw=2)
  plot_dot(p, color='#202020', ms=center_size)

  for q in np.linspace(0, 1, nticks):
    p0 = p1 * q + p2 * (1 - q)
    p3 = p0 - tan * tick_length / 2
    p4 = p0 + tan * tick_length / 2
    plot_line_seg(p3, p4, color='#202020', lw=1)

def plot_line_with_ticks(p1, p2, nticks, lw=2, tick_length=0.1):
  plot_line_seg(p1, p2, color='#202020', lw=lw)
  t = (p2 - p1) / np.linalg.norm(p2 - p1)
  S = np.array([
    [0, -1],
    [1, 0]
  ])
  n = S @ t

  for q in np.linspace(0, 1, nticks):
    p0 = p1 * q + p2 * (1 - q)
    p3 = p0 - n * tick_length / 2
    p4 = p0 + n * tick_length / 2
    plot_line_seg(p3, p4, color='#202020', lw=1)  

def arrow_from_to(p1, p2, lw=2, color = '#202020'):
  arrowprops = dict(
    arrowstyle = "-|>",
    lw = lw,
    color = color,
    mutation_scale = 20
  )
  plt.annotate("", xytext = p1, xy = p2, arrowprops = arrowprops)

def distance_from_to(p1, p2, lw=2, color = '#202020'):
  arrowprops = dict(
    arrowstyle = "<|-|>",
    lw = lw,
    color = color,
    mutation_scale = 20
  )
  plt.annotate("", xytext = p1, xy = p2, arrowprops = arrowprops)

def arrow_at_point(p, v, **kwargs):
  return arrow_from_to(p, p + v, **kwargs)

def plot_transverse():
  _, x_sample = calc_oscillator_traj([-2.6, -3], 10, 5e-3, mu=1.01)
  t, x_ref = calc_oscillator_traj([2.02, 0], 10, 5e-3, mu=1.01)
  sp = make_interp_spline(t, x_ref, k=5)
  deriv = sp(t, 1)
  tan = deriv / np.linalg.norm(deriv, axis=1, keepdims=True)

  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  ax.set_aspect('equal')
  plt.axvline(0, ls='--', lw=1, color='#202020B0')
  plt.axhline(0, ls='--', lw=1, color='#202020B0')
  add_annotation(R'$x_1$', [0.9, 0.05], fontsize=22)
  add_annotation(R'$x_2$', [0.02, 0.9], fontsize=22)

  plt.plot(x_sample[:,0], x_sample[:,1], color='blue', lw=2)
  plt.plot(x_ref[:,0], x_ref[:,1], color='orange', lw=3)

  for i in [580]:
    plot_section_axis(x_ref[i], tan[i], axis_length=3.0)

  i = 11
  p_src = x_sample[i]
  plot_dot(p_src, ms=10, color='#2020A0')
  add_annotation(R'$x$', p_src + np.array([-0.36, +0.08]), fontsize=32, xycoords='data', bgalpha=0)
  
  i = 580
  p_ref = x_ref[i]
  tan_ref = tan[i]
  add_annotation(R'$x_\star(\pi(x))$', p_ref + np.array([0.2, -0.3]), fontsize=32, xycoords='data', bgalpha=0)
  add_annotation(R'$n(\pi(x))$', p_ref + np.array([0.45, 0.7]), fontsize=32, xycoords='data', bgalpha=0)

  arrow_from_to(p_ref - tan_ref * 0.15, p_src - tan_ref * 0.15, color = '#303030')
  add_annotation(R'$\xi$', p_ref + np.array([-0.5, -0.75]), fontsize=32, xycoords='data', bgalpha=0)

  plt.xlim(-3.5, 1.5)
  plt.ylim(-3.1, 1)
  plt.tight_layout(pad=0.1)
  plt.savefig('../media/projection-operator-2d.svg')
  plt.show()


def plot_transverse2():
  _, x_sample = calc_oscillator_traj([-2.2, 2.5], 5, 1e-3, mu=0.5)
  t, x_ref = calc_oscillator_traj([2.0088, 0], 8, 1e-3, mu=0.5)
  sp = make_interp_spline(t, x_ref, k=5)
  deriv = sp(t, 1)
  tan = deriv / np.linalg.norm(deriv, axis=1, keepdims=True)

  fig, axes = plt.subplots(1, 1, figsize=(5, 3.3))
  axes.set_aspect('equal')
  plt.plot(x_sample[:, 0], x_sample[:, 1], color='blue', lw=2)
  plt.plot(x_ref[:, 0], x_ref[:, 1], color='orange', lw=3)

  ipt = 500
  pt = x_sample[ipt]
  iref = find_closest_point(pt, x_ref)
  refpt = x_ref[iref]
  ref_tan = tan[iref]
  S = np.array([
    [0, -1],
    [1, 0]
  ])

  l = 1.8
  arrow_at_point(refpt, l * ref_tan, color='#202060')
  arrow_at_point(refpt, l * S @ ref_tan, color='#202060')
  plot_dot(pt, ms=5, color='#202080')
  plot_dot(refpt, ms=5, color='#202080')

  add_annotation(R'$v(\tau)$', refpt + l * ref_tan + np.array([-0.08, -0.35]), xycoords='data', fontsize=20, bgalpha=0)
  add_annotation(R'$e(\tau)$', refpt + l * S @ ref_tan + np.array([0.3, -0.1]), xycoords='data', fontsize=20, bgalpha=0)
  add_annotation(R'$x$', pt + np.array([+0.05, +0.2]), xycoords='data', fontsize=22, bgalpha=0)
  add_annotation(R'$x_\star(\pi(x))$', refpt + np.array([-0.1, -0.35]), xycoords='data', fontsize=22, bgalpha=0)

  plot_line_with_ticks(refpt, pt, 5, lw=1)
  add_annotation(R'$\xi$', refpt / 2 + pt / 2 + np.array([-0.3, -0.2]), xycoords='data', fontsize=22, bgalpha=0)

  plt.xlim(-2.5, 2.5)
  plt.ylim(0.5, 3.7)

  add_annotation(R'$x_1$', [0.55, -0.11], fontsize=22)
  add_annotation(R'$x_2$', [-0.10, 0.58], fontsize=22)
  plt.tight_layout(pad=0.5)
  plt.savefig('../media/projection-operator-2d.svg')


def transform_coords(x, t_ref, x_ref):
  i = find_closest_point(x, x_ref)
  tau = t_ref[i]
  xi = np.linalg.norm(x - x_ref[i])
  return tau, xi

def plot_traj_in_new_coords():
  period = 6.39
  _, x_sample = calc_oscillator_traj([3, 1], 8, 2e-3, mu = 0.5)
  t, x_ref = calc_oscillator_traj([2.0032, 0], period, 2e-3, mu = 0.5)
  tau, xi = np.array([transform_coords(x, t, x_ref) for x in x_sample]).T
  for i in range(1, len(tau)):
    while tau[i] < tau[i - 1]:
      tau[i] += period

  tau -= period

  plt.figure(figsize=(5, 4))
  plt.plot(tau, xi)
  add_annotation(R'$\tau$', [0.55, -0.10], fontsize=22)
  add_annotation(R'$\xi$', [-0.06, 0.62], fontsize=22)
  plt.axvline(0, ls='--', lw=1, color='#202020')
  plt.axvline(period, ls='--', lw=1, color='#202020')
  plt.axhline(0, ls='--', lw=1, color='#202020')
  ticks = np.linspace(0, period, 5)
  plt.xticks(ticks)
  plt.grid(True)
  plt.tight_layout(pad=0.5)
  plt.savefig('../media/vanderpol-transverse.svg')

  fig, ax = plt.subplots(1, 1, figsize=(5.7, 5))
  ax.set_aspect('equal')
  plt.grid(True)

  plt.plot(x_sample[:,0], x_sample[:,1], color='blue', label=R'$x_s(t)$')
  plt.plot(x_ref[:,0], x_ref[:,1], color='orange', lw=3, label=R'$x_\star(t)$')

  for i, z in zip([400, 1000, 1600], [1.2, 2.7, 3.1]):
    v = x_ref[i + 1] - x_ref[i]
    v = v / np.linalg.norm(v)
    plot_section_axis(x_ref[i], v, axis_length=1.0)
    add_annotation(R'$\tau = ' + str(z) + '$', x_ref[i] + np.array([0.1, 0.4]), fontsize=18, xycoords='data', bgalpha=0)

  add_annotation(R'$x_1$', [0.45, -0.08], fontsize=22)
  add_annotation(R'$x_2$', [-0.10, 0.58], fontsize=22)
  plt.legend(fontsize=18)
  plt.tight_layout(pad=0.5)
  plt.savefig('../media/vanderpol-phase.svg')

if __name__ == '__main__':
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  # plot_sample()
  # plot_orbit_deviation()
  plot_transverse2()
  # plot_traj_in_new_coords()
  plt.show()