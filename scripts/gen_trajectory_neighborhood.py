import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def make_curve(points):
  difs = np.diff(points, axis=0)
  l = np.cumsum(np.linalg.norm(difs, axis=1))
  l = np.insert(l, 0, 0)
  curve = make_interp_spline(l, points)
  return l, curve

def get_curve_tangent(curve, s):
  deriv = curve(s, 1)
  return deriv / np.linalg.norm(deriv)

def gen_points():
  t = np.linspace(0, 2*np.pi, 300)
  points = np.array([t, np.sin(t)]).T
  return points

points = gen_points()
l, curve = make_curve(points)
knots = np.linspace(l[0], l[-1], 21)

plt.figure('trajectory-neighborhood', figsize=(6, 4))
plt.axis('equal')
plt.axis('off') 

S = np.array([
  [0, -1],
  [1, 0]
])

for s in knots:
  p = curve(s)
  v = S @ get_curve_tangent(curve, s)
  p1 = p + 2 * v
  p2 = p - 2 * v
  plt.plot([p1[0], p2[0]], [p1[1], p2[1]], lw=2, color='#B06060')

  p1 = p + 0.8 * v
  p2 = p - 0.8 * v
  plt.plot([p1[0], p2[0]], [p1[1], p2[1]], lw=3, color='#60B060')
  plt.plot(*p, 'o', color='#202020')

plt.plot(points[:,0], points[:,1], lw=4, color='#202020')
plt.tight_layout(pad=0, h_pad=0)
plt.savefig('../media/trajectory-neighborhood.svg')
plt.show()