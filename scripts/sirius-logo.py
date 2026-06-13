import drawsvg as draw
import numpy as np

scale = 10
d = draw.Drawing(800, 800, id_prefix='contour_demo')

stars = [
  [ 11.4, 3.4, 5.4 ],
  [ 21.1, 2.1, 4.5 ],
  [ 5.1, 10.3, 4.5 ],
  [ 12.5, 10.9, 4.5 ],
  [ 18.5, 7.9, 4.5 ],
  [ 3.1, 19.6, 4.0 ],
  [ 9.9, 16.4, 4.0 ],
  [ 18.5, 14.3, 4.0 ],
  [ 24.6, 8.1, 4.0 ],
  [ 30.3, 4.1, 4.0 ],
  [ 5.9, 28.9, 3.5 ],
  [ 10.1, 22.9, 3.5 ],
  [ 15.8, 19.9, 3.5 ],
  [ 21.7, 19.9, 3.5 ],
  [ 24.7, 14.6, 3.5 ],
  [ 30.4, 11.4, 3.5 ],
  [ 37.5, 10.8, 3.5 ],
  [ 13.7, 28.6, 2.9 ],
  [ 19.1, 25.5, 2.9 ],
  [ 25.3, 25.6, 2.5 ],
  [ 28.1, 20.2, 2.9 ],
  [ 33.4, 16.9, 3.1 ],
  [ 12.6, 35.8, 2.9 ],
  [ 19.0, 31.9, 2.5 ],
  [ 33.7, 23.4, 2.5 ],
  [ 39.8, 20.1, 3.1 ],
  [ 25.6, 32.2, 2.0 ],
  [ 30.8, 29.1, 2.0 ],
  [ 22.2, 38.4, 2.5 ],
  [ 37.7, 29.4, 2.5 ],
  [ 31.1, 36.1, 2.0 ],
]

nstars = len(stars)

for (i, (x, y, w)) in enumerate(stars):
  circle = draw.Circle(scale * x, scale * y, scale * w / 2, opacity=0.6, fill='#0e9da0ff')
  # circle.append_anim(draw.AnimateTransform('rotate', '1s', '1,100,10;20,100,10;60,100,10;90,100,10', repeatCount='indefinite'))
  # circle.append_anim(draw.AnimateTransform('translate', '1s', '10, 20; 100, 1;', repeatCount='indefinite'))
  # circle.append_anim(draw.AnimateMotion(traj, '1s', repeatCount='indefinite'))

  w = np.linspace(0, 2, 20)
  w = 2 * np.pi * np.clip(w - i / nstars, 0, 1)
  pts = np.array([40 * np.sin(w), 40 - 40 * np.cos(w)]).T
  s = ''
  for (x, y) in pts:
    s += f'{x},{y};'
    
  circle.append_anim(draw.AnimateTransform('translate', '5s', s, repeatCount='indefinite'))
  circle.append_anim(draw.Animate('opacity', '5s', '0;0.8;0.8;0.8;0;', repeatCount='indefinite'))

  d.append(circle)

d.save_svg('sirius-logo.svg')
print("✅ created: sirius-logo.svg")
