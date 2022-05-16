import numpy as np
import matplotlib.pyplot as plt
from cairosvg import svg2png
from PIL import Image, ImageFilter
from io import BytesIO
from glob import glob

def gen_color(brightness, red_rate=0.92, blue_rate=0.8, rate_deviation=0.15):
  g = brightness
  r = brightness * np.random.normal(red_rate, rate_deviation)
  b = brightness * np.random.normal(blue_rate, rate_deviation)
  return min(255,max(0,round(r))), min(255,max(0,round(g))), min(255,max(0,round(b)))

def variate_color(color, deviation=5):
  return tuple(color + np.random.normal(0, deviation, 3))

def generate_svg(W = 64, H = 64):
  T = 4
  R = 20
  x_payload, y_payload = np.random.normal(W/2, 2*W/3), np.random.normal(H/2, 0.75*H/4)
  x, y = np.random.uniform(0, W, 2)
  x_rel, y_rel = x - x_payload, y - y_payload
  scale_x, scale_y = np.random.normal(0.95, 0.15, 2)
  overall_brightness = np.random.uniform(10, 230)
  dark_multiplicative = (np.tanh(np.random.normal(2))+1)/2
  color_dark = gen_color(overall_brightness*dark_multiplicative)
  color_bright = gen_color(overall_brightness+30)
  color_bright2 = variate_color(color_bright, 5)
  color_bright3 = variate_color(color_bright, 15)
  color_payload = gen_color(overall_brightness*dark_multiplicative*np.random.normal(1.02,0.05))
  rotation1 = np.random.uniform(-90, 90)
  rotation2 = np.random.uniform(-90, 90)
  radius_glare = np.random.lognormal()*(overall_brightness/80)
  x_text_rel, y_text_rel = np.random.normal(0, W/2, 2)
  confidence = 1
  dist_lim = min(x, W-x, y, H - y,
                (2*W)**2 - (x_rel)**2 - (y_rel*2*W/0.75/H)**2)
  if dist_lim < 0:
    confidence = 0
  elif dist_lim < T:
    confidence = 1 - (T - dist_lim) / T
  svg_base_code = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="#ffffff" stroke="none">
  <defs>
    <clipPath id="payload">
      <ellipse cx="0" cy="0" rx="{2*W}" ry="{0.75*H}"/>
    </clipPath>
    <radialGradient id="grad_radial_light">
      <stop offset="0%" stop-color="rgb{color_bright}" stop-opacity="1"/>
      <stop offset="100%" stop-color="rgb{color_bright}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="grad_linear_light" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>
  <g transform="translate({x_payload}, {y_payload})" clip-path="url(#payload)">
    <ellipse cx="0" cy="0" rx="{2*W}" ry="{0.75*H}" fill="rgb{color_payload}"/>
    <g transform="translate({x_text_rel},{y_text_rel}), skewX({np.random.normal(0, 10)})">
      <rect x="0" y="0" width="20" height="15" fill="rgb{color_bright3}"/>
      <text x="1" y="12" font-family="liberation sans, sans-serif" font-size="11px" fill="black">C{np.random.random_integers(1, 17)}</text>
    </g>
    <g transform="translate({x_rel}, {y_rel}), rotate({rotation2}), scale({scale_x}, {scale_y}), rotate({rotation1})">
      <path style="fill:rgb{color_bright};"
        d="M -20,0 a 20,20 0 0 1 20,-20 V 0 Z" />
      <path style="fill:rgb{color_bright2};"
        d="M 20,0 a 20,20 0 0 1 -20,20 V 0 Z" />
      <path style="fill:rgb{color_dark};"
        d="M -20,0 a 20,20 0 0 0 20,20 V 0 Z" />
      <path style="fill:rgb{color_dark};"
        d="M 20,0 a 20,20 0 0 0 -20,-20 V 0 Z"/>
      <circle cx="0" cy="0" r="{radius_glare}" fill="url(#grad_radial_light)"/>
    </g>
  </g>
  <rect x="0" y="0" width="{W}" height="{H}" fill="url(#grad_linear_light)"/>
</svg>
"""
  return svg_base_code, (x, y, confidence)

def generate_dataset(quant, H=64, W=64):
  positions = np.zeros((quant, 3), dtype=np.float32)
  imgs = np.zeros((quant, 1, H, W), dtype=np.float32)
  for i in range(quant):
    svg, pos = generate_svg()
    out = BytesIO()
    svg2png(bytestring=svg, write_to=out)
    with Image.open(out) as img_png:
        img = np.array(img_png.filter(ImageFilter.GaussianBlur(radius = min(2,np.random.lognormal(-0.6, 0.5)))))
    imgs[i, 0, ::2, ::2] = img[::2, ::2, 1]
    imgs[i, 0, ::2, 1::2] = img[::2, ::2, 0]
    imgs[i, 0, 1::2, ::2] = img[::2, ::2, 2]
    imgs[i, 0, 1::2, 1::2] = img[1::2, 1::2, 1]
    positions[i, :] = pos
    out.close()
    if i%1000==0:
      print("Generating ", i)
  negative = np.sum(positions[:,2]==0)
  fuzzy = np.sum(positions[:,2]<1) - negative
  positive = positions.shape[0] - fuzzy - negative
  imgs_negat_big = []
  for file in glob("dados_treinamento/negativo/*.png"):
    with Image.open(file) as img_png:
      imgs_negat_big.append(np.array(img_png.getchannel(0)))
  size = [img.size for img in imgs_negat_big]
  imgs_negat = np.zeros((positive, 1, H, W))
  choices = np.random.choice(len(size), positive, p=size/np.sum(size))
  for i in range(imgs_negat.shape[0]):
    img = imgs_negat_big[choices[i]]
    y, x = np.random.randint(0, img.shape[0]-H), np.random.randint(0, img.shape[1]-W)
    y, x = y - y%2, x - x%2
    imgs_negat[i, 0, :, :] = img[y:y+H, x:x+W]
  imgs = np.concatenate([imgs, imgs_negat])
  positions = np.concatenate([positions, np.zeros((imgs_negat.shape[0],3))])
  permutation = np.random.permutation(imgs.shape[0])
  imgs = imgs[permutation]
  positions = positions[permutation]
  return imgs, positions

def linear_demosaic(img):
  d = np.zeros((img.shape[0]+2, img.shape[1]+2, 3), dtype=np.float_)
  d[1:-1:2, 2::2, 0] = img[::2, 1::2]
  d[1:-1:2, 1:-1:2, 1] = img[::2, ::2]
  d[2::2, 2::2, 1] = img[1::2, 1::2]
  d[2::2, 1:-1:2, 2] = img[1::2, ::2]
  d[0, :, :] = d[2, :, :]
  d[-1, :, :] = d[-3, :, :]
  d[:, 0, :] = d[:, 2, :]
  d[:, -1, :] = d[:, -3, :]
  d[1:-1:2, 1:-1:2, 0] = (d[1:-1:2, :-2:2, 0]+d[1:-1:2, 2::2, 0])/2
  d[2::2, 2::2, 0] = (d[1:-1:2, 2::2, 0]+d[3::2, 2::2, 0])/2
  d[2::2, 1:-1:2, 0] = (d[1:-1:2, :-2:2, 0]+d[1:-1:2, 2:-1:2, 0]+d[3::2, 2::2, 0]+d[3::2, :-2:2, 0])/4
  d[1:-1:2, 2::2, 1] = (d[:-2:2, 2::2, 1]+d[2::2, 2::2, 1]+d[1:-1:2, 1:-1:2, 1]+d[1:-1:2, 3::2, 1])/4
  d[2::2, 1:-1:2, 1] = (d[1:-1:2, 1:-1:2, 1]+d[3::2, 1:-1:2, 1]+d[2::2, :-2:2, 1]+d[2::2, 2::2, 1])/4
  d[1:-1:2, 2::2, 2] = (d[:-2:2, 1:-1:2, 2]+d[:-2:2, 3::2, 2]+d[2::2, 3::2, 2]+d[2::2, 1:-1:2, 2])/4
  d[1:-1:2, 1:-1:2, 2] = (d[:-2:2, 1:-1:2, 2]+d[2::2, 1:-1:2, 2])/2
  d[2::2, 2::2, 2] = (d[2::2, 1:-1:2, 2]+d[2::2, 3::2, 2])/2
  return d[1:-1, 1:-1, :].astype(img.dtype)
