from libs_contrib import EarlyStopping
import numpy as np
import cairosvg
import io
import os
import torch
import PIL.Image
import PIL.ImageFilter
import glob
import cv2

def svg_gen(W, H, pos_pad = 4, pos=None, scale=None, rot=None, colors=None):
    if pos is None:
        pos = [np.random.uniform(pos_pad, W-pos_pad), np.random.uniform(pos_pad, H-pos_pad)]
    if scale is None:
        scale = np.random.uniform(8, 30, size=2)
    if rot is None:
        rot = np.random.uniform(180, size=2)
    if colors is None:
        colors = [[np.random.uniform(90)]*3]
        colors.append([colors[0][0]+np.random.uniform(60,165)]*3)
        colors.append([colors[0][0]+np.random.uniform(0,60)]*3)
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" stroke="none">
<rect width="100%" height="100%" style="fill:rgb{tuple(colors[2])};" />
<g transform="translate({pos[0]}, {pos[1]}), rotate({rot[1]}), scale({scale[0]}, {scale[1]}), rotate({rot[0]})">
    <path style="fill:rgb{tuple(colors[0])};" d="M -1,0 a 1,1 0 0 1 1,-1 V 0 Z" />
    <path style="fill:rgb{tuple(colors[0])};" d="M 1,0 a 1,1 0 0 1 -1,1 V 0 Z" />
    <path style="fill:rgb{tuple(colors[1])};" d="M -1,0 a 1,1 0 0 0 1,1 V 0 Z" />
    <path style="fill:rgb{tuple(colors[1])};" d="M 1,0 a 1,1 0 0 0 -1,-1 V 0 Z"/>
</g>
</svg>"""
    out = io.BytesIO()
    cairosvg.svg2png(bytestring=svg, write_to=out)
    with PIL.Image.open(out) as img_png:
        img = np.array(img_png)
    return img, pos

def gen_dataset(n=96000, W=24, H=24, pos_pad = 4):
    imgs = np.zeros((n, 1, H, W))
    pos = np.zeros((n, 2))
    for i in range(n):
        im, p = svg_gen(W, H, pos_pad=pos_pad)
        imgs[i, 0] = im[:, :, 0]
        pos[i] = p
        if (i+1) % 10000 == 0:
            print(i+1, end=', ')
    return imgs, pos

def load_datafiles(path='saved_data'):
    with open(os.path.join(path, 'imgs.npy'), 'rb') as f:
        imgs = np.load(f).astype(np.float32)
    with open(os.path.join(path, 'positions.npy'), 'rb') as f:
        pos = np.load(f)
    return imgs, pos

def save_datafiles(imgs, pos, path='saved_data'):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'imgs.npy'), 'wb') as f:
        np.save(f, imgs.astype(np.uint8))
    with open(os.path.join(path, 'positions.npy'), 'wb') as f:
        np.save(f, pos.astype(np.float32))

def evaluate(model, loss_fn, dataloader):
    outputs = []
    val_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            outputs.append(output.cpu().numpy())
            val_loss += loss_fn(output, y).sum().item()
    val_loss /= len(dataloader)
    return np.vstack(outputs), val_loss

def train(model, criterion, optimizer, train_dataloader, val_dataloader=None, epochs=20, tensorboard_writer=None, tb_offset=0, path='checkpoint.pt',patience=10):
    train_losses, val_losses = [], []
    early_stopper = EarlyStopping(patience=patience, path=path, trace_func=lambda x: x)
    for epoch in range(epochs):
        train_loss = 0
        print('%d,' % epoch, end=' ')
        for x, y in train_dataloader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        _, val_loss = evaluate(model, criterion, val_dataloader)
        if tensorboard_writer:
            tensorboard_writer.add_scalar('Training Loss', train_loss, epoch + tb_offset + 1)
            tensorboard_writer.add_scalar('Validation Loss', val_loss, epoch + tb_offset + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            break
    model.load_state_dict(torch.load(path))
    return train_losses, val_losses

def calc_error(pred, pos):
    return np.linalg.norm(pred-pos, axis=1).mean()

def negative_samples(quant, H=24, W=24, images_glob_path='neg_imgs/*'):
    imgs = np.zeros((quant, 1, H, W), dtype=np.float32)
    imgs_negat_big = []
    for file in glob.glob(images_glob_path):
        with PIL.Image.open(file) as img_png:
            imgs_negat_big.append(np.array(img_png.convert('L')))
    size = [img.size for img in imgs_negat_big]
    choices = np.random.choice(len(size), quant, p=size/np.sum(size))
    for i in range(imgs.shape[0]):
        img = imgs_negat_big[choices[i]]
        y, x = np.random.randint(0, img.shape[0]-H), np.random.randint(0, img.shape[1]-W)
        imgs[i, 0, :, :] = img[y:y+H, x:x+W]
    np.random.shuffle(imgs)
    return imgs.astype(np.float32)/255

def incorporate_dataset(neg_imgs, dataset, batch_size=32):
    neg_pos = np.tile(np.array([0.5, 0.5, 0]), (len(neg_imgs), 1))
    device = dataset.tensors[0].device
    imgs = dataset.tensors[0].cpu().numpy()
    pos = dataset.tensors[1].cpu().numpy()
    tot_imgs = np.vstack([imgs, neg_imgs])
    tot_pos = np.vstack([np.hstack([pos, np.ones((pos.shape[0], 1))]), neg_pos])
    permutation = np.random.permutation(tot_imgs.shape[0])
    tot_imgs = tot_imgs[permutation]
    tot_pos = tot_pos[permutation]
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.Tensor(tot_imgs).to(device), torch.Tensor(tot_pos).to(device)), batch_size=batch_size)

def gen_color(brightness, red_rate=0.92, blue_rate=0.8, rate_deviation=0.15):
    g = brightness
    r = brightness * np.random.normal(red_rate, rate_deviation)
    b = brightness * np.random.normal(blue_rate, rate_deviation)
    return min(255,max(0,round(r))), min(255,max(0,round(g))), min(255,max(0,round(b)))

def variate_color(color, deviation=5):
    return tuple(color + np.random.normal(0, deviation, 3))

def generate_svg_harder(W = 64, H = 64):
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
            <text x="1" y="12" font-family="liberation sans, sans-serif" font-size="11px" fill="black">C{np.random.randint(1, 17)}</text>
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

def generate_svg_medium(W = 24, H = 24, pad=5):
    R = 20
    x, y = np.random.uniform(pad, W-pad), np.random.uniform(pad, H-pad)
    scale_x, scale_y = np.clip(np.random.normal(0.95, 0.25, 2), 0.2, 2)
    overall_brightness = np.random.uniform(10, 200)
    dark_multiplicative = (np.tanh(np.random.normal(2))+1)/2
    color_dark = gen_color(overall_brightness*dark_multiplicative)
    color_bright = gen_color(overall_brightness+30)
    color_bright2 = variate_color(color_bright, 5)
    color_bright3 = variate_color(color_bright, 15)
    color_payload = gen_color(overall_brightness*dark_multiplicative*np.random.normal(1.02,0.05))
    rotation1 = np.random.uniform(-90, 90)
    rotation2 = np.random.uniform(-90, 90)
    confidence = 1
    svg_base_code = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="#ffffff" stroke="none">
    <defs>
        <linearGradient id="grad_linear_light" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stop-color="#ffffff" stop-opacity="{np.random.uniform(0, 0.6)}"/>
            <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
        </linearGradient>
    </defs>
    <rect x="0" y="0" width="{W}" height="{H}" fill="rgb{color_payload}"/>
    <g transform="translate({x}, {y}), rotate({rotation2}), scale({scale_x}, {scale_y}), rotate({rotation1})">
        <path style="fill:rgb{color_bright};" d="M -20,0 a 20,20 0 0 1 20,-20 V 0 Z" />
        <path style="fill:rgb{color_bright2};" d="M 20,0 a 20,20 0 0 1 -20,20 V 0 Z" />
        <path style="fill:rgb{color_dark};" d="M -20,0 a 20,20 0 0 0 20,20 V 0 Z" />
        <path style="fill:rgb{color_dark};" d="M 20,0 a 20,20 0 0 0 -20,-20 V 0 Z"/>
    </g>
    <g transform="translate({W}, {H}),rotate({np.random.uniform(-90, 90)}))">
        <rect x="{-W}" y="{-H}" width="{2*W}" height="{2*H}" fill="url(#grad_linear_light)"/>
    </g>
</svg>
"""
    return svg_base_code, (x, y, confidence)

def generate_dataset_harder(quant, svg_gen, H=64, W=64):
    positions = np.zeros((quant, 3), dtype=np.float32)
    imgs = np.zeros((quant, 1, H, W), dtype=np.float32)
    for i in range(quant):
        svg, pos = svg_gen()
        out = io.BytesIO()
        cairosvg.svg2png(bytestring=svg, write_to=out)
        with PIL.Image.open(out) as img_png:
                img = np.array(img_png.filter(PIL.ImageFilter.GaussianBlur(radius = min(3,np.random.lognormal(-0.62, 0.7)))))
        noise = np.clip(np.random.normal(17, 9), 0, 30)
        img = np.clip(np.random.normal(0, noise, img.shape) + img, 0, 255).astype(np.uint8)
        imgs[i, 0, ::2, ::2] = img[::2, ::2, 1]
        imgs[i, 0, ::2, 1::2] = img[::2, ::2, 0]
        imgs[i, 0, 1::2, ::2] = img[::2, ::2, 2]
        imgs[i, 0, 1::2, 1::2] = img[1::2, 1::2, 1]
        positions[i, :] = pos
        out.close()
        if i%1000==0:
            print(i//1000, end=', ')
    negative = np.sum(positions[:,2]==0)
    fuzzy = np.sum(positions[:,2]<1) - negative
    positive = positions.shape[0] - fuzzy - negative
    imgs_negat_big = []
    for file in glob.glob("dados_treinamento/negativo2/*"):
        with PIL.Image.open(file) as img_png:
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

def generate_svg_easy(W, H, pos_pad = 4):
    pos = [np.random.uniform(pos_pad, W-pos_pad), np.random.uniform(pos_pad, H-pos_pad), 1]
    scale = np.random.uniform(12, 28, size=2)
    rot = np.random.uniform(180, size=2)
    colors = [[np.random.uniform(90)]*3]
    colors.append([colors[0][0]+np.random.uniform(60,165)]*3)
    colors.append([colors[0][0]+np.random.uniform(0,60)]*3)
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" stroke="none">
<rect width="100%" height="100%" style="fill:rgb{tuple(colors[2])};" />
<g transform="translate({pos[0]}, {pos[1]}), rotate({rot[1]}), scale({scale[0]}, {scale[1]}), rotate({rot[0]})">
    <path style="fill:rgb{tuple(colors[0])};" d="M -1,0 a 1,1 0 0 1 1,-1 V 0 Z" />
    <path style="fill:rgb{tuple(colors[0])};" d="M 1,0 a 1,1 0 0 1 -1,1 V 0 Z" />
    <path style="fill:rgb{tuple(colors[1])};" d="M -1,0 a 1,1 0 0 0 1,1 V 0 Z" />
    <path style="fill:rgb{tuple(colors[1])};" d="M 1,0 a 1,1 0 0 0 -1,-1 V 0 Z"/>
</g>
</svg>"""
    return svg, pos

def generate_dataset_real(quant, real_data, svg_gen, H=64, W=64):
    positions = np.zeros((quant, 3), dtype=np.float32)
    imgs = np.zeros((quant, 1, H, W), dtype=np.float32)
    for i in range(quant):
        svg, pos = svg_gen()
        out = io.BytesIO()
        cairosvg.svg2png(bytestring=svg, write_to=out)
        with PIL.Image.open(out) as img_png:
                img = np.array(img_png.filter(PIL.ImageFilter.GaussianBlur(radius = min(3,np.random.lognormal(-0.62, 0.7)))))
        noise = np.clip(np.random.normal(17, 9), 0, 50)
        img = np.clip(np.random.normal(0, noise, img.shape) + img, 0, 255).astype(np.uint8)
        imgs[i, 0, ::2, ::2] = img[::2, ::2, 1]
        imgs[i, 0, ::2, 1::2] = img[::2, ::2, 0]
        imgs[i, 0, 1::2, ::2] = img[::2, ::2, 2]
        imgs[i, 0, 1::2, 1::2] = img[1::2, 1::2, 1]
        positions[i, :] = pos
        out.close()
        if i%1000==0:
            print(i//1000, end=', ')
    negative = np.sum(positions[:,2]==0)
    fuzzy = np.sum(positions[:,2]<1) - negative
    positive = positions.shape[0] - fuzzy - negative
    imgs_negat_big = []
    for file in glob.glob("dados_treinamento/negativo2/*"):
        with PIL.Image.open(file) as img_png:
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

def gen_gauss(n, size=(24, 24), margin=4, intensity=0.15, sigma=(4, 20)):
    x = np.tile(np.arange(size[0]).reshape(1, 1, 1, -1), (n, 1, size[1], 1))
    y = np.tile(np.arange(size[1]).reshape(1, 1, -1, 1), (n, 1, 1, size[0]))
    x0 = np.random.uniform(-margin, size[0]+margin, size=(n, 1, 1, 1))
    y0 = np.random.uniform(-margin, size[1]+margin, size=(n, 1, 1, 1))
    I = np.random.uniform(-intensity, intensity, size=(n, 1, 1, 1))
    Var = np.random.uniform(sigma[0], sigma[1], size=(n, 1, 1, 1))**2
    a = np.random.uniform(0.5, 1.5, size=(n, 1, 1, 1))
    b = np.random.uniform(-1.25, 1.25, size=(n, 1, 1, 1))
    c = np.random.uniform(0.5, 1.5, size=(n, 1, 1, 1))
    return np.clip(I*np.exp(-(a*(x-x0)**2+b*(x-x0)*(y-y0)+c*(y-y0)**2)/Var), -intensity, intensity)

def evaluate_dyn(model, loss_fn, dataloader, device):
    outputs = []
    val_loss = 0
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            outputs.append(output.cpu().numpy())
            val_loss += loss_fn(output, y).sum().item()
    val_loss /= len(dataloader)
    return np.vstack(outputs), val_loss

def test_dyn(model, dataloader, device):
    outputs = []
    with torch.no_grad():
        model.eval()
        for x, in dataloader:
            x = x.to(device)
            output = model(x)
            outputs.append(output.cpu().numpy())
    return np.vstack(outputs)

def train_dyn(model, criterion, optimizer, train_dataloader, device, val_dataloader=None, epochs=20, tensorboard_writer=None, tb_offset=0, path='checkpoint.pt'):
    train_losses, val_losses = [], []
    early_stopper = EarlyStopping(patience=10, path=path)
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        _, val_loss = evaluate_dyn(model, criterion, val_dataloader, device)
        if tensorboard_writer:
            tensorboard_writer.add_scalar('Training Loss', train_loss, epoch + tb_offset + 1)
            tensorboard_writer.add_scalar('Validation Loss', val_loss, epoch + tb_offset + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            break
    model.load_state_dict(torch.load(path))
    return train_losses, val_losses

def getHoughIntersection(img):
    dst = cv2.Canny(img, 50, 200, None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 17, None, 0, 0)
    if lines is not None and len(lines) > 1:
        A = np.hstack([np.cos(lines[:2, 0, 1:2]), np.sin(lines[:2, 0, 1:2])])
        try:
            x0, y0 = np.linalg.solve(A, lines[:2, 0, 0:1])
            return x0[0], y0[0], 0.999
        except:
            pass
    return img.shape[1]//2, img.shape[0]//2, 0.001
