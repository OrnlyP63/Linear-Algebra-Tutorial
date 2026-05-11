"""Generate 1280x720 YouTube thumbnail covers for Linear Algebra for AI series."""

import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── palette ──────────────────────────────────────────────────────────────────
BG       = (13, 17, 23)        # #0D1117
CYAN     = (0, 255, 255)       # #00FFFF
PURPLE   = (155, 89, 182)      # #9B59B6
GREEN    = (0, 255, 136)       # #00FF88
GOLD     = (255, 215, 0)       # #FFD700
WHITE    = (255, 255, 255)
DIM      = (120, 130, 145)

W, H = 1280, 720
VIZ_X = 704   # right panel starts here

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
LOGO_SIZE  = 70

# ── font helpers ─────────────────────────────────────────────────────────────
def _font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _font_thai(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/SukhumvitSet.ttc",
        "/System/Library/Fonts/Supplemental/Ayuthaya.ttf",
        "/System/Library/Fonts/Supplemental/Krungthep.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── background ───────────────────────────────────────────────────────────────
def make_bg():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img, "RGBA")
    # subtle dot grid
    for x in range(0, W, 40):
        for y in range(0, H, 40):
            draw.ellipse([x-1, y-1, x+1, y+1], fill=(255, 255, 255, 13))
    # right-panel separator line
    draw.line([(VIZ_X, 0), (VIZ_X, H)], fill=(*CYAN, 60), width=1)
    return img


# ── glow helper ──────────────────────────────────────────────────────────────
def paste_with_glow(base, overlay_rgba, pos=(0, 0), glow_radius=12, glow_alpha=0.35):
    """Paste RGBA overlay with a blur-glow behind it."""
    glow = overlay_rgba.filter(ImageFilter.GaussianBlur(glow_radius))
    glow_arr = np.array(glow).astype(float)
    glow_arr[..., 3] *= glow_alpha
    glow_img = Image.fromarray(glow_arr.astype(np.uint8), "RGBA")
    base.paste(glow_img, pos, glow_img)
    base.paste(overlay_rgba, pos, overlay_rgba)


# ── text block ───────────────────────────────────────────────────────────────
def draw_text_block(img, chapter_num, title, hook, thai_hook, accent):
    draw = ImageDraw.Draw(img, "RGBA")

    # chapter badge
    badge_font  = _font(22, bold=True)
    badge_text  = f"CHAPTER {chapter_num}"
    bx, by = 60, 54
    bbox = draw.textbbox((bx, by), badge_text, font=badge_font)
    pad = 10
    draw.rounded_rectangle(
        [bbox[0]-pad, bbox[1]-6, bbox[2]+pad, bbox[3]+6],
        radius=14, fill=(*PURPLE, 200)
    )
    draw.text((bx, by), badge_text, font=badge_font, fill=WHITE)

    # title
    title_font = _font(108, bold=True)
    lines = title.split("\n")
    ty = 140
    for line in lines:
        draw.text((60, ty), line, font=title_font, fill=WHITE)
        ty += 118

    # accent divider
    dy = ty + 14
    draw.rectangle([60, dy, 60 + 420, dy + 4], fill=accent)

    # English hook
    hook_font = _font(38, bold=False)
    draw.text((60, dy + 24), hook, font=hook_font, fill=accent)

    # Thai hook
    thai_font = _font_thai(28)
    accent_dim = (*accent[:3], 200)
    draw.text((60, dy + 76), thai_hook, font=thai_font, fill=accent_dim)

    # logo + series tag bottom-left
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH).convert("RGBA").resize(
            (LOGO_SIZE, LOGO_SIZE), Image.LANCZOS
        )
        logo_y = H - LOGO_SIZE - 16
        img.paste(logo, (60, logo_y), logo)

    series_font = _font(26)
    series_x = 60 + LOGO_SIZE + 12
    series_y = H - LOGO_SIZE - 16 + (LOGO_SIZE - 26) // 2
    draw.text((series_x, series_y), "Linear Algebra for AI", font=series_font, fill=DIM)


# ── matplotlib viz → PIL RGBA ─────────────────────────────────────────────────
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=150,
                bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _ax_style(ax, bg=BG):
    ax.set_facecolor(tuple(c/255 for c in bg))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# ── per-chapter visualizations ───────────────────────────────────────────────
def viz_ch1():
    fig, ax = plt.subplots(figsize=(5.76, 7.2))
    fig.patch.set_facecolor(tuple(c/255 for c in BG))
    _ax_style(ax)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)

    vecs = [(0,0,2,1,"#00FFFF"), (0,0,1,2,"#9B59B6"), (0,0,3,3,"#00FF88")]
    for x,y,dx,dy,col in vecs:
        ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=2.5, mutation_scale=22))
    # labels
    for (x,y,dx,dy,col), lbl in zip(vecs, ["v₁", "v₂", "v₁+v₂"]):
        ax.text(x+dx+0.08, y+dy+0.08, lbl, color=col, fontsize=18, fontweight="bold")

    # dashed parallelogram
    ax.plot([2,3], [1,3], "--", color="#00FF88", alpha=0.4, lw=1.5)
    ax.plot([1,3], [2,3], "--", color="#00FF88", alpha=0.4, lw=1.5)
    return fig


def viz_ch2():
    fig, ax = plt.subplots(figsize=(5.76, 7.2))
    fig.patch.set_facecolor(tuple(c/255 for c in BG))
    _ax_style(ax)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)

    # original grid (faint)
    for i in np.arange(-2, 2.1, 1):
        ax.axhline(i, color="#9B59B6", alpha=0.2, lw=0.8)
        ax.axvline(i, color="#9B59B6", alpha=0.2, lw=0.8)

    # transformation matrix: shear
    M = np.array([[1, 0.8], [0.3, 1]])
    pts = np.array([[x, y] for x in np.arange(-2, 2.1, 0.5)
                             for y in np.arange(-2, 2.1, 0.5)]).T
    tp = M @ pts
    ax.scatter(pts[0], pts[1], color="#9B59B6", s=6, alpha=0.4)
    ax.scatter(tp[0], tp[1], color="#00FFFF", s=12, alpha=0.9)

    # draw arrows from original to transformed for a 3×3 subset
    sub = np.array([[x, y] for x in [-1, 0, 1] for y in [-1, 0, 1]]).T
    st = M @ sub
    for i in range(sub.shape[1]):
        ax.annotate("", xy=(st[0,i], st[1,i]), xytext=(sub[0,i], sub[1,i]),
                    arrowprops=dict(arrowstyle="-|>", color="#00FFFF",
                                   alpha=0.7, lw=1.5, mutation_scale=14))
    ax.text(0, 2.0, "Av", color="#00FFFF", fontsize=22, ha="center", fontweight="bold")
    return fig


def viz_ch3():
    fig, ax = plt.subplots(figsize=(5.76, 7.2))
    fig.patch.set_facecolor(tuple(c/255 for c in BG))
    _ax_style(ax)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)

    x = np.linspace(-3, 3, 300)
    # line 1: y = x + 1
    ax.plot(x, x + 1, color="#00FFFF", lw=2.5)
    # line 2: y = -0.5x + 0.5  →  intersection at x=1/3, y=4/3
    ax.plot(x, -0.5*x + 0.5, color="#9B59B6", lw=2.5)
    # line 3
    ax.plot(x, 2*x - 1.5, color="#00FF88", lw=2.5, alpha=0.7)

    ix, iy = 1/3, 4/3
    # glow dot at intersection
    for r, a in [(30, 0.15), (15, 0.25), (6, 0.9)]:
        ax.scatter([ix], [iy], s=r*r, color="#FFD700", alpha=a, zorder=5)
    ax.text(ix+0.15, iy+0.25, "x*", color="#FFD700", fontsize=22, fontweight="bold")

    ax.axhline(0, color=(*[c/255 for c in WHITE], 0.1), lw=0.8)
    ax.axvline(0, color=(*[c/255 for c in WHITE], 0.1), lw=0.8)
    return fig


def viz_ch4():
    fig = plt.figure(figsize=(5.76, 7.2))
    fig.patch.set_facecolor(tuple(c/255 for c in BG))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(tuple(c/255 for c in BG))
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor((1,1,1,0.05))
    ax.tick_params(colors=(0,0,0,0))
    ax.set_xlim(0,2); ax.set_ylim(0,2); ax.set_zlim(0,2)

    origin = [0,0,0]
    e1, e2, e3 = [1,0,0], [0,1,0], [0,0,1]
    cols = ["#00FFFF", "#9B59B6", "#00FF88"]
    lbls = ["e₁", "e₂", "e₃"]
    for vec, col, lbl in zip([e1, e2, e3], cols, lbls):
        ax.quiver(*origin, *vec, color=col, linewidth=2.5,
                  arrow_length_ratio=0.25)
        ax.text(vec[0]+0.1, vec[1]+0.1, vec[2]+0.1, lbl,
                color=col, fontsize=16, fontweight="bold")

    # projection line
    v = np.array([0.8, 0.6, 0])
    proj = np.array([0.8, 0, 0])
    ax.quiver(*origin, *v, color="#FFD700", linewidth=2, arrow_length_ratio=0.2)
    ax.plot([v[0], proj[0]], [v[1], proj[1]], [v[2], proj[2]],
            "--", color="#FFD700", alpha=0.5, lw=1.5)
    return fig


def viz_ch5():
    fig, ax = plt.subplots(figsize=(5.76, 7.2))
    fig.patch.set_facecolor(tuple(c/255 for c in BG))
    _ax_style(ax)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")

    # unit circle
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#9B59B6", alpha=0.3, lw=1.5)

    # transformed ellipse (A @ unit circle)
    A = np.array([[2.0, 0.5], [0.3, 1.2]])
    pts = np.vstack([np.cos(theta), np.sin(theta)])
    tp = A @ pts
    ax.plot(tp[0], tp[1], color="#9B59B6", lw=2)

    # eigenvectors
    vals, vecs = np.linalg.eig(A)
    cols_e = ["#FFD700", "#00FFFF"]
    for val, vec, col in zip(vals, vecs.T, cols_e):
        ev = vec * val.real
        ax.annotate("", xy=(ev[0]*1.1, ev[1]*1.1), xytext=(0,0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=3, mutation_scale=22))
        ax.text(ev[0]*1.2+0.05, ev[1]*1.2+0.05,
                f"λ={val.real:.1f}", color=col, fontsize=16, fontweight="bold")

    ax.axhline(0, color=(1,1,1,0.08), lw=0.8)
    ax.axvline(0, color=(1,1,1,0.08), lw=0.8)
    return fig


def viz_ch6():
    fig, axes = plt.subplots(1, 3, figsize=(5.76, 7.2),
                              gridspec_kw={"width_ratios": [2, 0.5, 2]})
    fig.patch.set_facecolor(tuple(c/255 for c in BG))

    def heatmap(ax, data, cmap, title):
        _ax_style(ax)
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
        ax.set_title(title, color="#00FFFF", fontsize=13, pad=6, fontweight="bold")

    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (5, 4))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    heatmap(axes[0], U, "cool", "U")
    heatmap(axes[2], Vt, "cool", "Vᵀ")

    # middle: singular values bar
    _ax_style(axes[1])
    axes[1].set_title("Σ", color="#9B59B6", fontsize=13, pad=6, fontweight="bold")
    axes[1].barh(range(len(s)), s, color="#9B59B6", height=0.6)
    axes[1].set_xlim(0, s.max()*1.3)
    axes[1].set_ylim(-0.5, len(s)-0.5)
    axes[1].invert_yaxis()
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    # = sign between panels drawn in figure coords
    for x in [0.365, 0.66]:
        fig.text(x, 0.5, "=", color="#00FFFF", fontsize=24,
                 ha="center", va="center", fontweight="bold")

    fig.tight_layout(pad=0.3)
    return fig


def viz_ch7():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.76, 7.2))
    fig.patch.set_facecolor(tuple(c/255 for c in BG))

    # top: mini neural net
    _ax_style(ax1)
    ax1.set_xlim(0, 4); ax1.set_ylim(0, 3); ax1.set_aspect("equal")
    layers = [[0.5, [0.5, 1.5, 2.5]], [2.0, [0.75, 1.5, 2.25]], [3.5, [1.0, 2.0]]]
    colors_nn = ["#00FFFF", "#9B59B6", "#FFD700"]
    for (lx, nodes), col in zip(layers, colors_nn):
        for ny in nodes:
            circ = plt.Circle((lx, ny), 0.2, color=col, zorder=3)
            ax1.add_patch(circ)
    # connections
    for i in range(len(layers)-1):
        lx1, n1 = layers[i]
        lx2, n2 = layers[i+1]
        for ny1 in n1:
            for ny2 in n2:
                ax1.plot([lx1, lx2], [ny1, ny2],
                         color=(1,1,1,0.12), lw=0.8, zorder=1)
    ax1.set_title("Neural Network", color="#00FFFF", fontsize=14,
                  pad=4, fontweight="bold")

    # bottom: PCA scatter
    _ax_style(ax2)
    rng = np.random.default_rng(7)
    pts = rng.multivariate_normal([0,0], [[2,1.5],[1.5,2]], 80)
    ax2.scatter(pts[:,0], pts[:,1], color="#9B59B6", s=18, alpha=0.6)

    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eig(cov)
    for val, vec, col in zip(vals, vecs.T, ["#FFD700", "#00FFFF"]):
        ax2.annotate("", xy=(vec[0]*val**0.5*1.5, vec[1]*val**0.5*1.5),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=2.5, mutation_scale=18))
    ax2.set_title("PCA", color="#FFD700", fontsize=14, pad=4, fontweight="bold")
    fig.tight_layout(pad=0.5)
    return fig


# ── chapter data ──────────────────────────────────────────────────────────────
CHAPTERS = [
    dict(n=1, title="VECTORS",          hook="The Math Behind Every AI Model",  thai_hook="พื้นฐานที่ AI ทุกตัวต้องรู้",              accent=CYAN,   viz=viz_ch1, slug="ch1-vectors"),
    dict(n=2, title="MATRICES",         hook="How AI Transforms Your Data",     thai_hook="วิธีที่ AI แปลงข้อมูลของคุณ",             accent=PURPLE, viz=viz_ch2, slug="ch2-matrices"),
    dict(n=3, title="LINEAR\nEQUATIONS",hook="How AI Finds the Answer",         thai_hook="วิธีที่ AI หาคำตอบ",                       accent=GREEN,  viz=viz_ch3, slug="ch3-linear-equations"),
    dict(n=4, title="VECTOR\nSPACES",   hook="Where AI Data Actually Lives",    thai_hook="พื้นที่ที่ข้อมูล AI อาศัยอยู่",           accent=CYAN,   viz=viz_ch4, slug="ch4-vector-spaces"),
    dict(n=5, title="EIGEN\nVALUES",    hook="AI's Secret Direction Finder",    thai_hook="อาวุธลับที่ AI ใช้วิเคราะห์ข้อมูล",       accent=GOLD,   viz=viz_ch5, slug="ch5-eigenvalues"),
    dict(n=6, title="MATRIX\nDECOMP",   hook="Breaking AI's Black Box Open",    thai_hook="แกะกล่องดำของ AI",                         accent=PURPLE, viz=viz_ch6, slug="ch6-decompositions"),
    dict(n=7, title="LINEAR\nALGEBRA\nIN AI", hook="Now Everything Makes Sense",thai_hook="เมื่อทุกอย่างเชื่อมกัน",                  accent=CYAN,   viz=viz_ch7, slug="ch7-applications"),
]


# ── compose one thumbnail ─────────────────────────────────────────────────────
def make_cover(ch):
    img = make_bg()

    # render viz
    fig = ch["viz"]()
    fig.set_size_inches((W - VIZ_X) / 150, H / 150)
    viz_pil = fig_to_pil(fig)
    plt.close(fig)

    viz_pil = viz_pil.resize((W - VIZ_X, H), Image.LANCZOS)
    viz_rgba = viz_pil.convert("RGBA")

    img_rgba = img.convert("RGBA")
    paste_with_glow(img_rgba, viz_rgba, pos=(VIZ_X, 0))
    img = img_rgba.convert("RGB")

    draw_text_block(img, ch["n"], ch["title"], ch["hook"], ch["thai_hook"], ch["accent"])
    return img


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.join(os.path.dirname(__file__), "covers")
    os.makedirs(out_dir, exist_ok=True)

    for ch in CHAPTERS:
        print(f"  generating {ch['slug']}...", end=" ", flush=True)
        img = make_cover(ch)
        path = os.path.join(out_dir, f"{ch['slug']}.png")
        img.save(path, "PNG")
        print(f"saved → {path}")

    print(f"\nDone. {len(CHAPTERS)} covers in {out_dir}/")


if __name__ == "__main__":
    main()
