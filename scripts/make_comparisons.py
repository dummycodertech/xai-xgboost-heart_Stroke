import os
from PIL import Image, ImageDraw, ImageFont

ARTIFACT_DIR = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f'
OUT_DIR = os.path.join(ARTIFACT_DIR, 'step2')
os.makedirs(OUT_DIR, exist_ok=True)

def create_side_by_side(img_left_path, img_right_path, label_left, label_right, out_path, target_h=700):
    im_l = Image.open(img_left_path).convert('RGB')
    im_r = Image.open(img_right_path).convert('RGB')

    # Scale left to target_h preserving aspect ratio
    w_l = int(im_l.width * (target_h / im_l.height))
    im_l = im_l.resize((w_l, target_h), Image.Resampling.LANCZOS)

    # Scale right to target_h preserving aspect ratio
    w_r = int(im_r.width * (target_h / im_r.height))
    im_r = im_r.resize((w_r, target_h), Image.Resampling.LANCZOS)

    header_h = 50
    total_w = w_l + w_r + 20
    total_h = target_h + header_h

    canvas = Image.new('RGB', (total_w, total_h), color=(20, 19, 17))
    canvas.paste(im_l, (0, header_h))
    canvas.paste(im_r, (w_l + 20, header_h))

    draw = ImageDraw.Draw(canvas)
    # Divider line
    draw.line([(w_l + 10, 0), (w_l + 10, total_h)], fill=(46, 44, 39), width=2)

    # Labels
    draw.text((20, 16), label_left, fill=(246, 242, 233))
    draw.text((w_l + 30, 16), label_right, fill=(212, 77, 110))

    canvas.save(out_path)
    print(f'Saved comparison: {out_path} ({total_w}x{total_h})')

# 1. Video A vs Hero Subject & HUD
create_side_by_side(
    'references/frames_A/frame_001.png',
    os.path.join(OUT_DIR, 'step2_heart_hero.png'),
    'REFERENCE VIDEO A (Jellyfish Bioluminescence + HUD Trackers)',
    'TRIAGE HERO (Fresnel X-Ray Anatomy + 14/14 HUD Tracking + Glitch Morph)',
    os.path.join(OUT_DIR, 'comparison_video_A.png')
)

# 2. Video B vs Hero Glass Blob
create_side_by_side(
    'references/frames_B/frame_001.png',
    os.path.join(OUT_DIR, 'step2_heart_hero.png'),
    'REFERENCE VIDEO B (Thin-Film Iridescent Glass Blob + Dot Grid)',
    'TRIAGE HERO BACKGROUND (4-Color Thin-Film Ramp + Refraction Caustics + Bayer Dither)',
    os.path.join(OUT_DIR, 'comparison_video_B.png')
)

# 3. Transition Sequence: Glitch & ASCII Flash
create_side_by_side(
    'references/frames_A/frame_005.png',
    os.path.join(ARTIFACT_DIR, 'step1', '03_mid_ascii_flash.png'),
    'REFERENCE VIDEO A (ASCII Morphed Flash ~240ms)',
    'TRIAGE HERO (Underlying Shared-Topology Morph + Monospace ASCII Flash)',
    os.path.join(OUT_DIR, 'comparison_glitch_ascii.png')
)
print('All comparison frames generated successfully!')
