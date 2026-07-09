"""
generate_dataset.py
--------------------
Generates a synthetic pedestrian-detection training dataset.

WHY SYNTHETIC DATA?
Real pedestrian datasets (INRIA Person, Caltech Pedestrian, PETS) are large
downloads (100MB-1GB+) that often aren't reachable from sandboxed/offline
environments. To let the FULL pipeline (preprocessing -> HOG -> SVM training
-> hyperparameter tuning -> evaluation) run end-to-end and produce real,
inspectable numbers, this script procedurally draws:
  - POSITIVE samples: simple humanoid silhouettes (head + torso + legs in
    varying poses, sizes, and positions) on varied backgrounds.
  - NEGATIVE samples: backgrounds, textures, and random non-human shapes
    (rectangles, circles, lines) that do NOT resemble a person.

This is a standard technique used for prototyping HOG+SVM pipelines, and the
code is written so that swapping in a real dataset (e.g. INRIA Person) later
is a one-line change -- just point `POS_DIR` / `NEG_DIR` at real images. The
notebook explains this clearly so it's honestly represented in any writeup.
"""

import os
import numpy as np
import cv2
import random

random.seed(42)
np.random.seed(42)

WIN_SIZE = (64, 128)  # standard HOG pedestrian detection window (w, h)
N_POSITIVE = 300
N_NEGATIVE = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POS_DIR = os.path.join(BASE_DIR, "data", "positive")
NEG_DIR = os.path.join(BASE_DIR, "data", "negative")
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)


def random_background(h, w):
    """Create a randomly textured/colored background to add variety."""
    choice = random.choice(["solid", "gradient", "noise", "stripes"])
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if choice == "solid":
        color = np.random.randint(40, 220, size=3).tolist()
        img[:] = color
    elif choice == "gradient":
        c1 = np.random.randint(20, 200, size=3)
        c2 = np.random.randint(20, 200, size=3)
        for y in range(h):
            alpha = y / h
            img[y, :] = (c1 * (1 - alpha) + c2 * alpha).astype(np.uint8)
    elif choice == "noise":
        img = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (7, 7), 0)
    elif choice == "stripes":
        color1 = np.random.randint(30, 220, size=3).tolist()
        color2 = np.random.randint(30, 220, size=3).tolist()
        img[:] = color1
        stripe_w = random.randint(5, 15)
        for x in range(0, w, stripe_w * 2):
            img[:, x:x + stripe_w] = color2

    # mild lighting variation (simulates varying lighting conditions)
    brightness = random.uniform(0.6, 1.3)
    img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return img


def draw_humanoid(img, cx, cy, scale, pose="standing"):
    """Draw a simple humanoid silhouette centered at (cx, cy)."""
    color = tuple(np.random.randint(10, 90, size=3).tolist())  # dark clothing tone
    head_r = int(8 * scale)

    # head
    head_cy = int(cy - 45 * scale)
    cv2.circle(img, (cx, head_cy), head_r, color, -1)

    # torso
    torso_top = head_cy + head_r
    torso_bottom = int(cy + 10 * scale)
    torso_w = int(16 * scale)
    cv2.rectangle(img, (cx - torso_w // 2, torso_top),
                  (cx + torso_w // 2, torso_bottom), color, -1)

    # arms (vary angle slightly for pose diversity)
    arm_len = int(35 * scale)
    arm_swing = random.randint(-8, 8)
    cv2.line(img, (cx - torso_w // 2, torso_top + 5),
              (cx - torso_w // 2 - 6, torso_top + arm_len + arm_swing),
              color, max(2, int(4 * scale)))
    cv2.line(img, (cx + torso_w // 2, torso_top + 5),
              (cx + torso_w // 2 + 6, torso_top + arm_len - arm_swing),
              color, max(2, int(4 * scale)))

    # legs (walking pose = offset legs, standing = parallel)
    leg_len = int(45 * scale)
    if pose == "walking":
        offset = random.randint(8, 16)
    else:
        offset = 2
    cv2.line(img, (cx - 4, torso_bottom),
              (cx - 4 - offset, torso_bottom + leg_len), color, max(2, int(5 * scale)))
    cv2.line(img, (cx + 4, torso_bottom),
              (cx + 4 + offset, torso_bottom + leg_len), color, max(2, int(5 * scale)))

    return img


def make_positive_sample():
    h, w = WIN_SIZE[1], WIN_SIZE[0]
    img = random_background(h, w)
    scale = random.uniform(0.85, 1.05)
    pose = random.choice(["standing", "walking"])
    cx = w // 2 + random.randint(-4, 4)
    cy = h // 2 + random.randint(-5, 5)
    img = draw_humanoid(img, cx, cy, scale, pose)
    # slight blur to simulate camera/motion conditions
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def make_negative_sample():
    h, w = WIN_SIZE[1], WIN_SIZE[0]
    img = random_background(h, w)
    # sometimes add a non-human distractor shape
    if random.random() < 0.5:
        shape = random.choice(["rect", "circle", "lines"])
        color = tuple(np.random.randint(10, 200, size=3).tolist())
        if shape == "rect":
            x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
            x2, y2 = x1 + random.randint(10, w // 2), y1 + random.randint(10, h // 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif shape == "circle":
            cv2.circle(img, (random.randint(10, w - 10), random.randint(10, h - 10)),
                       random.randint(5, 20), color, -1)
        else:
            for _ in range(3):
                p1 = (random.randint(0, w), random.randint(0, h))
                p2 = (random.randint(0, w), random.randint(0, h))
                cv2.line(img, p1, p2, color, 2)
    return img


def generate():
    print(f"Generating {N_POSITIVE} positive samples -> {POS_DIR}")
    for i in range(N_POSITIVE):
        img = make_positive_sample()
        cv2.imwrite(os.path.join(POS_DIR, f"pos_{i:04d}.png"), img)

    print(f"Generating {N_NEGATIVE} negative samples -> {NEG_DIR}")
    for i in range(N_NEGATIVE):
        img = make_negative_sample()
        cv2.imwrite(os.path.join(NEG_DIR, f"neg_{i:04d}.png"), img)

    print("Done.")


if __name__ == "__main__":
    generate()
