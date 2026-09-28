from pathlib import Path

import cv2
import numpy as np


def main():
    rng = np.random.default_rng(7)
    h, w = 512, 512

    # Speckled ultrasound-like background.
    base = rng.normal(loc=118, scale=25, size=(h, w)).astype(np.float32)

    y, x = np.mgrid[0:h, 0:w]

    # Gentle depth-dependent attenuation.
    depth = (y / h) * 25
    base -= depth

    # Dark synthetic lesion.
    cx, cy = 285, 270
    rx, ry = 78, 54
    lesion = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    base[lesion] -= 62

    # Heterogeneous internal texture.
    texture = rng.normal(0, 9, size=(h, w))
    base[lesion] += texture[lesion]

    # Posterior enhancement-like region.
    posterior = (
        (np.abs(x - cx) < 55)
        & (y > cy + 45)
        & (y < cy + 145)
    )
    base[posterior] += 18

    image = np.clip(base, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Fan-like border/shadow for a more ultrasound-like demonstration.
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        color,
        "SYNTHETIC RESEARCH DEMO",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    output = Path("data/demo_ultrasound.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), color)
    print(f"Synthetic demo image written to: {output.resolve()}")


if __name__ == "__main__":
    main()
