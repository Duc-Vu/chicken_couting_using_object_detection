import os

# ===== CONFIG =====
LABELS_IN  = "dataset/yolo/valid/labels_raw"        
LABELS_OUT = "dataset/yolo/valid/labels"   
MIN_SIZE  = 1e-4                     

os.makedirs(LABELS_OUT, exist_ok=True)

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

for root, _, files in os.walk(LABELS_IN):
    rel = os.path.relpath(root, LABELS_IN)
    out_dir = os.path.join(LABELS_OUT, rel)
    os.makedirs(out_dir, exist_ok=True)

    for fname in files:
        if not fname.endswith(".txt"):
            continue

        in_path  = os.path.join(root, fname)
        out_path = os.path.join(out_dir, fname)

        with open(in_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        out_lines = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5:
                continue

            cls = int(parts[0])

            # ---- CASE A: YOLO DETECT (5 số) ----
            if len(parts) == 5:
                _, xc, yc, w, h = parts

            # ---- CASE B: YOLO SEG (polygon) ----
            else:
                coords = parts[1:]
                xs = coords[0::2]
                ys = coords[1::2]

                x_min = clamp(min(xs))
                x_max = clamp(max(xs))
                y_min = clamp(min(ys))
                y_max = clamp(max(ys))

                w = x_max - x_min
                h = y_max - y_min
                xc = x_min + w / 2
                yc = y_min + h / 2

            # ---- Reject tiny boxes ----
            if w < MIN_SIZE or h < MIN_SIZE:
                continue

            xc = clamp(xc)
            yc = clamp(yc)
            w  = clamp(w)
            h  = clamp(h)

            out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)

print("✅ Convert xong: YOLO-seg + detect → YOLO-detect (5 số)")