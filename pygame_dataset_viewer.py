import csv
import os
import sys

try:
    import pygame
except ImportError as exc:
    raise SystemExit("pygame is required. Install it with: pip install pygame") from exc


CSV_PATH = os.path.join("dataset", "lineups.csv")
IMAGE_DIR = os.path.join("dataset", "images")
WINDOW_SIZE = (1100, 750)
TEXT_COLOR = (235, 235, 235)

THEMES = {
    "True": {
        "background": (16, 44, 28),
        "panel": (22, 64, 38),
        "accent": (120, 255, 170),
    },
    "False": {
        "background": (52, 18, 20),
        "panel": (76, 24, 30),
        "accent": (255, 136, 136),
    },
    "default": {
        "background": (20, 20, 24),
        "panel": (34, 34, 42),
        "accent": (110, 170, 255),
    },
}


def _normalize_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return "True"
    if value in {"false", "0", "no", "n"}:
        return "False"
    return None


def load_lineups(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    headers = [h.strip() for h in rows[0]]
    data_rows = rows[1:]

    if not data_rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    max_len = max(len(r) for r in data_rows)
    if len(headers) < max_len:
        for i in range(len(headers), max_len):
            headers.append(f"column_{i + 1}")

    records = []
    for row in data_rows:
        if not row:
            continue
        padded = row + [""] * (len(headers) - len(row))
        rec = {headers[i]: padded[i].strip() for i in range(len(headers))}
        records.append(rec)

    id_key = "id" if "id" in headers else headers[0]

    bool_key = None
    for key in headers:
        vals = [records[i][key] for i in range(len(records)) if records[i][key] != ""]
        if not vals:
            continue
        normalized = [_normalize_bool(v) for v in vals]
        if all(v is not None for v in normalized):
            bool_key = key
            break

    if bool_key is None and len(headers) > 1:
        bool_key = headers[-1]

    normalized = []
    for rec in records:
        image_id = rec.get(id_key, "").strip()
        if not image_id:
            continue
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.webp")
        normalized.append(
            {
                "id": image_id,
                "map": rec.get("map", ""),
                "callout": rec.get("callout", ""),
                "bool": _normalize_bool(rec.get(bool_key, "")) or rec.get(bool_key, ""),
                "image_path": image_path,
            }
        )

    if not normalized:
        raise ValueError("No usable rows found in CSV.")

    return normalized, bool_key


def fit_size(img_w, img_h, max_w, max_h):
    scale = min(max_w / img_w, max_h / img_h)
    return max(1, int(img_w * scale)), max(1, int(img_h * scale))


def load_surface(path):
    if not os.path.exists(path):
        return None
    try:
        return pygame.image.load(path).convert()
    except pygame.error:
        return None


def get_theme(value):
    return THEMES.get(value, THEMES["default"])


def draw(screen, fonts, record, index, total, bool_key):
    theme = get_theme(record["bool"])
    screen.fill(theme["background"])
    w, h = screen.get_size()
    img_area_h = int(h * 0.78)
    text_y = img_area_h + 12
    margin = 24

    image_surface = load_surface(record["image_path"])
    if image_surface is not None:
        iw, ih = image_surface.get_size()
        tw, th = fit_size(iw, ih, w - margin * 2, img_area_h - margin * 2)
        image_scaled = pygame.transform.smoothscale(image_surface, (tw, th))
        x = (w - tw) // 2
        y = (img_area_h - th) // 2
        screen.blit(image_scaled, (x, y))
    else:
        missing = fonts["main"].render(f"Missing image: {record['image_path']}", True, (255, 120, 120))
        screen.blit(missing, (margin, margin))

    line1 = f"[{index + 1}/{total}] id={record['id']}  map={record['map']}  callout={record['callout']}"
    line2 = f"{bool_key}: {record['bool']}"
    line3 = "Click anywhere to go to the next image. (Esc to quit)"

    panel_rect = pygame.Rect(0, img_area_h, w, h - img_area_h)
    pygame.draw.rect(screen, theme["panel"], panel_rect)

    screen.blit(fonts["main"].render(line1, True, TEXT_COLOR), (margin, text_y))
    screen.blit(fonts["big"].render(line2, True, theme["accent"]), (margin, text_y + 34))
    screen.blit(fonts["small"].render(line3, True, TEXT_COLOR), (margin, text_y + 78))

    pygame.display.flip()


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    rows, bool_key = load_lineups(csv_path)

    pygame.init()
    pygame.display.set_caption("Lineups CSV Viewer")
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    fonts = {
        "main": pygame.font.SysFont("Segoe UI", 24),
        "big": pygame.font.SysFont("Segoe UI", 34, bold=True),
        "small": pygame.font.SysFont("Segoe UI", 20),
    }

    i = 0
    draw(screen, fonts, rows[i], i, len(rows), bool_key)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT):
                    i = (i + 1) % len(rows)
                    draw(screen, fonts, rows[i], i, len(rows), bool_key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                i = (i + 1) % len(rows)
                draw(screen, fonts, rows[i], i, len(rows), bool_key)
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                draw(screen, fonts, rows[i], i, len(rows), bool_key)

    pygame.quit()


if __name__ == "__main__":
    main()
