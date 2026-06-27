import pygame
import json
import os
import sys
import pandas as pd

# --- CONFIGURATION ---
CSV_PATH = os.path.join("dataset", "lineups.csv")
JSONL_PATH = os.path.join("hidden", "batch_results.jsonl")
IMAGES_DIR = os.path.join("dataset", "images")
WINDOW_SIZE = (1280, 720)
FPS = 60

def load_data():
    """Loads the CSV and JSONL files and merges them based on ID."""
    
    # 1. Load ground truth from CSV
    ground_truth = {}
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            ground_truth[str(row["id"])] = f"{row['map']} - {row['callout']}"
    else:
        print(f"Warning: CSV not found at {CSV_PATH}")

    # 2. Load predictions from JSONL
    results = []
    if not os.path.exists(JSONL_PATH):
        print(f"Error: JSONL not found at {JSONL_PATH}")
        sys.exit(1)
        
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # Extract ID (e.g., "image_task_6405" -> "6405")
            full_id = data.get("id", "")
            img_id = full_id.replace("image_task_", "")
            
            # Extract nested prediction text
            try:
                raw_text = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
                # The text itself is a JSON string, so we parse it again
                pred_json = json.loads(raw_text)
                prediction = str(pred_json.get("text"))
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                prediction = "Parse Error or No Prediction"
                
            results.append({
                "id": int(img_id) + 1,
                "prediction": prediction,
                "truth": ground_truth.get(img_id, "Unknown")
            })
            
    return results

def load_image(img_id):
    """Attempts to load an image, checking common extensions."""
    base_path = os.path.join(IMAGES_DIR, str(img_id))
    for ext in [".jpg", ".webp", ".jpeg"]:
        full_path = base_path + ext
        if os.path.exists(full_path):
            return pygame.image.load(full_path)
    return None

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Dataset Result Viewer")
    clock = pygame.time.Clock()
    
    # Fonts
    font_large = pygame.font.SysFont("Arial", 36, bold=True)
    font_medium = pygame.font.SysFont("Arial", 28)
    font_small = pygame.font.SysFont("Arial", 18)

    # Load data
    print("Loading data...")
    data = load_data()
    if not data:
        print("No data found to display.")
        sys.exit()
        
    current_idx = 0
    running = True

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Click moves to next image
                if current_idx < len(data) - 1:
                    current_idx += 1
                else:
                    running = False # End of list
            elif event.type == pygame.KEYDOWN:
                # Add keyboard support (right arrow / space to advance, left arrow to go back)
                if event.key in (pygame.K_RIGHT, pygame.K_SPACE, pygame.K_RETURN):
                    if current_idx < len(data) - 1:
                        current_idx += 1
                elif event.key == pygame.K_LEFT:
                    if current_idx > 0:
                        current_idx -= 1

        # --- Rendering ---
        screen.fill((30, 32, 36)) # Dark gray background

        current_item = data[current_idx]
        img_id = current_item["id"]
        pred_text = current_item["prediction"]
        truth_text = current_item["truth"]

        # 1. Render Image (Left Side)
        img = load_image(img_id)
        if img:
            # Scale image to fit the left portion of the screen (max 800x680)
            img_rect = img.get_rect()
            scale = min(800 / img_rect.width, 680 / img_rect.height)
            new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
            img = pygame.transform.smoothscale(img, new_size)
            
            # Center it on the left half
            img_x = (800 - new_size[0]) // 2 + 20
            img_y = (720 - new_size[1]) // 2
            screen.blit(img, (img_x, img_y))
        else:
            err_surface = font_medium.render(f"Image {img_id} not found in {IMAGES_DIR}", True, (255, 80, 80))
            screen.blit(err_surface, (50, WINDOW_SIZE[1]//2))

        # 2. Render Text (Right Side)
        text_start_x = 850
        
        # Progress Counter
        prog_surface = font_small.render(f"Result {current_idx + 1} of {len(data)}", True, (150, 150, 150))
        screen.blit(prog_surface, (text_start_x, 50))
        
        # ID
        id_surface = font_large.render(f"ID: {img_id}", True, (255, 255, 255))
        screen.blit(id_surface, (text_start_x, 90))

        # Ground Truth
        truth_label = font_small.render("Ground Truth (CSV):", True, (180, 180, 180))
        screen.blit(truth_label, (text_start_x, 160))
        truth_val = font_medium.render(truth_text, True, (100, 255, 100)) # Green
        screen.blit(truth_val, (text_start_x, 190))

        # Prediction
        pred_label = font_small.render("Model Prediction (JSONL):", True, (180, 180, 180))
        screen.blit(pred_label, (text_start_x, 260))
        
        # Color code prediction: Red if 'null', Blue otherwise
        pred_color = (255, 100, 100) if pred_text == "None" else (100, 200, 255)
        pred_val = font_medium.render(pred_text, True, pred_color)
        screen.blit(pred_val, (text_start_x, 290))

        # Instructions
        inst_surface = font_small.render("Click anywhere or press Space to continue ->", True, (100, 100, 100))
        screen.blit(inst_surface, (text_start_x, WINDOW_SIZE[1] - 50))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()