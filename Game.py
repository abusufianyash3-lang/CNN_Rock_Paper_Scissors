import random
import cv2
import numpy as np
import pygame
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path("rps_cnn.keras")
LABELS = ["rock", "paper", "scissors"]
IMG_SIZE = (128, 128)
WIDTH, HEIGHT = 1000, 650
WHITE = (245, 245, 245)
DARK = (24, 29, 42)
ACCENT = (70, 180, 150)
YELLOW = (255, 210, 90)

if not MODEL_PATH.exists():
    raise SystemExit("Model not found. First run collect_data.py, then train_model.py.")

model = tf.keras.models.load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Check that it is connected and not being used by another app.")

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CNN Rock Paper Scissors")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 28)
small = pygame.font.SysFont("arial", 20)
big = pygame.font.SysFont("arial", 38, bold=True)

player_score = 0
computer_score = 0
draws = 0
player_move = "—"
computer_move = "—"
message = "Show your hand in the green square, then press SPACE."
confidence = 0.0
last_frame = None

def draw_text(text, x, y, font_obj=font, color=WHITE):
    surface = font_obj.render(text, True, color)
    screen.blit(surface, (x, y))

def winner(player, computer):
    if player == computer:
        return "Draw"
    if (player, computer) in [
        ("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")
    ]:
        return "You win!"
    return "Computer wins!"

running = True
while running:
    ok, frame = cap.read()
    if ok:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        size = min(280, h - 20, w - 20)
        x1, y1 = (w - size) // 2, (h - size) // 2
        x2, y2 = x1 + size, y1 + size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        last_frame = frame.copy()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE and last_frame is not None:
                h, w = last_frame.shape[:2]
                size = min(280, h - 20, w - 20)
                x1, y1 = (w - size) // 2, (h - size) // 2
                crop = last_frame[y1:y1+size, x1:x1+size]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, IMG_SIZE)
                batch = np.expand_dims(resized.astype("float32"), axis=0)
                probs = model.predict(batch, verbose=0)[0]
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                player_move = LABELS[idx]
                computer_move = random.choice(LABELS)
                result = winner(player_move, computer_move)
                if result == "You win!":
                    player_score += 1
                elif result == "Computer wins!":
                    computer_score += 1
                else:
                    draws += 1
                message = f"{result} (CNN confidence: {confidence:.0%})"

    screen.fill(DARK)
    draw_text("CNN ROCK PAPER SCISSORS", 30, 20, big, ACCENT)
    draw_text("Webcam preview", 30, 80, font)
    if last_frame is not None:
        rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        preview = pygame.surfarray.make_surface(np.rot90(rgb))
        preview = pygame.transform.smoothscale(preview, (540, 405))
        screen.blit(preview, (30, 120))
    else:
        draw_text("Waiting for webcam...", 30, 150)

    pygame.draw.rect(screen, (40, 48, 65), (610, 100, 355, 425), border_radius=16)
    draw_text("YOUR MOVE", 640, 130, small, YELLOW)
    draw_text(player_move.upper(), 640, 160, big, ACCENT)
    draw_text("COMPUTER MOVE", 640, 230, small, YELLOW)
    draw_text(computer_move.upper(), 640, 260, big, ACCENT)
    draw_text("SCORE", 640, 335, font)
    draw_text(f"You: {player_score}", 640, 375)
    draw_text(f"Computer: {computer_score}", 640, 415)
    draw_text(f"Draws: {draws}", 640, 455)
    draw_text(message, 30, 550, small, YELLOW)
    draw_text("SPACE = play round    ESC = exit", 30, 590, small)
    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()
cv2.destroyAllWindows()
