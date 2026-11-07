import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Landmark IDs (MediaPipe Pose has 33 points, 0-32)
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16

# ---- Tuning knobs: play with these! ----
FLAP_THRESHOLD = 0.02     # how hard you must flap (watch the FLAP meter)
SCARE_HALF_WIDTH = 0.22   # scare zone = this fraction of screen width each side of you
RICE_EAT_RATE = 0.06      # rice lost per eating crow per frame

# Colors are BGR in OpenCV, not RGB!
WHITE = (255, 255, 255)
RED = (40, 40, 230)
YELLOW = (0, 220, 255)
GREEN = (40, 150, 40)
PADDY = (90, 230, 170)
MUD = (40, 80, 110)
CROW_BODY = (28, 24, 24)
CROW_WING = (50, 42, 42)
CROW_SHEEN = (120, 90, 80)   # bluish edge so the crow stands out on dark backgrounds

SHOUTS = ["KAA!", "SHOO!", "POYI!", "KAA KAA!"]


# ======================= DRAWING =======================

def draw_crow(frame, crow, t):
    """Side-view crow. `facing` = 1 looks right, -1 looks left."""
    x, y, f = crow["x"], crow["y"], crow["facing"]
    peck = 0

    if crow["state"] == "eating":
        wing = 0.12                                    # wings folded
        peck = 7 if int(t * 8 + crow["seed"]) % 2 else 0  # head bobs to peck rice
    elif crow["state"] == "fleeing":
        wing = math.sin(t * 32 + crow["seed"])         # panicked fast flapping
    else:
        wing = math.sin(t * 14 + crow["seed"])         # normal flapping
        y += math.sin(t * 5 + crow["seed"]) * 4        # gentle bobbing in the air

    def P(dx, dy):
        """Point relative to the crow, mirrored when it faces left."""
        return (int(x + dx * f), int(y + dy))

    def poly(points, color, edge=None):
        pts = np.array([P(*p) for p in points], np.int32)
        cv2.fillPoly(frame, [pts], color)
        if edge:
            cv2.polylines(frame, [pts], True, edge, 1, cv2.LINE_AA)

    # legs (only when standing on the field)
    if crow["state"] == "eating":
        for lx in (-5, 6):
            cv2.line(frame, P(lx, 8), P(lx - 2, 20), (60, 60, 60), 2)

    # tail feathers, then body, then neck + head
    poly([(-20, -4), (-46, -12), (-50, -2), (-46, 6), (-20, 6)], CROW_BODY, CROW_SHEEN)
    cv2.ellipse(frame, P(0, 0), (27, 13), 0, 0, 360, CROW_BODY, -1)
    poly([(14, -8), (22, -14 + peck), (26, -4 + peck), (14, 6)], CROW_BODY)
    cv2.circle(frame, P(25, -10 + peck), 11, CROW_BODY, -1)

    # thick dark beak (real crows have black beaks, not yellow!)
    poly([(32, -14 + peck), (50, -7 + peck), (32, -3 + peck)], (55, 55, 55), (110, 110, 110))

    # eye with a tiny glint
    cv2.circle(frame, P(28, -13 + peck), 3, (210, 210, 210), -1)
    cv2.circle(frame, P(28, -13 + peck), 1, (0, 0, 0), -1)

    # wing: tip goes up when wing > 0 and down when wing < 0
    lift = -44 * wing
    poly([(12, -4), (-2, -4 + lift), (-14, -2 + lift * 0.8),
          (-24, -2 + lift * 0.55), (-16, 0)], CROW_WING, CROW_SHEEN)


def draw_field(frame, field_top, w, h, rice):
    """Paddy field. Stalks disappear as the rice gets eaten."""
    cv2.rectangle(frame, (0, field_top), (w, h), MUD, -1)
    for i, sx in enumerate(range(8, w, 18)):
        if (i * 37) % 100 < rice:           # keep roughly `rice`% of stalks
            top = field_top - 18 + (i * 13) % 12
            cv2.line(frame, (sx, h), (sx - 5, top), PADDY, 2)
            cv2.line(frame, (sx, h), (sx + 5, top + 4), GREEN, 2)
            cv2.circle(frame, (sx - 5, top), 3, YELLOW, -1)   # grain


def draw_bar(frame, x, y, w, h, frac, color, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * max(0, min(frac, 1))), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 1)
    cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)


# ======================= GAME LOGIC =======================

def new_game():
    now = time.time()
    return {"crows": [], "popups": [], "score": 0, "rice": 100.0,
            "start": now, "last_spawn": now, "over": False}


def spawn_crow(w, h, field_top, speed):
    """Crow enters from a random side and flies toward a random spot on the field."""
    x = random.choice([-60, w + 60])
    y = random.uniform(0.08 * h, 0.35 * h)
    tx, ty = random.uniform(60, w - 60), field_top - 20
    dist = math.hypot(tx - x, ty - y)
    return {"x": x, "y": y, "ty": ty,
            "vx": (tx - x) / dist * speed, "vy": (ty - y) / dist * speed,
            "facing": 1 if tx > x else -1,
            "state": "flying", "seed": random.random() * 10}


game = new_game()
prev_wrist_y = None
flap_energy = 0.0

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    print("Stand back so your whole upper body is visible. q = quit, r = restart")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        field_top = int(h * 0.82)
        t = time.time()
        elapsed = t - game["start"]

        # ---------- 1. SENSE ----------
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        player_x = None
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            # where is the farmer standing? (midpoint of shoulders)
            player_x = (lm[L_SHOULDER].x + lm[R_SHOULDER].x) / 2 * w

            # how fast are the wrists moving up/down? (change per frame)
            wrist_y = (lm[L_WRIST].y + lm[R_WRIST].y) / 2
            if prev_wrist_y is not None:
                dy = abs(wrist_y - prev_wrist_y)
                flap_energy = 0.7 * flap_energy + 0.3 * dy   # smooth out jitter
            prev_wrist_y = wrist_y
        else:
            prev_wrist_y = None
            flap_energy *= 0.8

        flapping = flap_energy > FLAP_THRESHOLD and player_x is not None

        # ---------- 2. UPDATE ----------
        if not game["over"]:
            speed = min(3 + elapsed * 0.05, 9)               # crows get faster...
            interval = max(0.5, 2.2 - elapsed * 0.03)        # ...and come more often
            if t - game["last_spawn"] > interval:
                game["crows"].append(spawn_crow(w, h, field_top, speed))
                game["last_spawn"] = t

            for crow in game["crows"]:
                if crow["state"] == "flying":
                    crow["x"] += crow["vx"]
                    crow["y"] += crow["vy"]
                    if crow["y"] >= crow["ty"]:
                        crow["y"] = crow["ty"]
                        crow["state"] = "eating"
                elif crow["state"] == "eating":
                    game["rice"] -= RICE_EAT_RATE
                elif crow["state"] == "fleeing":
                    crow["x"] += crow["vx"]
                    crow["y"] += crow["vy"]

                # SHOO! crows near the flapping farmer panic and flee
                if (flapping and crow["state"] != "fleeing"
                        and abs(crow["x"] - player_x) < SCARE_HALF_WIDTH * w):
                    game["score"] += 2 if crow["state"] == "eating" else 1
                    crow["facing"] = 1 if crow["x"] > player_x else -1
                    crow["vx"], crow["vy"] = crow["facing"] * 10, -8
                    crow["state"] = "fleeing"
                    game["popups"].append({"x": crow["x"], "y": crow["y"] - 30,
                                           "text": random.choice(SHOUTS), "ttl": 25})

            # forget crows that left the screen
            game["crows"] = [c for c in game["crows"]
                             if -120 < c["x"] < w + 120 and c["y"] > -100]

            game["rice"] = max(0.0, game["rice"])
            if game["rice"] == 0:
                game["over"] = True

        for p in game["popups"]:
            p["y"] -= 1.5
            p["ttl"] -= 1
        game["popups"] = [p for p in game["popups"] if p["ttl"] > 0]

        # ---------- 3. DRAW ----------
        # scare zone: a translucent band around the player while flapping
        if player_x is not None:
            overlay = frame.copy()
            x1 = int(player_x - SCARE_HALF_WIDTH * w)
            x2 = int(player_x + SCARE_HALF_WIDTH * w)
            color = YELLOW if flapping else (200, 200, 200)
            cv2.rectangle(overlay, (x1, 0), (x2, field_top), color, -1)
            cv2.addWeighted(overlay, 0.18 if flapping else 0.06, frame, 0.82 if flapping else 0.94, 0, frame)

        draw_field(frame, field_top, w, h, game["rice"])
        for crow in game["crows"]:
            draw_crow(frame, crow, t)
        for p in game["popups"]:
            cv2.putText(frame, p["text"], (int(p["x"]) - 30, int(p["y"])),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(frame, p["text"], (int(p["x"]) - 30, int(p["y"])),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, YELLOW, 2)

        # HUD
        cv2.putText(frame, f"Score: {game['score']}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2)
        draw_bar(frame, w - 220, 25, 200, 18, game["rice"] / 100, GREEN, "RICE")
        draw_bar(frame, w - 220, 70, 200, 12, flap_energy / (FLAP_THRESHOLD * 2),
                 YELLOW if flapping else (150, 150, 150), "FLAP")
        cv2.line(frame, (w - 120, 66), (w - 120, 86), RED, 2)   # threshold mark
        if player_x is None:
            cv2.putText(frame, "Step into the frame, farmer!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

        if game["over"]:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "THE CROWS WON!", (w // 2 - 210, h // 2 - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, RED, 3)
            cv2.putText(frame, f"Score: {game['score']}   Press R to replay",
                        (w // 2 - 210, h // 2 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2)

        cv2.imshow("Kaakka Shoo!", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            game = new_game()

cap.release()
cv2.destroyAllWindows()