import json
import math
import os
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

# ================== TUNING KNOBS: play with these! ==================
# Flap detection (watch the FLAP meter on screen while you tune these)
FLAP_THRESHOLD = 0.02     # how hard you must flap; lower = easier to scare crows
FLAP_SMOOTHING = 0.7      # 0..1, higher = smoother but slower-reacting flap meter
SCARE_HALF_WIDTH = 0.22   # scare zone = this fraction of screen width each side of you

# Difficulty: crows get faster and come more often as time passes
START_SPEED = 3           # crow speed (pixels per frame) at the start
MAX_SPEED = 9             # crows never get faster than this
SPEED_RAMP = 0.05         # speed added per second of play
START_INTERVAL = 2.2      # seconds between new crows at the start
MIN_INTERVAL = 0.5        # crows never come more often than this
INTERVAL_RAMP = 0.03      # seconds shaved off the interval per second of play
FLEE_SPEED = 10           # how fast a scared crow flies away
RICE_EAT_RATE = 0.06      # rice % lost per eating crow per frame

# Start / restart gesture
ARMS_UP_HOLD = 0.5        # seconds both arms must stay up to start / restart
RESTART_DELAY = 2.0       # after game over, ignore the arms-up gesture this long

# Combos, warnings and golden crows
COMBO_WINDOW = 1.0        # scares within this many seconds chain into a combo
LOW_RICE = 25             # below this % of rice the screen shakes as a warning
SHAKE_TIME = 0.6          # how long (seconds) the low-rice shake lasts
GOLDEN_CHANCE = 0.08      # chance that a new crow is a rare golden crow (0.08 = 8%)
GOLDEN_SPEED = 1.6        # golden crows fly this many times faster
GOLDEN_POINTS = 5         # points for scaring a golden crow

# Sound
CAW_COOLDOWN = 0.25       # seconds between caws, so a flock doesn't caw-spam

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
GOLD_BODY = (0, 170, 230)
GOLD_WING = (20, 200, 255)
GOLD_SHEEN = (150, 245, 255)

SHOUTS = ["KAA!", "SHOO!", "POYI!", "KAA KAA!"]


# ======================= SOUND =======================
# All sounds are generated with numpy (no audio files). sd.play() returns
# immediately and plays in a background thread, so the video loop never waits.
# If audio can't start (no PortAudio, no output device...), we just stay silent.

SAMPLE_RATE = 44100

try:
    import sounddevice as sd
    sd.query_devices(kind="output")   # raises if there is no usable output device
except Exception as e:
    sd = None
    print(f"[sound] audio disabled ({e}); playing silently.")


def make_caw():
    """Harsh 'KAA!': a buzzy sawtooth sliding down in pitch, plus a bit of rasp."""
    dur = 0.28
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    freq = 750 - 350 * (t / dur) + 25 * np.sin(2 * np.pi * 30 * t)   # slide down + wobble
    phase = 2 * np.cumsum(freq) / SAMPLE_RATE * np.pi
    saw = 2 * ((phase / (2 * np.pi)) % 1.0) - 1                      # sawtooth = buzzy
    rasp = np.random.uniform(-1, 1, t.size) * 0.35                   # throaty noise
    env = np.minimum(t / 0.02, 1) * np.exp(-t * 6)                   # quick attack, fade out
    return (0.35 * (saw + rasp) * env).astype(np.float32)


def make_sad():
    """Game-over 'wah wah wah waaah': four falling notes, the last one wobbles."""
    notes = [(392, 0.35), (370, 0.35), (349, 0.35), (330, 1.0)]
    parts = []
    for i, (f, dur) in enumerate(notes):
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        vib = 6 * np.sin(2 * np.pi * 5 * t) if i == len(notes) - 1 else 0
        phase = 2 * np.pi * np.cumsum(f + vib) / SAMPLE_RATE
        tone = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)  # brassy
        env = np.minimum(t / 0.03, 1) * np.minimum((dur - t) / 0.08, 1)        # no clicks
        parts.append(tone * env)
    return (0.25 * np.concatenate(parts)).astype(np.float32)


CAW_SOUND = make_caw()
SAD_SOUND = make_sad()
last_caw = 0.0


def play(sound):
    """Fire-and-forget playback. Never raises, never blocks."""
    global sd
    if sd is None:
        return
    try:
        sd.play(sound, SAMPLE_RATE)
    except Exception as e:
        print(f"[sound] playback failed ({e}); disabling audio.")
        sd = None


def caw():
    global last_caw
    now = time.time()
    if now - last_caw > CAW_COOLDOWN:
        last_caw = now
        play(CAW_SOUND)


# ======================= DRAWING =======================

def draw_crow(frame, crow, t):
    """Side-view crow. `facing` = 1 looks right, -1 looks left."""
    x, y, f = crow["x"], crow["y"], crow["facing"]
    peck = 0
    if crow["golden"]:
        body, wing_color, sheen = GOLD_BODY, GOLD_WING, GOLD_SHEEN
    else:
        body, wing_color, sheen = CROW_BODY, CROW_WING, CROW_SHEEN

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
    poly([(-20, -4), (-46, -12), (-50, -2), (-46, 6), (-20, 6)], body, sheen)
    cv2.ellipse(frame, P(0, 0), (27, 13), 0, 0, 360, body, -1)
    poly([(14, -8), (22, -14 + peck), (26, -4 + peck), (14, 6)], body)
    cv2.circle(frame, P(25, -10 + peck), 11, body, -1)

    # thick dark beak (real crows have black beaks, not yellow!)
    poly([(32, -14 + peck), (50, -7 + peck), (32, -3 + peck)], (55, 55, 55), (110, 110, 110))

    # eye with a tiny glint
    cv2.circle(frame, P(28, -13 + peck), 3, (210, 210, 210), -1)
    cv2.circle(frame, P(28, -13 + peck), 1, (0, 0, 0), -1)

    # wing: tip goes up when wing > 0 and down when wing < 0
    lift = -44 * wing
    poly([(12, -4), (-2, -4 + lift), (-14, -2 + lift * 0.8),
          (-24, -2 + lift * 0.55), (-16, 0)], wing_color, sheen)


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


# ======================= HIGH SCORE =======================
# Saved next to this script, so it works no matter which folder you run from.
HIGHSCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highscore.json")


def load_highscore():
    try:
        with open(HIGHSCORE_FILE) as f:
            return int(json.load(f)["best"])
    except Exception:          # first run, or the file is broken: start from 0
        return 0


def save_highscore(best):
    try:
        with open(HIGHSCORE_FILE, "w") as f:
            json.dump({"best": best}, f)
    except Exception as e:     # can't save? keep playing anyway
        print(f"[highscore] could not save ({e})")


# ======================= GAME LOGIC =======================

def new_game(waiting=True):
    """waiting=True shows the start screen until the player raises both arms."""
    now = time.time()
    return {"crows": [], "popups": [], "score": 0, "rice": 100.0,
            "start": now, "last_spawn": now, "over": False, "over_at": 0.0,
            "waiting": waiting,
            "scare_times": [],     # when recent crows were scared (for combos)
            "combo": None,         # big "COMBO xN!" banner currently on screen
            "warned": False,       # has the low-rice warning already fired?
            "shake_until": 0.0}


def spawn_crow(w, h, field_top, speed):
    """Crow enters from a random side and flies toward a random spot on the field."""
    golden = random.random() < GOLDEN_CHANCE
    if golden:
        speed *= GOLDEN_SPEED
    x = random.choice([-60, w + 60])
    y = random.uniform(0.08 * h, 0.35 * h)
    tx, ty = random.uniform(60, w - 60), field_top - 20
    dist = math.hypot(tx - x, ty - y)
    return {"x": x, "y": y, "ty": ty,
            "vx": (tx - x) / dist * speed, "vy": (ty - y) / dist * speed,
            "facing": 1 if tx > x else -1,
            "state": "flying", "seed": random.random() * 10, "golden": golden}


game = new_game()
best = load_highscore()
prev_wrist_y = None
flap_energy = 0.0
arms_up_since = None   # when the player first raised both arms (None = arms down)

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
        arms_up = False
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            # where is the farmer standing? (midpoint of shoulders)
            player_x = (lm[L_SHOULDER].x + lm[R_SHOULDER].x) / 2 * w

            # how fast are the wrists moving up/down? (change per frame)
            wrist_y = (lm[L_WRIST].y + lm[R_WRIST].y) / 2
            if prev_wrist_y is not None:
                dy = abs(wrist_y - prev_wrist_y)
                flap_energy = FLAP_SMOOTHING * flap_energy + (1 - FLAP_SMOOTHING) * dy   # smooth jitter
            prev_wrist_y = wrist_y

            # both arms up? (y grows DOWNWARD in images, so "above" = smaller y)
            arms_up = (lm[L_WRIST].y < lm[L_SHOULDER].y and
                       lm[R_WRIST].y < lm[R_SHOULDER].y)
        else:
            prev_wrist_y = None
            flap_energy *= 0.8

        flapping = flap_energy > FLAP_THRESHOLD and player_x is not None

        # arms must stay up for ARMS_UP_HOLD seconds, so a quick flap doesn't count
        if arms_up:
            if arms_up_since is None:
                arms_up_since = t
        else:
            arms_up_since = None
        arms_up_held = arms_up_since is not None and t - arms_up_since > ARMS_UP_HOLD

        # ---------- 2. UPDATE ----------
        if game["waiting"]:
            if arms_up_held:                  # start! reset the clock so difficulty starts low
                game["waiting"] = False
                game["start"] = game["last_spawn"] = t
                arms_up_since = None
        elif game["over"]:
            if arms_up_held and t - game["over_at"] > RESTART_DELAY:
                game = new_game(waiting=False)
                arms_up_since = None
        else:
            speed = min(START_SPEED + elapsed * SPEED_RAMP, MAX_SPEED)           # faster...
            interval = max(MIN_INTERVAL, START_INTERVAL - elapsed * INTERVAL_RAMP)  # ...and more often
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
                    if crow["golden"]:
                        game["score"] += GOLDEN_POINTS
                        shout = f"GOLD +{GOLDEN_POINTS}!"
                    else:
                        game["score"] += 2 if crow["state"] == "eating" else 1
                        shout = random.choice(SHOUTS)
                    crow["facing"] = 1 if crow["x"] > player_x else -1
                    crow["vx"], crow["vy"] = crow["facing"] * FLEE_SPEED, -0.8 * FLEE_SPEED
                    crow["state"] = "fleeing"
                    game["popups"].append({"x": crow["x"], "y": crow["y"] - 30,
                                           "text": shout, "ttl": 25})
                    caw()

                    # COMBO: count scares in the last COMBO_WINDOW seconds
                    game["scare_times"].append(t)
                    game["scare_times"] = [s for s in game["scare_times"]
                                           if t - s < COMBO_WINDOW]
                    combo = len(game["scare_times"])
                    if combo >= 2:
                        game["score"] += combo - 1          # bigger combo, bigger bonus
                        game["combo"] = {"n": combo, "ttl": 40}

            # forget crows that left the screen
            game["crows"] = [c for c in game["crows"]
                             if -120 < c["x"] < w + 120 and c["y"] > -100]

            game["rice"] = max(0.0, game["rice"])
            if game["rice"] < LOW_RICE and not game["warned"]:   # fire once per game
                game["warned"] = True
                game["shake_until"] = t + SHAKE_TIME
            if game["rice"] == 0:
                game["over"] = True
                game["over_at"] = t
                play(SAD_SOUND)
                if game["score"] > best:
                    best = game["score"]
                    save_highscore(best)   # runs once: UPDATE is skipped once the game is over

        for p in game["popups"]:
            p["y"] -= 1.5
            p["ttl"] -= 1
        game["popups"] = [p for p in game["popups"] if p["ttl"] > 0]
        if game["combo"]:
            game["combo"]["ttl"] -= 1
            if game["combo"]["ttl"] <= 0:
                game["combo"] = None

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
        if game["combo"]:
            text = f"COMBO x{game['combo']['n']}!"
            cv2.putText(frame, text, (w // 2 - 130, 90), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 7)
            cv2.putText(frame, text, (w // 2 - 130, 90), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 140, 255), 3)

        # LOW RICE warning: shake the picture and flash it red for a moment
        if t < game["shake_until"]:
            dx, dy = random.randint(-12, 12), random.randint(-12, 12)
            frame = np.roll(frame, (dy, dx), axis=(0, 1))
            if int(t * 10) % 2:                       # flash on/off 5 times a second
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), RED, -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # HUD
        cv2.putText(frame, f"Score: {game['score']}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2)
        cv2.putText(frame, f"Best: {max(best, game['score'])}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
        draw_bar(frame, w - 220, 25, 200, 18, game["rice"] / 100, GREEN, "RICE")
        draw_bar(frame, w - 220, 70, 200, 12, flap_energy / (FLAP_THRESHOLD * 2),
                 YELLOW if flapping else (150, 150, 150), "FLAP")
        cv2.line(frame, (w - 120, 66), (w - 120, 86), RED, 2)   # threshold mark
        if player_x is None:
            cv2.putText(frame, "Step into the frame, farmer!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

        if game["waiting"]:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "KAAKKA SHOO!", (w // 2 - 180, h // 2 - 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, YELLOW, 3)
            cv2.putText(frame, "Raise both arms to start", (w // 2 - 200, h // 2 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2)

        if game["over"]:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "THE CROWS WON!", (w // 2 - 210, h // 2 - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, RED, 3)
            cv2.putText(frame, f"Score: {game['score']}   Best: {best}",
                        (w // 2 - 210, h // 2 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2)
            cv2.putText(frame, "Raise both arms (or press R) to replay",
                        (w // 2 - 210, h // 2 + 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, WHITE, 2)

        cv2.imshow("Kaakka Shoo!", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if game["score"] > best:     # don't lose a record by quitting mid-game
                save_highscore(game["score"])
            break
        if key == ord('r'):
            game = new_game(waiting=False)   # R jumps straight back into play

cap.release()
cv2.destroyAllWindows()