# Kaakka Shoo! 🐦‍⬛🌾

![Project Cover](assets/cover.jpeg)

**Be the scarecrow: flap your arms in front of a webcam to chase crows off a Kerala paddy field before they eat the whole harvest.**

Crows and other birds raid paddy fields in Kerala at sowing and ripening time, and farmers still guard them by hand: shouting "kaakka!", waving cloth, clapping. *Kaakka Shoo!* turns that everyday farm chore into a full-body game using MediaPipe Pose.

Built for **TinkerHub Saturday Hack Night – Season 02: Agriculture & Food (MediaPipe sprint)**.

## 🎮 How to play

1. Stand back so the camera sees your whole upper body.
2. **Raise both arms** above your shoulders and hold for half a second to start.
3. Crows fly in from both sides, land on the paddy and start eating. The **RICE** bar drops while they eat.
4. Move to stand under the crows and **flap your arms** up and down. Any crow inside the glowing scare zone around you flees with a "KAA!".
5. Points:
   - Scaring a flying crow: **1**, a crow that is already eating: **2**.
   - A rare **golden crow** flies faster and is worth **5**.
   - Scare several crows within about a second for a **COMBO** bonus.
6. When rice drops below 25% the screen shakes as a warning. At 0% the crows win.
7. Raise both arms again (or press **R**) to replay. **Q** quits.

Your best score is saved in `highscore.json`.

## 🧠 How it uses MediaPipe

The game uses **MediaPipe Pose** (legacy `mp.solutions.pose` API, `mediapipe==0.10.21`), which gives 33 body landmarks per frame. Only four are used:

| Landmark | ID | Used for |
|---|---|---|
| Left / right shoulder | 11, 12 | Player position: the midpoint of the shoulders is the centre of the scare zone |
| Left / right wrist | 15, 16 | Flap detection, and the arms-up start gesture |

**Flap detection.** Every frame the game takes the average height (`y`) of both wrists and measures how much it changed since the last frame. That vertical speed is smoothed into a `flap_energy` value:

```python
flap_energy = FLAP_SMOOTHING * flap_energy + (1 - FLAP_SMOOTHING) * abs(wrist_y - prev_wrist_y)
```

Smoothing removes webcam jitter, so standing still does not count but real flapping quickly fills the meter. When `flap_energy > FLAP_THRESHOLD` you are flapping, and crows within `SCARE_HALF_WIDTH` of your shoulder midpoint flee. The **FLAP** meter on screen shows this live, with a red mark at the threshold.

**Arms-up gesture.** Both wrists above their shoulders (smaller `y`, because image `y` grows downward), held for `ARMS_UP_HOLD` seconds.

## ⚙️ Setup

Needs **Python 3.12** (mediapipe 0.10.21 has no wheels for newer Pythons) and a webcam. Uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/abhi-jithb/SHN-MEDIAPIPE.git
cd SHN-MEDIAPIPE
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -r requirements.txt
python kaakka.py
```

Sound effects are generated in code with numpy and played with `sounddevice`. If no audio is available, the game prints a note and runs silently. On Linux, sound needs PortAudio (`sudo apt install libportaudio2` or `sudo dnf install portaudio`).

## 🎛️ Tuning knobs

All knobs live at the top of `kaakka.py`:

| Knob | Default | What it does |
|---|---|---|
| `FLAP_THRESHOLD` | 0.02 | How hard you must flap. Lower = easier |
| `FLAP_SMOOTHING` | 0.7 | Higher = smoother but slower flap meter |
| `SCARE_HALF_WIDTH` | 0.22 | Scare zone width each side of you (fraction of screen) |
| `START_SPEED` / `MAX_SPEED` / `SPEED_RAMP` | 3 / 9 / 0.05 | Crow speed and how fast it increases |
| `START_INTERVAL` / `MIN_INTERVAL` / `INTERVAL_RAMP` | 2.2 / 0.5 / 0.03 | Seconds between new crows and how fast that shrinks |
| `FLEE_SPEED` | 10 | How fast scared crows fly off |
| `RICE_EAT_RATE` | 0.06 | Rice lost per eating crow per frame |
| `ARMS_UP_HOLD` | 0.5 | Seconds to hold arms up to start / restart |
| `RESTART_DELAY` | 2.0 | Seconds after game over before the gesture restarts |
| `COMBO_WINDOW` | 1.0 | Scares within this many seconds form a combo |
| `LOW_RICE` / `SHAKE_TIME` | 25 / 0.6 | Low-rice warning level and shake duration |
| `GOLDEN_CHANCE` / `GOLDEN_SPEED` / `GOLDEN_POINTS` | 0.08 / 1.6 / 5 | Golden crow rarity, speed multiplier and points |
| `CAW_COOLDOWN` | 0.25 | Minimum seconds between caw sounds |

## 🛠️ Tech stack

Python 3.12 · MediaPipe Pose · OpenCV · NumPy · sounddevice

## 📸 Demo

[Demo video: add link here]

## 👥 Team

- Abhijith B – [abhi-jithb](https://github.com/abhi-jithb)

---

**Built with curiosity at Saturday Hack Night 💜**
