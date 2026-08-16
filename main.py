import pygame as pg
import numpy as np
from scipy.io import wavfile
import glob
import os

from UI import Generator, Pedal, Sink
from pedal_functions import *

SAMPLE_RATE = 48000
CHUNK_SIZE = 512
pg.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=CHUNK_SIZE)

pg.init()
pg.font.init()

WIDTH, HEIGHT = 1440, 720
display = pg.display.set_mode((WIDTH, HEIGHT))

audio_channel = pg.mixer.Channel(0)

# --- WAV list setup --------------------------------------------------
# collect all .wav files in the working directory
wav_files = sorted(glob.glob("*.wav"))
if not wav_files and os.path.exists("emo.wav"):
    wav_files = ["emo.wav"]

# load wav files into memory (simple normalisation, first channel)
wav_datas = []
for p in wav_files:
    try:
        sr, d = wavfile.read(p)
    except Exception:
        continue

    # take first channel if stereo
    if d.ndim > 1:
        d = d[:, 0]

    d = d.astype(np.float32)
    maxv = np.max(np.abs(d))
    if maxv > 0:
        d = d / maxv

    wav_datas.append(d)

# fallback: if nothing was found, create a silent buffer
if not wav_datas:
    wav_datas = [np.zeros(48000, dtype=np.float32)]
    wav_files = ["(silent)"]

current_wav_index = 0

def current_wav_name():
    return os.path.basename(wav_files[current_wav_index])

# --- DSP functions -----------------------------------------------------
# Generators are called as func(phase, num_samples) -> float32[-1, 1].
# Pedals are called as func(data) -> float32[-1, 1] of the same shape.

def sine_source(phase, num_samples):
    t = (phase + np.arange(num_samples)) / SAMPLE_RATE
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

def get_wav(pos, num_samples):
    data = wav_datas[current_wav_index]
    if data is None or len(data) == 0:
        return np.zeros(num_samples, dtype=np.float32)

    pos = int(pos) % len(data)
    end = pos + num_samples

    if end <= len(data):
        chunk = data[pos:end]
    else:
        first_part = data[pos:]
        remaining = num_samples - len(first_part)
        second_part = data[:remaining]
        chunk = np.concatenate([first_part, second_part])

    return chunk.astype(np.float32)


# --- Graph ---------------------------------------------------------------

generator = Generator((60, 260, 250, 120), f"OSC ({current_wav_name()})", func=get_wav)

'''pedals = [
    Pedal((300, 60, 220, 120), "OVERDRIVE", overdrive, knobs={"gain": (0.0, 1.0, 1.0)}),
    Pedal((300, 460, 220, 120), "FUZZ", fuzz2, knobs={"gain": (0.0, 1.0, 1.0)}),
    Pedal((600, 460, 220, 120), "PITCH SHIFT", td_psola_pitch_shift, knobs={"shift": (0.0, 1.0, 1.0)}),
    Pedal((300, 260, 220, 120), "DISTORTION", distortion, knobs={"gain": (0.0, 1.0, 1.0)}),
    Pedal((600, 260, 220, 120), "FUZZ GATE", fuzz_gate, knobs={"gate_thresh": (0.01, 0.05, 0.2), "gate_speed": (0, 1, 5)}),
]'''

pedals = []

sink = Sink((900, 260, 140, 120), "OUT")

# Everything that needs events/drawing/terminal lookups, source to sink.
nodes = [generator, sink]

# --- Spawnable pedals -------------------------------------------------
# list of (display name, function, knobs mapping or None)
spawnable_defs = [
    ("OVERDRIVE", overdrive, {"gain": (0.0, 1.0, 1.0)}),
    ("FUZZ", fuzz2, {"gain": (0.0, 1.0, 1.0)}),
    ("FUZZ_SOFT", fuzz, {"gain": (0.0, 1.0, 1.0)}),
    ("DISTORTION", distortion, {"gain": (0.0, 1.0, 1.0)}),
    ("FUZZ_GATE", fuzz_gate, {"gate_thresh": (0.01, 0.05, 0.2), "gate_speed": (0, 1, 5)}),
    ("DIODE_CLIP", diode_clip, {"gain": (0.0, 1.0, 1.0), "vt": (0.001, 0.05, 0.5), "vth": (0.05, 0.5, 1.0)}),
    ("PITCH_PSOLA", td_psola_pitch_shift, {"shift": (0.0, 1.0, 1.0)}),
    ("PITCH_GRAIN", better_pitch_shift, {"shift": (0.0, 1.0, 1.0)}),
    ("PITCH_SIMPLE", pitch_shift, {"shift": (0.0, 1.0, 5.0 , 1.0)}),
    ("NORMALIZER", normalizer, {"window": (0.5, 2.0, 10.0)}),
    ("LOWPASS", low_pass, {"cutoff": (0.0, 0.5, 1.0)}),
    ("REVERB", reverb, {"delay": (0.0, 0.0, 1.0)}),
]

# spawn counts and top-bar buttons (max 2 per pedal type)
spawn_counts = {name: 0 for name, _, _ in spawnable_defs}
top_buttons = []

def spawn_pedal(name):
    print("waza", name)
    # enforce max 2 per type
    if spawn_counts.get(name, 0) >= 5:
        return

    # find definition
    for n, func, knobs in spawnable_defs:
        if n == name:
            mx, my = pg.mouse.get_pos()
            # create pedal rect located below the mouse
            x = max(0, mx - 110)
            y = min(HEIGHT - 140, my + 10)
            new_p = Pedal((x, y, 220, 120), name, func, knobs=knobs)

            # add to pedals and nodes (before sink)
            pedals.append(new_p)
            # insert before sink so sink stays last
            try:
                sink_index = nodes.index(sink)
                nodes.insert(sink_index, new_p)
            except ValueError:
                nodes.append(new_p)

            spawn_counts[name] = spawn_counts.get(name, 0) + 1
            # update button label to show count
            for btn in top_buttons:
                if btn.label.startswith(name):
                    btn.label = f"{name} ({spawn_counts[name]}/5)"
            return

# create top buttons (wrap to next row if needed)
for i, (name, _, _) in enumerate(spawnable_defs):
    per_row = max(1, WIDTH // 110)
    row = i // per_row
    col = i % per_row
    bx = 10 + col * 110
    by = 10 + row * 40
    b = Generator.__module__ and None
    # use UI.Button class imported via Generator import
    from UI import Button as _Button
    btn = _Button((bx, by, 100, 30), func=(lambda n=name: spawn_pedal(n)), label=f"{name} (0/5)")
    top_buttons.append(btn)

def generate_next_audio_chunk():
    data = sink.get_data(CHUNK_SIZE)
    data = np.clip(data, -1.0, 1.0)
    
    mono = (data * 32767).astype(np.int16)
    _, _, mixer_channels = pg.mixer.get_init()
    if mixer_channels == 2:
        audio_array = np.column_stack((mono, mono))
    else:
        audio_array = mono

    return pg.sndarray.make_sound(audio_array), data

wire_start = None
cut_pos = None


def get_terminal(pos):
    for node in nodes:
        terminal = node.terminal_at(pos)

        if terminal:
            return terminal

    return None


def connect(a, b):
    if a.kind == "in" and b.kind == "out":
        a, b = b, a

    if a.kind != "out" or b.kind != "in":
        return

    if a.owner is b.owner:
        return

    if b in a.connections:
        return

    a.connections.append(b)
    b.connections.append(a)

run = True
clock = pg.time.Clock()
data = np.zeros(CHUNK_SIZE)
waveform_max = 100

bg = pg.transform.scale(pg.image.load("bg.png"), (1440,720)).convert()

while run:
    display.fill((25, 25, 25))
    display.blit(bg, (0,0))

    for e in pg.event.get():
        if e.type == pg.QUIT:
            run = False

        if e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
            terminal = get_terminal(e.pos)

            if terminal:
                wire_start = terminal

        elif e.type == pg.MOUSEBUTTONUP and e.button == 1:
            if wire_start:
                terminal = get_terminal(e.pos)

                if terminal:
                    connect(wire_start, terminal)

                wire_start = None

        if e.type == pg.MOUSEBUTTONDOWN and e.button == 3:
            cut_pos = e.pos

        elif e.type == pg.MOUSEBUTTONUP and e.button == 3:
            cut_pos = None

        elif e.type == pg.MOUSEMOTION and cut_pos:
            p1, p2 = pg.Vector2(cut_pos), pg.Vector2(e.pos)

            for node in nodes:
                if node.out_terminal:
                    for other in list(node.out_terminal.connections):
                        p3, p4 = node.out_terminal.pos, other.pos

                        d1 = (p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)
                        d2 = (p4.x - p3.x) * (p2.y - p3.y) - (p4.y - p3.y) * (p2.x - p3.x)
                        d3 = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
                        d4 = (p2.x - p1.x) * (p4.y - p1.y) - (p2.y - p1.y) * (p4.x - p1.x)

                        if d1 * d2 < 0 and d3 * d4 < 0:
                            node.out_terminal.connections.remove(other)
                            other.connections.remove(node.out_terminal)

            cut_pos = e.pos

        # cycle WAV when clicking the OSC area (left click release)
        if e.type == pg.MOUSEBUTTONUP and e.button == 1:
            if generator.rect.collidepoint(e.pos) and not generator.terminal_at(e.pos) and not generator.on_button.rect.collidepoint(e.pos):
                current_wav_index = (current_wav_index + 1) % len(wav_datas)
                generator.label = f"OSC ({current_wav_name()})"
                generator.phase = 0

        # reset current wav position to start on 'R'
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_r:
                generator.phase = 0

        # update top-bar spawn buttons
        for btn in top_buttons:
            btn.update(e)

        for node in nodes:
            node.update(e)

    if not audio_channel.get_queue():
        chunk, data = generate_next_audio_chunk()
        if not audio_channel.get_busy():
            audio_channel.play(chunk)
        else:
            audio_channel.queue(chunk)

    for node in nodes:
        node.draw(display)

    # draw generator progress bar (for wav playback)
    try:
        data_len = len(wav_datas[current_wav_index])
    except Exception:
        data_len = 0

    if data_len > 0:
        pos = int(generator.phase) % data_len
        pct = pos / data_len

        gb = generator.rect
        bar_height = 8
        bar_rect_bg = pg.Rect(gb.left + 6, gb.bottom - bar_height - 6, gb.width - 12, bar_height)
        pg.draw.rect(display, (40, 40, 40), bar_rect_bg)
        filled_rect = pg.Rect(bar_rect_bg.left, bar_rect_bg.top, int(bar_rect_bg.width * pct), bar_height)
        pg.draw.rect(display, (100, 200, 255), filled_rect)

    # draw top-bar spawn buttons
    for btn in top_buttons:
        btn.draw(display)

    # For now, draw the wire being dragged.
    if wire_start:
        pg.draw.line(
            display,
            (180, 180, 180),
            wire_start.pos,
            pg.mouse.get_pos(),
            3,
        )

    for node in nodes:
        if node.out_terminal:
            for other in node.out_terminal.connections:
                pg.draw.line(
                    display,
                    (180, 180, 180),
                    node.out_terminal.pos,
                    other.pos,
                    3,
                )

    # waveform draw
    for n,i in enumerate(data[1:]):
        p1 = (n-1,620 - int(data[n-1]*waveform_max))
        p2 = (n,620 - int(i*waveform_max))
        pg.draw.line(display, (155,155,255), p1, p2, 2)

    # freq domain draw (contiguous bars)
    fft_output = np.fft.rfft(data)
    magnitude = np.abs(fft_output)

    num_bars = 40
    bar_area_width = WIDTH // 2  # right half of the window (720 px)
    bar_width = max(1, bar_area_width // num_bars)
    bucket_size = max(1, len(magnitude) // num_bars)

    max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0

    for n in range(num_bars):
        start = n * bucket_size
        end = min(len(magnitude), start + bucket_size)
        if start >= end:
            peak = 0.0
        else:
            peak = np.max(magnitude[start:end])

        # scale peak to pixel height (0..HEIGHT)
        height = int((peak / max_mag) * HEIGHT) * .25

        x = 720 + n * bar_width
        rect = pg.Rect(x, HEIGHT - height, bar_width, height)
        pg.draw.rect(display, (155, 155, 255), rect)
    pg.display.flip()
    fps = clock.get_fps()
    clock.tick(120)
    # show caption normally, but reveal FPS if it drops below 100
    if fps < 100:
        pg.display.set_caption(f"Pedals Maker — FPS: {fps:.1f}")
    else:
        pg.display.set_caption("Pedals Maker")
pg.quit()
