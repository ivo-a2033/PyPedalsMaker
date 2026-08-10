import pygame as pg
import numpy as np
from scipy.io import wavfile

from UI import Generator, Pedal, Sink


SAMPLE_RATE = 48000
CHUNK_SIZE = 512
pg.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=CHUNK_SIZE)

pg.init()
pg.font.init()

WIDTH, HEIGHT = 1440, 720
display = pg.display.set_mode((WIDTH, HEIGHT))

audio_channel = pg.mixer.Channel(0)

# importing a file for testing
sample_rate, wav_data = wavfile.read("guitar1.wav")
wav_data = wav_data.astype(np.float32) / np.max(np.abs(wav_data))

# --- DSP functions -----------------------------------------------------
# Generators are called as func(phase, num_samples) -> float32[-1, 1].
# Pedals are called as func(data) -> float32[-1, 1] of the same shape.

def sine_source(phase, num_samples):
    t = (phase + np.arange(num_samples)) / SAMPLE_RATE
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

def get_wav(pos, num_samples):
    pos = int(pos) % len(wav_data)
    end = pos + num_samples

    if end <= len(wav_data):
        chunk = wav_data[pos:end]
    else:
        # wraps past the end -> stitch the tail + the head together
        first_part = wav_data[pos:]
        remaining = num_samples - len(first_part)
        second_part = wav_data[:remaining]
        chunk = np.concatenate([first_part, second_part])

    if chunk.ndim > 1:
        chunk = chunk[:, 0]

    return chunk.astype(np.float32) 


def distortion(data, knobs):
    data = data * (1+knobs["gain"].value*90)
    return np.clip(data, -1, 1) 

def overdrive(x, knobs):
    x = x * (1+knobs["gain"].value*90)
    return np.tanh(x)

def fuzz(x, knobs):
    x = x * (1+knobs["gain"].value*90)
    return np.sign(x) * (1 - np.exp(-np.abs(x)))

def fuzz2(x, knobs):
    x = x * (1+knobs["gain"].value*90)
    return x/ (1 + abs(x))

def diode_clip(x, knobs):
    x =  x * (1+knobs["gain"].value*90)
    vt, vth = knobs["vt"].value, knobs["vth"].value
    # vt = "knee softness", vth = voltage where it starts clipping hard
    ax = np.abs(x)
    lin = ax < vth - vt
    quad = (ax >= vth - vt) & (ax < vth + vt)
    hard = ax >= vth + vt

    out = np.zeros_like(x)
    out[lin] = x[lin]
    # quadratic knee region
    sign = np.sign(x[quad])
    out[quad] = sign * (ax[quad] - (ax[quad] - (vth - vt))**2 / (4*vt))
    out[hard] = np.sign(x[hard]) * vth

    return out

def pitch_shift(x, knobs):
    return abs(x)

GRAIN_SIZE = 1024
HISTORY_SIZE = 2048
pitch_history = np.zeros(HISTORY_SIZE, dtype=np.float32)

def better_pitch_shift(data, knobs):
    global pitch_history

    ratio = 0.5 + knobs["shift"].value * 1.5  # 0.5x (down) to 2x (up)

    if not hasattr(better_pitch_shift, "read_pos"):
        better_pitch_shift.read_pos = 0.0

    buf = np.concatenate([pitch_history, data])
    H = len(pitch_history)
    g = GRAIN_SIZE
    out = np.zeros(len(data), dtype=np.float32)

    for i in range(len(data)):
        write_idx = H + i

        r1 = better_pitch_shift.read_pos % g
        r2 = (r1 + g / 2) % g

        idx1 = int(np.clip(write_idx - g + r1, 0, len(buf) - 1))
        idx2 = int(np.clip(write_idx - g + r2, 0, len(buf) - 1))

        w1 = 0.5 - 0.5 * np.cos(2 * np.pi * r1 / g)
        w2 = 0.5 - 0.5 * np.cos(2 * np.pi * r2 / g)

        out[i] = buf[idx1] * w1 + buf[idx2] * w2
        better_pitch_shift.read_pos += ratio

    # update history: append this chunk's dry input, keep last HISTORY_SIZE samples
    pitch_history = np.concatenate([pitch_history, data])[-HISTORY_SIZE:]

    return out.astype(np.float32)

PSOLA_HISTORY_SIZE = 4096
psola_history = np.zeros(PSOLA_HISTORY_SIZE, dtype=np.float32)


def estimate_pitch_period(buf, sr, fmin=70, fmax=800): # helper function to psola
    """Autocorrelation pitch period estimate, in samples. Returns None if unvoiced/noisy."""
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    max_lag = min(max_lag, len(buf) - 1)
    if max_lag <= min_lag:
        return None

    seg = buf[-max_lag * 3:] if len(buf) > max_lag * 3 else buf
    seg = seg - np.mean(seg)

    corr = np.correlate(seg, seg, mode="full")
    mid = len(corr) // 2
    corr = corr[mid:mid + max_lag + 1]

    if corr[0] <= 1e-8:
        return None  # near-silence

    corr_norm = corr / corr[0]
    window = corr_norm[min_lag:max_lag + 1]
    if len(window) == 0:
        return None

    peak_idx = int(np.argmax(window))
    peak_val = window[peak_idx]

    if peak_val < 0.3:
        return None  # too noisy/inharmonic to trust (e.g. after heavy distortion)

    return min_lag + peak_idx


def td_psola_pitch_shift(data, knobs, sr=48000):
    global psola_history

    ratio = 0.5 + knobs["shift"].value * 1.5  # 0.5x (down) to 2x (up)

    if not hasattr(td_psola_pitch_shift, "period"):
        td_psola_pitch_shift.period = int(sr / 150)  # fallback guess, ~150Hz
    if not hasattr(td_psola_pitch_shift, "read_pos"):
        td_psola_pitch_shift.read_pos = float(len(psola_history))

    # Estimate this chunk's pitch period from history (causal, no lookahead into `data`)
    detected = estimate_pitch_period(psola_history, sr)
    if detected is not None:
        detected = int(np.clip(detected, sr // 800, sr // 70))
        # smooth so period doesn't jump wildly chunk to chunk
        td_psola_pitch_shift.period = int(0.7 * td_psola_pitch_shift.period + 0.3 * detected)

    T = max(td_psola_pitch_shift.period, 32)
    grain = 2 * T  # PSOLA convention: window spans ~2 pitch periods

    buf = np.concatenate([psola_history, data]).astype(np.float32)
    H = len(psola_history)
    out = np.zeros(len(data), dtype=np.float32)

    for i in range(len(data)):
        write_idx = H + i
        rp = td_psola_pitch_shift.read_pos

        r1 = rp % grain
        r2 = (r1 + grain / 2) % grain  # second tap, one pitch period out of phase

        idx1 = int(np.clip(write_idx - grain + r1, 0, len(buf) - 1))
        idx2 = int(np.clip(write_idx - grain + r2, 0, len(buf) - 1))

        w1 = 0.5 - 0.5 * np.cos(2 * np.pi * r1 / grain)
        w2 = 0.5 - 0.5 * np.cos(2 * np.pi * r2 / grain)

        out[i] = buf[idx1] * w1 + buf[idx2] * w2
        td_psola_pitch_shift.read_pos += ratio

    psola_history = np.concatenate([psola_history, data])[-PSOLA_HISTORY_SIZE:]
    return out.astype(np.float32)

# --- Graph ---------------------------------------------------------------

generator = Generator((60, 260, 160, 120), "OSC", func=get_wav)

pedals = [
    Pedal((300, 60, 220, 120), "OVERDRIVE", overdrive, knobs={"gain": (0.0, 1.0, 1.0)}),
    Pedal((300, 460, 220, 120), "FUZZ", fuzz2, knobs={"gain": (0.0, 1.0, 1.0)}),
    Pedal((500, 460, 220, 120), "PITCH SHIFT", td_psola_pitch_shift, knobs={"shift": (0.0, 1.0, 1.0)}),
    Pedal((300, 260, 220, 120), "DISTORTION", distortion, knobs={"gain": (0.0, 1.0, 1.0)}),
    Pedal((600, 260, 220, 120), "REVERB"),
]

sink = Sink((900, 260, 140, 120), "OUT")

# Everything that needs events/drawing/terminal lookups, source to sink.
nodes = [generator, *pedals, sink]

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
    # We only allow OUT -> IN.
    if a.kind == "in" and b.kind == "out":
        a, b = b, a

    if a.kind != "out" or b.kind != "in":
        return

    # Don't connect a node to itself.
    if a.owner is b.owner:
        return

    a.connections.append(b)
    b.connections.append(a)


run = True
clock = pg.time.Clock()
data = np.zeros(CHUNK_SIZE)
waveform_max = 100

while run:
    display.fill((25, 25, 25))

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

    # freq domain draw
    fft_output = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/SAMPLE_RATE)
    magnitude = np.abs(fft_output)
    pos_mask = freqs >= 0 # no need for neg freqs

    pos_freqs = freqs[pos_mask]
    pos_magnitude = magnitude[pos_mask]

    for n,i in enumerate(pos_freqs):
        pg.draw.line(display, (155,155,255), (n+720,720), (n+720,720-int(pos_magnitude[n])), 1)

    pg.display.flip()
    clock.tick(120)
pg.quit()
