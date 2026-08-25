import numpy as np

LOOP_SAMPLE_RATE = 48000
LOOP_BUFFER = np.zeros(LOOP_SAMPLE_RATE, dtype=np.float32)
loop_buffer_pos = 0
loop_play_pos = 0
loop_recording = True
loop_done = False

def reset_loop():
    global loop_buffer_pos, loop_play_pos, loop_recording, loop_done

    LOOP_BUFFER.fill(0.0)
    loop_buffer_pos = 0
    loop_play_pos = 0
    loop_recording = True
    loop_done = False

def loop(data, knobs):
    global loop_buffer_pos, loop_play_pos, loop_recording, loop_done

    if loop_recording:
        end = min(loop_buffer_pos + len(data), len(LOOP_BUFFER))
        captured = end - loop_buffer_pos
        LOOP_BUFFER[loop_buffer_pos:end] = data[:captured]
        output = np.empty(len(data), dtype=np.float32)
        output[:captured] = data[:captured]
        loop_buffer_pos = end

        if loop_buffer_pos >= len(LOOP_BUFFER):
            loop_recording = False
            loop_done = True
            loop_play_pos = 0

        if captured < len(data):
            output[captured:] = loop(data[captured:], knobs)

    loop_length = int(knobs["length"].value * LOOP_SAMPLE_RATE)
    loop_length = max(1, min(len(LOOP_BUFFER), loop_length))
    positions = (loop_play_pos + np.arange(len(data))) % loop_length
    loop_play_pos = (loop_play_pos + len(data)) % loop_length
    return LOOP_BUFFER[positions]

loop.reset = reset_loop

mix_phase = 0

def mix(data, knobs):
    global mix_phase

    frequency = knobs["frequency"].value * 1000.0
    bias = knobs["bias"].value
    positions = mix_phase + np.arange(len(data))
    carrier = np.sin(2.0 * np.pi * frequency * positions / LOOP_SAMPLE_RATE) + bias
    mix_phase += len(data)
    return (data * carrier).astype(np.float32)

dumb_shifter_previous_sample = 0.0

def dumb_shifter(data, knobs):
    global dumb_shifter_previous_sample

    ratio = knobs["shift"].value
    output_length = max(1, int(len(data) / ratio))
    source_positions = np.arange(output_length) * ratio
    source = np.arange(len(data))
    stretched = np.interp(source_positions, source, data)

    output = np.zeros(len(data), dtype=np.float32)
    output[:min(len(output), len(stretched))] = stretched[:len(output)]

    fade_length = min(32, len(output))
    fade = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
    output[:fade_length] = (
        dumb_shifter_previous_sample * (1.0 - fade)
        + output[:fade_length] * fade
    )
    dumb_shifter_previous_sample = float(output[-1])
    return output

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
    x = x * (1+knobs["gain"].value*190)
    return x/ (1 + abs(x)) * .5

env_state = {"level": 0.0}

def sustain_boost(x, knobs, sr=48000):
    """Auto-gain: keeps decaying signal 'hot' by boosting quiet parts."""
    env = envelope_follower(x, sr=sr)
    threshold = knobs.get("threshold", type("K", (), {"value": 0.05})()).value
    ratio = knobs.get("comp_ratio", type("K", (), {"value": 8})()).value

    # makeup gain grows as envelope falls below threshold, capped to avoid noise blowup
    gain = np.ones_like(env)
    below = env > 1e-6
    gain[below] = np.clip((threshold / (env[below] + 1e-6)) ** (1 - 1/ratio), 1.0, 40.0)

    return x * gain

gate_state = {"level": 0.0, "gate_gain": 1.0}

def envelope_follower(x, attack=0.001, release=0.05, sr=48000, state=None):
    a = np.exp(-1.0 / (sr * attack))
    r = np.exp(-1.0 / (sr * release))
    env = np.zeros_like(x)
    level = state["level"]
    for i in range(len(x)):
        rect = abs(x[i])
        coeff = a if rect > level else r
        level = coeff * level + (1 - coeff) * rect
        env[i] = level
    state["level"] = level
    return env

def fuzz_gate(x, knobs, sr=48000):
    """Chokes the tail hard once signal drops below threshold, like a starved fuzz circuit."""
    threshold = knobs["gate_thresh"].value      # e.g. 0.02-0.1, tune by ear
    choke_speed = knobs["gate_speed"].value      # how fast it slams shut, in ms-ish

    env = envelope_follower(x, attack=0.0005, release=0.03, sr=sr, state=gate_state)

    # smoothed on/off gate gain, itself has attack/release so it doesn't click
    gate_a = np.exp(-1.0 / (sr * 0.0005))                     # fast open
    gate_r = np.exp(-1.0 / (sr * (0.005 + choke_speed*0.05))) # tunable close speed

    out = np.zeros_like(x)
    g = gate_state["gate_gain"]
    for i in range(len(x)):
        target = 1.0 if env[i] > threshold else 0.0
        coeff = gate_a if target > g else gate_r
        g = coeff * g + (1 - coeff) * target
        out[i] = x[i] * g
    gate_state["gate_gain"] = g

    return out

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

x1 = 0.0
y1 = 0.0
R = 0.995   # closer to 1 = slower / lower cutoff

def pitch_shift(x, knobs):
    global x1
    global y1
    global R
    reps = knobs["shift"].value
    for _ in range(int(reps)):
        x = (abs(x) - 0.25) * 2.0

    # DC blocker
    y = x - x1 + R * y1
    x1 = x
    y1 = y
    return y

GRAIN_SIZE = 1024
HISTORY_SIZE = 2048
pitch_history = np.zeros(HISTORY_SIZE, dtype=np.float32)

def better_pitch_shift(data, knobs):
    print("grain")

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

# simple state for low-pass filter (shared across instances)
lowpass_state = {"y": 0.0}


def low_pass(x, knobs, sr=48000):
    """First-order low-pass filter.

    `knobs["cutoff"].value` expected in 0..1 mapping to 0..8000 Hz.
    A cutoff of 0 produces near-silence (only DC passes); a cutoff of 1
    sets cutoff to 8 kHz.
    """
    global lowpass_state

    fc = 0.0
    if knobs and isinstance(knobs, dict) and "cutoff" in knobs:
        fc = knobs["cutoff"].value * 8000.0

    if fc <= 0.0:
        # effectively remove high-frequency content --> near-zero output
        return np.zeros_like(x)

    # discrete-time smoothing coefficient (one-pole)
    # a = 1 - exp(-2*pi*fc / sr)
    a = 1.0 - np.exp(-2.0 * np.pi * fc / float(sr))

    y = np.empty_like(x)
    yp = lowpass_state.get("y", 0.0)
    for i in range(len(x)):
        yp = a * x[i] + (1.0 - a) * yp
        y[i] = yp

    lowpass_state["y"] = yp
    return y.astype(np.float32)


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

    if peak_val < 0.1:
        return None  # too noisy/inharmonic to trust (e.g. after heavy distortion)

    return min_lag + peak_idx


def td_psola_pitch_shift(data, knobs, sr=48000):
    print("psola")
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


# simple reverb: add delayed copy of the signal (past) back onto itself
# uses a single-sample delay at most; preserves last_sample across chunks
MAX_REVERB_DELAY = 48000  # maximum delay in samples (1 second at 48k)
reverb_state = {"history": np.zeros(MAX_REVERB_DELAY, dtype=np.float32), "filled": 0}

def reverb(x, knobs):
    """Simple reverb/comb: add a delayed copy of the signal back onto itself.

    knobs["delay"].value is expected in 0..1 and maps to 0..MAX_REVERB_DELAY samples.
    A history buffer preserves past samples so delays larger than the chunk
    length are supported without truncation.
    """
    global reverb_state

    if knobs and isinstance(knobs, dict) and "delay" in knobs:
        delay = int(round(knobs["delay"].value * MAX_REVERB_DELAY))
        delay = max(0, min(MAX_REVERB_DELAY, delay))
    else:
        delay = 0

    if delay == 0 or len(x) == 0:
        # update history with current chunk and pass through
        hist = reverb_state["history"]
        filled = reverb_state["filled"]
        buf = np.concatenate([hist[-filled:] if filled > 0 else np.array([], dtype=np.float32), x])
        # keep last MAX_REVERB_DELAY samples
        new_hist = buf[-MAX_REVERB_DELAY:]
        # store at end-aligned
        reverb_state["history"][-len(new_hist):] = new_hist
        reverb_state["filled"] = min(MAX_REVERB_DELAY, filled + len(x))
        return x

    hist = reverb_state["history"]
    filled = reverb_state["filled"]

    hist_tail = hist[-filled:] if filled > 0 else np.array([], dtype=np.float32)
    buf = np.concatenate([hist_tail, x])
    H = len(hist_tail)

    delayed = np.zeros_like(x)
    for i in range(len(x)):
        idx = H + i - delay
        if 0 <= idx < len(buf):
            delayed[i] = buf[idx]
        else:
            delayed[i] = 0.0

    out = x + delayed

    # update history to include this chunk
    new_hist = buf[-MAX_REVERB_DELAY:]
    reverb_state["history"][-len(new_hist):] = new_hist
    reverb_state["filled"] = min(MAX_REVERB_DELAY, filled + len(x))

    return np.clip(out, -1.0, 1.0)


# --- Normalizer pedal ---------------------------------------------------
# Keeps a rolling record of recent chunk peaks (few seconds) and scales
# the current chunk by the rolling max so output stays in -1..1 even after
# aggressive gain/fuzz. The knob `window` controls how many seconds to
# look back for the rolling max.
normalizer_state = {"history": [], "window_secs": 2.0}


def normalizer(x, knobs, sr=48000):
    """Normalize `x` by the rolling max over the last `window` seconds.

    knobs['window'].value expected in seconds (0.5..10 by default).
    This stores one peak value per chunk and uses the maximum of the
    recent peaks as the normalizer denominator. If the rolling max is
    near zero the function is a no-op.
    """
    global normalizer_state

    if x is None or len(x) == 0:
        return x

    # read knob or fallback
    window = normalizer_state.get("window_secs", 2.0)
    if knobs and isinstance(knobs, dict) and "window" in knobs:
        try:
            window = float(knobs["window"].value)
        except Exception:
            pass

    chunk_len = len(x)
    if chunk_len <= 0:
        return x

    # how many chunks fit into the window
    chunks_needed = max(1, int(round(window * float(sr) / float(chunk_len))))

    # compute this chunk's mean absolute deviation from 0 (perceived loudness proxy)
    mean_abs = float(np.mean(np.abs(x))) if np.any(x) else 0.0

    hist = normalizer_state.get("history", [])
    hist.append(mean_abs)
    if len(hist) > chunks_needed:
        hist = hist[-chunks_needed:]
    normalizer_state["history"] = hist

    # use the average of recent mean-abs values as the rolling loudness
    rolling_level = float(np.mean(hist)) if hist else 0.0

    eps = 1e-9
    if rolling_level < eps:
        # near silence: if user explicitly requests zero level, obey; otherwise passthrough
        # read target level knob if present
        target = 1.0
        if knobs and isinstance(knobs, dict) and "level" in knobs:
            try:
                target = float(knobs["level"].value)
            except Exception:
                pass
        if target <= 0.0:
            return np.zeros_like(x, dtype=np.float32)
        return x.astype(np.float32)

    # read target level knob (0.0..1.0), default to 1.0
    target_level = 1.0
    if knobs and isinstance(knobs, dict) and "level" in knobs:
        try:
            target_level = float(knobs["level"].value)
        except Exception:
            pass

    # scale so the rolling mean-abs maps to the knob-specified level
    scale = target_level / (rolling_level + eps)
    out = x * scale
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32)

def bitcrush(x, knobs):
    n = int(knobs["factor"].value)
    x = np.repeat(x[::n], n)
    return x

import numpy as np
from scipy.signal.windows import tukey

buffer = np.zeros(2048)
ola = np.zeros(2048)
counter = 0

def fft_pitch_shift(x, knobs):
    global buffer, ola, counter

    n = float(knobs["factor"].value)
    factor = 2.0 ** n

    hop = len(x) * 2
    frame_size = len(buffer)

    # Add input to rolling buffer
    buffer[:-len(x)] = buffer[len(x):]
    buffer[-len(x):] = x

    counter += len(x)

    if counter >= hop:
        counter -= hop

        # Less aggressive than Hann
        window = tukey(frame_size, alpha=0.25)

        spectrum = np.fft.rfft(buffer * window)

        # Move frequency bins by `factor`
        shifted = np.zeros_like(spectrum)

        src = np.arange(len(spectrum))
        dst = src * factor

        valid = dst < len(shifted) - 1

        src = src[valid]
        dst = dst[valid]

        # Linear interpolation between destination bins
        lo = np.floor(dst).astype(int)
        hi = lo + 1
        frac = dst - lo

        shifted[lo] += spectrum[src] * (1.0 - frac)
        shifted[hi] += spectrum[src] * frac

        frame = np.fft.irfft(shifted, n=frame_size)

        # Synthesis window
        frame *= window

        # Overlap-add
        ola += frame

    # Emit exactly the input chunk size
    out = ola[:len(x)].copy()

    # Advance OLA buffer
    ola[:-len(x)] = ola[len(x):]
    ola[-len(x):] = 0.0

    return out