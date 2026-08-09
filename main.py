import pygame as pg
import numpy as np

from UI import Generator, Pedal, Sink


SAMPLE_RATE = 44100
CHUNK_SIZE = 512
pg.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=CHUNK_SIZE)

pg.init()
pg.font.init()

WIDTH, HEIGHT = 1440, 720
display = pg.display.set_mode((WIDTH, HEIGHT))

audio_channel = pg.mixer.Channel(0)


# --- DSP functions -----------------------------------------------------
# Generators are called as func(phase, num_samples) -> float32[-1, 1].
# Pedals are called as func(data) -> float32[-1, 1] of the same shape.

def sine_source(phase, num_samples):
    t = (phase + np.arange(num_samples)) / SAMPLE_RATE
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


def distortion(data, knobs):
    threshold = 1.1 - knobs["gain"].value
    return np.clip(data, -threshold, threshold) * (1+knobs["gain"].value*9)

# Reverb has no func yet, so the pedal is a no-op passthrough until
# it's implemented (bypass behavior when func is None or disabled).


# --- Graph ---------------------------------------------------------------

generator = Generator((60, 260, 160, 120), "OSC", func=sine_source)

pedals = [
    Pedal((300, 60, 220, 120), "PITCH SHIFT", distortion, knobs={"gain": (0.0, 1.0, 1.0)}),
    Pedal((300, 260, 220, 120), "DISTORTION", distortion, knobs={"gain": (0.0, 1.0, 1.0)}),
    Pedal((600, 260, 220, 120), "REVERB"),
]

sink = Sink((900, 260, 140, 120), "OUT")

# Everything that needs events/drawing/terminal lookups, source to sink.
nodes = [generator, *pedals, sink]


def generate_next_audio_chunk():
    # This single call is what kicks off the pull: the sink asks its
    # in_ for data, which asks ITS in_ for data, all the way back to
    # the generator, which is the only node with no in_ and actually
    # produces samples. Everything on the way back out gets processed.
    data = sink.get_data(CHUNK_SIZE)
    data = np.clip(data, -1.0, 1.0)
    mono = (data * 32767).astype(np.int16)
    #stereo = np.column_stack((mono, mono))
    return pg.sndarray.make_sound(mono), data


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
