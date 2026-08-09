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


def distortion(data):
    return np.tanh(data * 5.0).astype(np.float32) * 0.6


# Reverb has no func yet, so the pedal is a no-op passthrough until
# it's implemented (bypass behavior when func is None or disabled).


# --- Graph ---------------------------------------------------------------

generator = Generator((60, 260, 160, 120), "OSC", func=sine_source)

pedals = [
    Pedal((300, 260, 220, 120), "DISTORTION", distortion),
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
    stereo = np.column_stack((mono, mono))
    return pg.sndarray.make_sound(stereo)


wire_start = None


def get_terminal(pos):
    for node in nodes:
        terminal = node.terminal_at(pos)

        if terminal:
            return terminal

    return None


def creates_cycle(source_owner, sink_owner):
    """True if connecting source_owner -> sink_owner would close a loop
    (i.e. sink_owner is already upstream of source_owner). The pull
    model recurses via in_, so a cycle here means infinite recursion."""
    node = source_owner

    while node is not None:
        if node is sink_owner:
            return True

        node = node.in_

    return False


def connect(a, b):
    # We only allow OUT -> IN.
    if a.kind == "in" and b.kind == "out":
        a, b = b, a

    if a.kind != "out" or b.kind != "in":
        return

    # Don't connect a node to itself.
    if a.owner is b.owner:
        return

    # Don't create a loop the pull recursion can't terminate on.
    if creates_cycle(a.owner, b.owner):
        return

    a.owner.out = b.owner
    b.owner.in_ = a.owner


run = True

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

        for node in nodes:
            node.update(e)

    if not audio_channel.get_queue():
        chunk = generate_next_audio_chunk()
        if not audio_channel.get_busy():
            audio_channel.play(chunk)
        else:
            audio_channel.queue(chunk)

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
        node.draw(display)

    for node in nodes:
        if node.out:
            pg.draw.line(
                display,
                (180, 180, 180),
                node.out_terminal.pos,
                node.out.in_terminal.pos,
                3,
            )

    pg.display.flip()

pg.quit()
