import pygame as pg
import numpy as np
from UI import Pedal

SAMPLE_RATE = 44100
CHUNK_SIZE = 512
pg.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=CHUNK_SIZE)

pg.init()
pg.font.init()

WIDTH, HEIGHT = 1440, 720
display = pg.display.set_mode((WIDTH, HEIGHT))

audio_channel = pg.mixer.Channel(0)

phase = 0.0
frequency = 440.0


def distortion():
    print("distortion")


pedals = [
    Pedal((200, 250, 220, 140), "DISTORTION", distortion),
    Pedal((600, 250, 220, 140), "REVERB"),
]


def generate_next_audio_chunk():
    global phase
    t = (phase + np.arange(CHUNK_SIZE)) / SAMPLE_RATE
    mono = (np.sin(2 * np.pi * frequency * t) * 15000).astype(np.int16)
    phase += CHUNK_SIZE
    stereo = np.column_stack((mono, mono))
    return pg.sndarray.make_sound(stereo)

wire_start = None


def get_terminal(pos):
    for pedal in pedals:
        terminal = pedal.terminal_at(pos)

        if terminal:
            return terminal

    return None


def connect(a, b):
    # We only allow OUT -> IN.
    if a.kind == "in" and b.kind == "out":
        a, b = b, a

    if a.kind != "out" or b.kind != "in":
        return

    # Don't connect a pedal to itself.
    if a.pedal is b.pedal:
        return

    a.pedal.out = b.pedal
    b.pedal.in_ = a.pedal


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

        for pedal in pedals:
            pedal.update(e)

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

    for pedal in pedals:
        pedal.draw(display)

    for pedal in pedals:
        if pedal.out:
            pg.draw.line(
                display,
                (180, 180, 180),
                pedal.out_terminal.pos,
                pedal.out.in_terminal.pos,
                3,
            )

    pg.display.flip()

pg.quit()