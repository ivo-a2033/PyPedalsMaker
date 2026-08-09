import math
import numpy as np
import pygame as pg


class Button:
    def __init__(
        self,
        rect,
        func,
        label="",
        color=(80, 80, 80),
        pressed_color=(120, 120, 120),
    ):
        self.rect = pg.Rect(rect)
        self.func = func
        self.label = label

        self.color = color
        self.pressed_color = pressed_color

        self.pressed = False
        self.font = pg.font.Font(None, 16)

    def update(self, event):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True

        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            if self.pressed:
                if self.rect.collidepoint(event.pos):
                    self.func()

                self.pressed = False

    def draw(self, surface):
        color = self.pressed_color if self.pressed else self.color

        pg.draw.rect(surface, color, self.rect)

        if self.label:
            text = self.font.render(self.label, True, (255, 255, 255))
            text_rect = text.get_rect(center=self.rect.center)
            surface.blit(text, text_rect)


class Knob(Button):
    """A glorified Button: no click action, its value is set by scrolling
    the mouse wheel while hovering over it."""

    def __init__(self, rect, label="", value=0.5, min_value=0.0, max_value=1.0, step=0.05):
        super().__init__(rect, func=lambda: None, label=label)
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.hover_font = pg.font.Font(None, 22)

    def update(self, event):
        if event.type == pg.MOUSEWHEEL and self.rect.collidepoint(pg.mouse.get_pos()):
            self.value = min(self.max_value, max(self.min_value, self.value + event.y * self.step))

    def draw(self, surface):
        center = self.rect.center
        radius = min(self.rect.width, self.rect.height) // 2
        hovered = self.rect.collidepoint(pg.mouse.get_pos())

        if hovered:
            pg.draw.circle(surface, (180, 180, 180), center, radius + 8, 1)

        pg.draw.circle(surface, self.color, center, radius)

        t = (self.value - self.min_value) / (self.max_value - self.min_value)
        angle = math.radians(-135 + 270 * t)
        tip = (center[0] + radius * math.sin(angle), center[1] - radius * math.cos(angle))
        pg.draw.line(surface, (255, 255, 255), center, tip, 2)

        if hovered:
            text = self.hover_font.render(f"{self.label} {self.value:.2f}", True, (255, 255, 255))
            surface.blit(text, text.get_rect(center=(center[0], center[1] + radius + 16)))


class Terminal:
    """A connection point on a GraphNode (an in_ or out socket)."""

    def __init__(self, owner, kind):
        self.owner = owner
        self.kind = kind  # "in" or "out"
        self.connections = []

    @property
    def pos(self):
        if self.kind == "in":
            return pg.Vector2(
                self.owner.rect.left,
                self.owner.rect.centery,
            )

        return pg.Vector2(
            self.owner.rect.right,
            self.owner.rect.centery,
        )

    def contains(self, pos):
        return self.pos.distance_to(pos) <= 16

    def draw(self, surface):
        if self.contains(pg.mouse.get_pos()):
            pg.draw.circle(surface, (180, 180, 180), self.pos, 16, 2)
        pg.draw.circle(surface, (180, 180, 180), self.pos, 6)


class GraphNode:
    """
    Shared base for anything that sits in the pedalboard graph and can
    be dragged around: sources, effects, and the final output.

    Audio flows via `get_data(num_samples)`: a node PULLS its input by
    calling `get_data(num_samples)` on every node connected to its
    in_terminal and summing the results, does whatever it does to
    that data, and returns the result.
    A source node overrides this to generate instead of pulling. A
    sink just pulls and hands the result off to the audio backend.
    This is called once per audio chunk, from the sink end, and the
    call recurses upstream through the whole chain on the spot.
    """

    def __init__(self, rect, label=""):
        self.rect = pg.Rect(rect)
        self.label = label
        self.font = pg.font.Font(None, 28)

        self.dragging = False
        self.drag_offset = (0, 0)

        # Subclasses that don't have a given side leave it as None
        # (a Generator has no in_terminal, a Sink has no out_terminal).
        self.in_terminal = None
        self.out_terminal = None

    def terminal_at(self, pos):
        for terminal in (self.in_terminal, self.out_terminal):
            if terminal and terminal.contains(pos):
                return terminal

        return None

    def _update_control_positions(self):
        """Overridden by subclasses with buttons/knobs to reposition."""

    def handle_drag(self, event):
        """Shared click-and-drag behavior. Subclasses call this from update()."""
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            # Don't start dragging if we clicked a terminal instead.
            if self.terminal_at(event.pos):
                return

            if self.rect.collidepoint(event.pos):
                self.dragging = True

                self.drag_offset = (
                    self.rect.x - event.pos[0],
                    self.rect.y - event.pos[1],
                )

        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

        elif event.type == pg.MOUSEMOTION and self.dragging:
            self.rect.x = event.pos[0] + self.drag_offset[0]
            self.rect.y = event.pos[1] + self.drag_offset[1]

            self._update_control_positions()

    def get_data(self, num_samples):
        """
        Default pull behavior: ask upstream, pass through unchanged.
        Sources (Generator) override this to generate instead of
        pulling. Effects (Pedal) override it to process what they pull.
        """
        if self.in_terminal and self.in_terminal.connections:
            return sum(t.owner.get_data(num_samples) for t in self.in_terminal.connections)

        return np.zeros(num_samples, dtype=np.float32)

    def draw_label_and_terminals(self, surface):
        if self.label:
            text = self.font.render(self.label, True, (255, 255, 255))
            surface.blit(text, (self.rect.x + 10, self.rect.y + 8))

        if self.in_terminal:
            self.in_terminal.draw(surface)

        if self.out_terminal:
            self.out_terminal.draw(surface)


class Generator(GraphNode):
    """
    A sound source: no input, just an out terminal. `func`, if given,
    is called as `func(phase, num_samples)` and must return
    `num_samples` of float32 audio in [-1, 1]. Falls back to a plain
    sine oscillator at `freq`. `phase` is the running sample count,
    handed to `func` so it can keep waveforms continuous across chunks.
    """

    def __init__(self, rect, label="", func=None, freq=440.0, sample_rate=44100):
        super().__init__(rect, label)

        self.func = func
        self.freq = freq
        self.sample_rate = sample_rate
        self.phase = 0.0

        self.enabled = True

        self.on_button = Button(
            (0, 0, 50, 30),
            self.toggle,
            "ON",
            color=(60, 100, 60),
            pressed_color=(100, 160, 100),
        )

        self.out_terminal = Terminal(self, "out")

        self._update_control_positions()

    def toggle(self):
        self.enabled = not self.enabled

    def _update_control_positions(self):
        self.on_button.rect.topleft = (
            self.rect.right - 60,
            self.rect.bottom - 40,
        )

    def update(self, event):
        self.handle_drag(event)
        self.on_button.update(event)

    def get_data(self, num_samples):
        if not self.enabled:
            self.phase += num_samples
            return np.zeros(num_samples, dtype=np.float32)

        if self.func:
            data = self.func(self.phase, num_samples)
        else:
            t = (self.phase + np.arange(num_samples)) / self.sample_rate
            data = np.sin(2 * np.pi * self.freq * t).astype(np.float32)

        self.phase += num_samples
        return data

    def draw(self, surface):
        pg.draw.rect(surface, (50, 50, 50), self.rect)
        pg.draw.rect(surface, (120, 120, 120), self.rect, 2)

        self.draw_label_and_terminals(surface)
        self.on_button.draw(surface)

        pos = pg.Vector2(self.rect.right - 10, self.rect.top + 10)
        color = (100, 255, 100) if self.enabled else (0, 25, 0)
        pg.draw.circle(surface, color, pos, 5)


class Pedal(GraphNode):
    """
    A movable pedal UI element. Represents one effect in the signal
    chain. `func`, if given, is called as `func(data)` where `data` is
    the float32 [-1, 1] audio pulled from upstream, and must return
    audio of the same shape. When disabled, the pedal is a bypass:
    input passes through unchanged.

    Attributes:
        rect: Position and size of the pedal.
        label: Text displayed on the pedal.
        func: Effect-processing function, func(data) -> data.
        in_: Upstream node (Generator or Pedal), or None.
        out: Downstream node (Pedal or Sink), or None.
        enabled: Whether the effect is applied (False = bypass).
    """

    def __init__(self, rect, label="", func=None, knobs=None):
        super().__init__(rect, label)

        self.func = func
        self.enabled = True

        self.on_button = Button(
            (0, 0, 50, 30),
            self.toggle,
            "ON",
            color=(60, 100, 60),
            pressed_color=(100, 160, 100),
        )

        # knobs: {name: (min_value, default_value, max_value)}
        self.knobs = {
            name: Knob((0, 0, 60, 24), name, default, lo, hi)
            for name, (lo, default, hi) in (knobs or {}).items()
        }

        self.in_terminal = Terminal(self, "in")
        self.out_terminal = Terminal(self, "out")

        self._update_control_positions()

    def toggle(self):
        self.enabled = not self.enabled

    def _update_control_positions(self):
        self.on_button.rect.topleft = (
            self.rect.right - 60,
            self.rect.bottom - 40,
        )

        for i, knob in enumerate(self.knobs.values()):
            knob.rect.topleft = (self.rect.left + 10, self.rect.top + 40 + i * 26)

    def update(self, event):
        self.handle_drag(event)
        self.on_button.update(event)

        for knob in self.knobs.values():
            knob.update(event)

    def get_data(self, num_samples):
        if self.in_terminal.connections:
            data = sum(t.owner.get_data(num_samples) for t in self.in_terminal.connections)
        else:
            data = np.zeros(num_samples, dtype=np.float32)

        if self.enabled and self.func:
            data = self.func(data, self.knobs) if self.knobs else self.func(data)

        return data

    def draw(self, surface):
        pg.draw.rect(surface, (50, 50, 50), self.rect)
        pg.draw.rect(surface, (120, 120, 120), self.rect, 2)

        self.draw_label_and_terminals(surface)
        self.on_button.draw(surface)

        for knob in self.knobs.values():
            knob.draw(surface)

        pos = pg.Vector2(self.rect.right - 10, self.rect.top + 10)
        color = (100, 255, 100) if self.enabled else (0, 25, 0)
        pg.draw.circle(surface, color, pos, 5)


class Sink(GraphNode):
    """
    The final output node: no out terminal, just an in. Calling
    `get_data` here is what kicks off the whole pull chain — call this
    once per audio chunk in place of the old generate_next_audio_chunk.
    """

    def __init__(self, rect, label="OUT"):
        super().__init__(rect, label)
        self.in_terminal = Terminal(self, "in")

    def update(self, event):
        self.handle_drag(event)

    def draw(self, surface):
        pg.draw.rect(surface, (45, 45, 60), self.rect)
        pg.draw.rect(surface, (140, 140, 180), self.rect, 2)
        self.draw_label_and_terminals(surface)