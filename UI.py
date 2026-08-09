import pygame as pg


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


class Terminal:
    def __init__(self, pedal, kind):
        self.pedal = pedal
        self.kind = kind  # "in" or "out"

    @property
    def pos(self):
        if self.kind == "in":
            return pg.Vector2(
                self.pedal.rect.left,
                self.pedal.rect.centery,
            )

        return pg.Vector2(
            self.pedal.rect.right,
            self.pedal.rect.centery,
        )

    def contains(self, pos):
        return self.pos.distance_to(pos) <= 16

    def draw(self, surface):
        pg.draw.circle(surface, (180, 180, 180), self.pos, 6)

class Pedal:
    """
    A movable pedal UI element.

    Represents one effect in the signal chain. The pedal owns its
    UI controls, while `func` will eventually perform the effect's
    audio processing.

    Attributes:
        rect: Position and size of the pedal.
        label: Text displayed on the pedal.
        func: Effect-processing function.
        in_: Incoming pedal connection.
        out: Outgoing pedal connection.
        enabled: Whether the pedal is active.
    """

    def __init__(self, rect, label="", func=None):
        self.rect = pg.Rect(rect)

        self.label = label
        self.func = func

        self.in_ = None
        self.out = None

        self.enabled = True

        self.dragging = False
        self.drag_offset = (0, 0)

        self.font = pg.font.Font(None, 28)

        self.on_button = Button(
            (0, 0, 50, 30),
            self.toggle,
            "ON",
            color=(60, 100, 60),
            pressed_color=(100, 160, 100),
        )

        self.in_terminal = Terminal(self, "in")
        self.out_terminal = Terminal(self, "out")

        self._update_control_positions()

    def toggle(self):
        self.enabled = not self.enabled

    def do_func(self):
        if self.func:
            self.func()

    def _update_control_positions(self):
        self.on_button.rect.topleft = (
            self.rect.right - 60,
            self.rect.bottom - 40,
        )

    def terminal_at(self, pos):
        if self.in_terminal.contains(pos):
            return self.in_terminal

        if self.out_terminal.contains(pos):
            return self.out_terminal

        return None

    def update(self, event):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            # Don't start dragging the pedal if we clicked a terminal.
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

        self.on_button.update(event)

    def draw(self, surface):
        pg.draw.rect(surface, (50, 50, 50), self.rect)
        pg.draw.rect(surface, (120, 120, 120), self.rect, 2)

        if self.label:
            text = self.font.render(self.label, True, (255, 255, 255))
            surface.blit(
                text,
                (self.rect.x + 10, self.rect.y + 8),
            )

        self.on_button.draw(surface)

        pos = pg.Vector2(
            self.rect.right - 10,
            self.rect.top + 10,
        )

        if self.enabled:
            pg.draw.circle(surface, (100, 255, 100), pos, 5)
        else:
            pg.draw.circle(surface, (0, 25, 0), pos, 5)

        self.in_terminal.draw(surface)
        self.out_terminal.draw(surface)
