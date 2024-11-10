#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Button

"Button"

# Programmed by CoolCat467

__title__ = 'Button'
__author__ = 'CoolCat467'
__version__ = '0.0.0'

from typing import Any, Callable

from pygame.locals import SRCALPHA# pylint: disable=no-name-in-module
from pygame.surface import Surface# pylint: disable=no-name-in-module
from pygame import draw, font, cursors, mouse

from component import Component, Event
from vector import Vector

class BoxOutlineComponent(Component):
    "Rounded rectangle box with outline component"
    __slots__ = ('outline', 'color', 'border_radius', 'border_width')
    def __init__(self):
        super().__init__('box_outline')

        self.outline = (0, 0, 0)
        self.color = (0xff, 0xff, 0xff)
        self.border_radius = 15
        self.border_width = 3

    def draw_outline(self, width: int=None, height: int=None) -> None:
        "Clear image"
        sprite = self.get_component('sprite')
        if width is None:
            width = sprite.rect.width
        if height is None:
            height = sprite.rect.height
        sprite.image = Surface((width, height), flags=SRCALPHA)
        blit_rect = (0, 0) + sprite.image_dims

        draw.rect(sprite.image, self.color, blit_rect,
                  border_radius=self.border_radius)
        if self.color != self.outline and self.border_width:
            draw.rect(sprite.image, self.outline, blit_rect,
                      width=self.border_width,
                      border_radius=self.border_radius)
        sprite.dirty = 1

class LabelComponent(Component):
    "Label - Outlined box of text"
    __slots__ = ('__text', 'text_color', 'font')
    def __init__(self):
        super().__init__('label')

        self.__text = ''
        self.text_color = (0, 0, 0)
        self.font = font.Font(font_path, text_size)

    # pylint: disable=unused-private-member
    def __set_text(self, value: str) -> None:
        "Set text to value"
        self.__text = value
        self.reset_image()

    def __get_text(self) -> str:
        "Return text"
        return self.__text

    text = property(__get_text, __set_text, doc='Text')

    def reset_image(self, width=None, height=None) -> tuple:
        "Update image. Return rect of where text is and movement"
        box_outline = self.get_component('box_outline')

        surf = self.font.render(self.text, True, self.text_color)
        width, height = surf.get_size()
        height_add = self.border_width
        width_add = height_add + self.border_radius / 2 + 3

        box_outline.draw_outline( width+(2*width_add), height+(2*height_add) )

        self.image.blit(surf, (width_add, height_add))
        return surf.get_rect(), (width_add, height_add)

# pylint: disable=too-many-instance-attributes
class TextBox(Component):
    "Editable text box"
    __slots__ = ('initial_text', 'focused', 'orig_cursor', '__flash',
                 'flash_time_left', 'submit_response')
    flash_time = 0.5
    def __init__(self, font_path: str, text_size: int, initial_text, *groups,
                 submit_response: Callable=None,
                 text_color: tuple=(0, 0, 0),
                 outline: tuple=(0, 0, 0),
                 button_color: tuple=(0xff, 0xff, 0xff),
                 border_radius: int=5, border_width: int=3):
        self.__flash = True
        self.initial_text = '<'+initial_text+'>'
        super().__init__(font_path, text_size, self.initial_text, *groups,
                         text_color=text_color, outline=outline,
                         button_color=button_color,
                         border_radius=border_radius,
                         border_width=border_width)

        self.focused = False
        self.orig_cursor = mouse.get_cursor()
        self.flash_time_left = self.flash_time
        self.submit_response = submit_response

    # pylint: disable=unused-private-member
    def __get_flash(self) -> bool:
        return self.__flash
    def __set_flash(self, value: bool) -> None:
        "Set flash, but also reset image if needed."
        if value != self.__flash:
            self.__flash = value
            self.reset_image()

    flash = property(__get_flash, __set_flash, 'Bool of should show text input line')

    def update(self, time_passed: float) -> None:
        "Update flashing cursor"
        if self.focused:
            if self.flash_time_left > 0:
                self.flash_time_left -= time_passed
            else:
                self.flash = not self.flash
                self.flash_time_left = self.flash_time

    def reset_image(self, width=None, height=None) -> None:
        "Update image"
        rect, pos_mod = super().reset_image(width, height)
        if self.flash:
            return
        height = rect.height
        width = 2
        pos = tuple(Vector.from_iter(rect.topright)+pos_mod) + (width, height)
        draw.rect(self.image, self.text_color, pos)

    async def change_focus(self, new_focus: bool) -> None:
        "Change focus state"
        if new_focus == self.focused:
            return
        self.focused = new_focus
        if self.focused:
            cursor = cursors.compile(cursors.textmarker_strings)
            mouse.set_cursor((8, 16), (0, 0), *cursor)
            self.flash_time_left = self.flash_time
        else:
##            mouse.set_cursor(*cursors.arrow)
            mouse.set_cursor(self.orig_cursor)
            self.flash = True

            text = self.text if self.text != self.initial_text else ''
            if self.submit_response is None:
                return
            if iscoroutine(self.submit_response):
                await self.submit_response(text)
                return
            self.submit_response(text)

    async def on_click(self, from_top: int) -> str:
        "On button pressed handler."
        if from_top > 0:
            return
        await self.change_focus(True)

    async def on_event(self, event) -> None:
        "Handle events"
        if not mouse.get_focused():
            return await self.change_focus(False)
        if event.type == 'MouseButtonDown':
            if event['button'] == 1:
                return await self.change_focus(False)
            if event['button'] == 3 and self.rect.collidepoint(event['pos']):
                self.text = self.initial_text
                return await self.change_focus(False)
        if self.focused:
            if self.text == self.initial_text:
                self.text = ''#' '*len(self.initial_text)
            if event.type == 'TextInput':
                self.text += event['text']
            if event.type == 'KeyDown':
                if event['unicode'] == '\x08':#delete
                    self.text = self.text[:-1]

##                try:
##                    if event['unicode'] == '\x03':#copy
##                        scrap.put(SCRAP_TEXT, self.text.encode('utf-8'))
##                    elif event['unicode'] == '\x16':#paste
##                        if scrap.contains(SCRAP_TEXT):
##                            selt.text = scrap.get(SCRAP_TEXT).decode('utf-8')
##                except Exception:
##                    pass

                if event['unicode'] == '\r':
                    return await self.change_focus(False)
##            if event.type == 'MouseMotion':
##                self.change_focus(self.rect.collidepoint(event['pos']))
        elif self.text == '':
            self.text = self.initial_text



if __name__ == '__main__':
    print(f'{__title__}\nProgrammed by {__author__}.')
