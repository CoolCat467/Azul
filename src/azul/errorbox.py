"""errorbox, by Pete Shinners.

Tries multiple python GUI libraries to display an error box
on the screen. No matter what is successful, the error will also
be sent to stdout. For GUI platforms, this can be a lot nicer than
opening a shell with errors dumped into it.

Call the "errorbox()" function with an error title and error message.

The different GUI libraries will leave the program in undefined
states, so this function will never return, but instead end the
program when the error has been dismissed. That makes this only
useful for errors, and not general message boxes.

Feel free to perhaps add some GUI HANDLERS, as well as enhance
any that are here. They have all been tested on their appropriate
platforms.

There is even a decent pygame handler, if all else fails.

Use it to report things like missing modules (image, numeric, etc?).
Perhaps pygame raised an exception while initializing. This little
messagebox can sure be a lot nicer than a stack trace. ;]
"""

from __future__ import annotations

__title__ = "errorbox"


def errorbox(title: str, message: str) -> None:
    """Attempt to error with a gui."""
    __stdout(title, message)
    for handler in HANDLERS:
        try:
            handler(title, message)
            break
        except (ImportError, NameError):
            pass
    raise SystemExit


def __pyqt4(title: str, message: str) -> None:
    """Error with PyQt4."""
    from PyQt4 import QtGui

    QtGui.QApplication(["Error"])
    QtGui.QMessageBox.critical(None, title, message)


def __wxpython(title: str, message: str) -> None:
    """Error with wxPython."""
    from wxPython.wx import wxApp, wxICON_EXCLAMATION, wxMessageDialog, wxOK

    class LameApp(wxApp):  # type: ignore[misc]
        __slots__ = ()

        def OnInit(self) -> int:  # noqa: N802
            return 1

    LameApp()
    dlg = wxMessageDialog(None, message, title, wxOK | wxICON_EXCLAMATION)
    dlg.ShowModal()
    dlg.Destroy()


def __tkinter(title: str, message: str) -> None:
    """Error with tkinter."""
    import tkinter as tk
    from tkinter import messagebox

    tk.Tk().wm_withdraw()
    # types: attr-defined error: Module has no attribute "messagebox"
    messagebox.showerror(title, message)


def __pygame(title: str, message: str) -> None:
    """Error with pygame."""
    try:
        import pygame
        import pygame.font

        pygame.quit()  # clean out anything running
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((460, 140))
        pygame.display.set_caption(title)
        font = pygame.font.Font(None, 18)
        foreg = 0, 0, 0
        backg = 200, 200, 200
        liteg = 255, 255, 255
        ok = font.render("Ok", True, foreg)
        screen.fill(backg)
        okbox = ok.get_rect().inflate(20, 10)
        okbox.centerx = screen.get_rect().centerx
        okbox.bottom = screen.get_rect().bottom - 10
        screen.fill(liteg, okbox)
        screen.blit(ok, okbox.inflate(-20, -10))
        pos = [20, 20]
        for text in message.split("\n"):
            msg = font.render(text, True, foreg)
            screen.blit(msg, pos)
            pos[1] += font.get_height()

        pygame.display.flip()
        while True:
            e = pygame.event.wait()
            if (
                e.type == pygame.QUIT
                or e.type == pygame.MOUSEBUTTONDOWN
                or (
                    pygame.KEYDOWN
                    and hasattr(e, "key")
                    and e.key
                    in (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN)
                )
            ):
                break
        pygame.quit()
    except pygame.error as exc:
        raise ImportError from exc


def __stdout(title: str, message: str) -> None:
    """Error with stdout."""
    text = "ERROR: " + title + "\n" + message
    print(text)


HANDLERS = __pyqt4, __tkinter, __wxpython, __pygame, __stdout


# test the error box
if __name__ == "__main__":
    errorbox(
        "Testing",
        "This is only a test.\nHad this been "
        + "a real emergency, you would be very afraid.",
    )
