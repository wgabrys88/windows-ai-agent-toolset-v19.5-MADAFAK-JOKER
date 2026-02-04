from __future__ import annotations

"""FRANZ: Stateless action-biased Windows 11 desktop agent.

Architecture:
  - Model receives ONLY the current screenshot per step (no chat history, no hidden state).
  - The cyan FRANZ MEMORY window is the sole persistence mechanism (visible in screenshots).
  - One tool call per step enforced via tool_choice=required.
  - PAUSE/RESUME gates the loop; memory text is editable while paused.
  - OBSERVATION/EXECUTION mode toggle with dynamic sampling parameters.
  - Win32 APIs via ctypes; RichEdit for HUD; SendInput for actions.
  
Image Processing:
  - Lanczos3 downsampling with unsharp mask for text clarity at low resolutions.
  - Custom PNG encoder (no external dependencies).
  
Prompting & Sampling:
  - Mode-aware dynamic sampling: OBSERVATION (temp=1.8, exploratory) vs EXECUTION (temp=0.8, deterministic).
  - Action-first tool ordering: observe() is LAST, framed as "LAST RESORT".
  - Negative pressure prompting: using observe() in EXECUTION mode = task failure.
  - Empty initial memory: forces model to bootstrap from visual inspection.
"""

import argparse
import base64
import ctypes
import ctypes.wintypes as w
import json
import math
import struct
import threading
import time
import urllib.request
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ----------------------------- Configuration -----------------------------

API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen3-vl-2b-instruct"

SCREENSHOT_QUALITY = 2
SCREEN_W, SCREEN_H = {1: (1536, 864), 2: (1024, 576), 3: (512, 288)}[SCREENSHOT_QUALITY]

DUMP_FOLDER = Path("dump")
HUD_SIZE = 1
HUD_FONT_HEIGHT = 20

MODE_OBSERVATION = 0
MODE_EXECUTION = 1


def get_sampling_config(mode: int) -> dict[str, Any]:
    """Return sampling parameters optimized for current mode."""
    if mode == MODE_OBSERVATION:
        return {
            "temperature": 1.8,
            "top_p": 0.90,
            "top_k": 40,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repeat_penalty": 1.10,
            "max_tokens": 600,
            "min_completion_tokens": 120,
        }
    return {
        "temperature": 0.8,
        "top_p": 0.75,
        "top_k": 20,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_penalty": 1.20,
        "max_tokens": 600,
        "min_completion_tokens": 120,
    }


SYSTEM_PROMPT = """You control Windows. One screenshot. One action.

Your memory: the cyan FRANZ MEMORY window. You write reports there and read them from screenshots.

Button color = mode:
- RED (OBSERVATION): scan screen for written tasks. Write descriptive reports. Use observe() ONLY if no task is visible.
- BLUE (EXECUTION): execute task NOW. Write action plans with exact coordinates. MUST use click/drag/type. Using observe() means task failure.

Coordinates: 0,0 = top-left, 1000,1000 = bottom-right.

Report structure (120-200 words):
- Recently: prior action or observation.
- Now: button color, visible task or target UI element.
- Soon: if RED, describe task. If BLUE, state exact action with coordinates.

EXECUTION mode reports must be decisive action plans, not descriptions.

One tool. Act decisively."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Primary action. Click coordinates to interact with UI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate (0-1000)"},
                    "y": {"type": "number", "description": "Y coordinate (0-1000)"},
                    "report": {
                        "type": "string",
                        "description": "Status report (120-200 words, Recently/Now/Soon).",
                        "minLength": 120,
                    },
                },
                "required": ["x", "y", "report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag",
            "description": "Drag from start to end. Use for drawing, moving, selecting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "number", "description": "Start X (0-1000)"},
                    "y1": {"type": "number", "description": "Start Y (0-1000)"},
                    "x2": {"type": "number", "description": "End X (0-1000)"},
                    "y2": {"type": "number", "description": "End Y (0-1000)"},
                    "report": {
                        "type": "string",
                        "description": "What you are dragging and why (120-200 words).",
                        "minLength": 120,
                    },
                },
                "required": ["x1", "y1", "x2", "y2", "report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Type text into focused field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                    "report": {
                        "type": "string",
                        "description": "What field you are typing into and why (120-200 words).",
                        "minLength": 120,
                    },
                },
                "required": ["text", "report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double-click to open or activate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate (0-1000)"},
                    "y": {"type": "number", "description": "Y coordinate (0-1000)"},
                    "report": {"type": "string", "minLength": 120},
                },
                "required": ["x", "y", "report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Right-click for context menu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate (0-1000)"},
                    "y": {"type": "number", "description": "Y coordinate (0-1000)"},
                    "report": {"type": "string", "minLength": 120},
                },
                "required": ["x", "y", "report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll to reveal content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dy": {"type": "number", "description": "Scroll amount (positive=up, negative=down)"},
                    "report": {"type": "string", "minLength": 120},
                },
                "required": ["dy", "report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "observe",
            "description": "LAST RESORT. Use ONLY if: RED button AND no task visible on screen. If BLUE button, DO NOT use this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report": {
                        "type": "string",
                        "description": "Explain why no action is possible (120-200 words, Recently/Now/Soon).",
                        "minLength": 120,
                    }
                },
                "required": ["report"],
            },
        },
    },
]

INITIAL_STORY = ""

# ----------------------------- Win32 setup -----------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
comctl32 = ctypes.WinDLL("comctl32", use_last_error=True)

try:
    ctypes.WinDLL("Shcore").SetProcessDpiAwareness(2)
except Exception:
    pass

try:
    kernel32.LoadLibraryW("Msftedit.dll")
except Exception:
    pass

# Constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
WHEEL_DELTA = 120

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800

KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002

WS_OVERLAPPED = 0x00000000
WS_CAPTION = 0x00C00000
WS_SYSMENU = 0x00080000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_VISIBLE = 0x10000000
WS_VSCROLL = 0x00200000
WS_CHILD = 0x40000000
WS_BORDER = 0x00800000

ES_MULTILINE = 0x0004
ES_AUTOVSCROLL = 0x0040
ES_READONLY = 0x0800

WS_EX_TOPMOST = 0x00000008
WS_EX_LAYERED = 0x00080000

BS_PUSHBUTTON = 0x00000000
SS_CENTER = 0x00000001
SS_NOTIFY = 0x00000100

TBS_HORZ = 0x0000
TBS_AUTOTICKS = 0x0001

WM_SETFONT = 0x0030
WM_SIZE = 0x0005
WM_DESTROY = 0x0002
WM_CTLCOLORSTATIC = 0x0138
WM_CLOSE = 0x0010
WM_COMMAND = 0x0111
WM_HSCROLL = 0x0114

EM_SETBKGNDCOLOR = 0x0443
EM_SETREADONLY = 0x00CF

SW_SHOWNOACTIVATE = 4
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

HWND_TOPMOST = -1
SRCCOPY = 0x00CC0020
LWA_ALPHA = 0x00000002

TBM_GETPOS = 0x0400
TBM_SETPOS = 0x0405
TBM_SETRANGE = 0x0406

CS_HREDRAW = 0x0002
CS_VREDRAW = 0x0001
IDC_ARROW = 32512
COLOR_WINDOW = 5

ICC_BAR_CLASSES = 0x00000004

STN_CLICKED = 0
STN_DBLCLK = 1

HICON = w.HANDLE
HCURSOR = w.HANDLE
HBRUSH = w.HANDLE


def MAKEINTRESOURCEW(i: int) -> w.LPCWSTR:
    return ctypes.cast(ctypes.c_void_p(i & 0xFFFF), w.LPCWSTR)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", w.LONG),
        ("dy", w.LONG),
        ("mouseData", w.DWORD),
        ("dwFlags", w.DWORD),
        ("time", w.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", w.WORD),
        ("wScan", w.WORD),
        ("dwFlags", w.DWORD),
        ("time", w.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", w.DWORD), ("wParamL", w.WORD), ("wParamH", w.WORD)]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("union", _INPUTunion)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", w.DWORD),
        ("biWidth", w.LONG),
        ("biHeight", w.LONG),
        ("biPlanes", w.WORD),
        ("biBitCount", w.WORD),
        ("biCompression", w.DWORD),
        ("biSizeImage", w.DWORD),
        ("biXPelsPerMeter", w.LONG),
        ("biYPelsPerMeter", w.LONG),
        ("biClrUsed", w.DWORD),
        ("biClrImportant", w.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", w.DWORD * 3)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", w.HWND),
        ("message", ctypes.c_uint),
        ("wParam", w.WPARAM),
        ("lParam", w.LPARAM),
        ("time", w.DWORD),
        ("pt", w.POINT),
    ]


WNDPROC = ctypes.WINFUNCTYPE(w.LPARAM, w.HWND, ctypes.c_uint, w.WPARAM, w.LPARAM)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", w.HINSTANCE),
        ("hIcon", HICON),
        ("hCursor", HCURSOR),
        ("hbrBackground", HBRUSH),
        ("lpszMenuName", w.LPCWSTR),
        ("lpszClassName", w.LPCWSTR),
        ("hIconSm", HICON),
    ]


class INITCOMMONCONTROLSEX(ctypes.Structure):
    _fields_ = [("dwSize", w.DWORD), ("dwICC", w.DWORD)]


# Function signatures
gdi32.CreateSolidBrush.argtypes = [w.COLORREF]
gdi32.CreateSolidBrush.restype = w.HBRUSH
gdi32.DeleteObject.argtypes = [w.HGDIOBJ]
gdi32.DeleteObject.restype = w.BOOL
gdi32.SetTextColor.argtypes = [w.HDC, w.COLORREF]
gdi32.SetTextColor.restype = w.COLORREF
gdi32.SetBkColor.argtypes = [w.HDC, w.COLORREF]
gdi32.SetBkColor.restype = w.COLORREF

user32.CreateWindowExW.argtypes = [
    w.DWORD, w.LPCWSTR, w.LPCWSTR, w.DWORD, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, w.HWND, w.HMENU, w.HINSTANCE, w.LPVOID,
]
user32.CreateWindowExW.restype = w.HWND
user32.ShowWindow.argtypes = [w.HWND, ctypes.c_int]
user32.ShowWindow.restype = w.BOOL
user32.SetWindowPos.argtypes = [w.HWND, w.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.SetWindowPos.restype = w.BOOL
user32.DestroyWindow.argtypes = [w.HWND]
user32.DestroyWindow.restype = w.BOOL
user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = ctypes.c_uint
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.GetDC.argtypes = [w.HWND]
user32.GetDC.restype = w.HDC
user32.ReleaseDC.argtypes = [w.HWND, w.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.SetWindowTextW.argtypes = [w.HWND, w.LPCWSTR]
user32.SetWindowTextW.restype = w.BOOL
user32.SendMessageW.argtypes = [w.HWND, ctypes.c_uint, w.WPARAM, w.LPARAM]
user32.SendMessageW.restype = w.LPARAM
user32.PostMessageW.argtypes = [w.HWND, ctypes.c_uint, w.WPARAM, w.LPARAM]
user32.PostMessageW.restype = w.BOOL
user32.GetMessageW.argtypes = [ctypes.POINTER(MSG), w.HWND, ctypes.c_uint, ctypes.c_uint]
user32.GetMessageW.restype = w.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
user32.TranslateMessage.restype = w.BOOL
user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
user32.DispatchMessageW.restype = w.LPARAM
user32.SetLayeredWindowAttributes.argtypes = [w.HWND, w.COLORREF, ctypes.c_ubyte, w.DWORD]
user32.SetLayeredWindowAttributes.restype = w.BOOL
user32.DefWindowProcW.argtypes = [w.HWND, ctypes.c_uint, w.WPARAM, w.LPARAM]
user32.DefWindowProcW.restype = w.LPARAM
user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype = w.ATOM
user32.LoadCursorW.argtypes = [w.HINSTANCE, w.LPCWSTR]
user32.LoadCursorW.restype = HCURSOR
user32.GetClientRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT)]
user32.GetClientRect.restype = w.BOOL
user32.GetWindowRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT)]
user32.GetWindowRect.restype = w.BOOL
user32.InvalidateRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT), w.BOOL]
user32.InvalidateRect.restype = w.BOOL
user32.MoveWindow.argtypes = [w.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.BOOL]
user32.MoveWindow.restype = w.BOOL

gdi32.CreateCompatibleDC.argtypes = [w.HDC]
gdi32.CreateCompatibleDC.restype = w.HDC
gdi32.CreateDIBSection.argtypes = [w.HDC, ctypes.POINTER(BITMAPINFO), ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), w.HANDLE, w.DWORD]
gdi32.CreateDIBSection.restype = w.HBITMAP
gdi32.SelectObject.argtypes = [w.HDC, w.HGDIOBJ]
gdi32.SelectObject.restype = w.HGDIOBJ
gdi32.BitBlt.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HDC, ctypes.c_int, ctypes.c_int, w.DWORD]
gdi32.BitBlt.restype = w.BOOL
gdi32.DeleteDC.argtypes = [w.HDC]
gdi32.DeleteDC.restype = w.BOOL
gdi32.CreateFontW.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.LPCWSTR,
]
gdi32.CreateFontW.restype = w.HFONT

kernel32.GetModuleHandleW.argtypes = [w.LPCWSTR]
kernel32.GetModuleHandleW.restype = w.HMODULE
comctl32.InitCommonControlsEx.argtypes = [ctypes.POINTER(INITCOMMONCONTROLSEX)]
comctl32.InitCommonControlsEx.restype = w.BOOL

# ----------------------------- Helpers -----------------------------


@dataclass(slots=True)
class Coord:
    sw: int
    sh: int

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(max(0.0, min(1000.0, x)) * self.sw / 1000),
            int(max(0.0, min(1000.0, y)) * self.sh / 1000),
        )

    def to_win32(self, x: int, y: int) -> tuple[int, int]:
        return (
            int(x * 65535 / self.sw) if self.sw > 0 else 0,
            int(y * 65535 / self.sh) if self.sh > 0 else 0,
        )


def send_input(inputs: list[INPUT]) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(0.05)


def make_mouse_input(dx: int, dy: int, flags: int, data: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=data, dwFlags=flags, time=0, dwExtraInfo=None)
    return inp


def mouse_click(x: int, y: int, conv: Coord) -> None:
    ax, ay = conv.to_win32(x, y)
    send_input([
        make_mouse_input(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE),
        make_mouse_input(0, 0, MOUSEEVENTF_LEFTDOWN),
        make_mouse_input(0, 0, MOUSEEVENTF_LEFTUP),
    ])


def mouse_right_click(x: int, y: int, conv: Coord) -> None:
    ax, ay = conv.to_win32(x, y)
    send_input([
        make_mouse_input(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE),
        make_mouse_input(0, 0, MOUSEEVENTF_RIGHTDOWN),
        make_mouse_input(0, 0, MOUSEEVENTF_RIGHTUP),
    ])


def mouse_double_click(x: int, y: int, conv: Coord) -> None:
    ax, ay = conv.to_win32(x, y)
    click_down = make_mouse_input(0, 0, MOUSEEVENTF_LEFTDOWN)
    click_up = make_mouse_input(0, 0, MOUSEEVENTF_LEFTUP)
    send_input([make_mouse_input(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE), click_down, click_up])
    time.sleep(0.05)
    send_input([click_down, click_up])


def mouse_drag(x1: int, y1: int, x2: int, y2: int, conv: Coord) -> None:
    ax1, ay1 = conv.to_win32(x1, y1)
    ax2, ay2 = conv.to_win32(x2, y2)

    send_input([
        make_mouse_input(ax1, ay1, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE),
        make_mouse_input(0, 0, MOUSEEVENTF_LEFTDOWN),
    ])
    time.sleep(0.05)

    steps = 10
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(ax1 + (ax2 - ax1) * t)
        iy = int(ay1 + (ay2 - ay1) * t)
        send_input([make_mouse_input(ix, iy, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)])
        time.sleep(0.01)

    send_input([make_mouse_input(0, 0, MOUSEEVENTF_LEFTUP)])


def type_text(text: str) -> None:
    if not text:
        return

    inputs: list[INPUT] = []
    utf16_bytes = text.encode("utf-16le")

    for i in range(0, len(utf16_bytes), 2):
        code = utf16_bytes[i] | (utf16_bytes[i + 1] << 8)

        inp_down = INPUT()
        inp_down.type = INPUT_KEYBOARD
        inp_down.union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=None)
        inputs.append(inp_down)

        inp_up = INPUT()
        inp_up.type = INPUT_KEYBOARD
        inp_up.union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)
        inputs.append(inp_up)

    if inputs:
        send_input(inputs)


def scroll(dy: float) -> None:
    ticks = max(1, int(abs(dy) / WHEEL_DELTA))
    direction = 1 if dy > 0 else -1
    send_input([make_mouse_input(0, 0, MOUSEEVENTF_WHEEL, WHEEL_DELTA * direction) for _ in range(ticks)])


def capture_screen(sw: int, sh: int) -> bytes:
    sdc = user32.GetDC(0)
    if not sdc:
        raise ctypes.WinError(ctypes.get_last_error())

    mdc = gdi32.CreateCompatibleDC(sdc)
    if not mdc:
        user32.ReleaseDC(0, sdc)
        raise ctypes.WinError(ctypes.get_last_error())

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = sw
    bmi.bmiHeader.biHeight = -sh
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32

    bits = ctypes.c_void_p()
    hbm = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    if not hbm:
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        raise ctypes.WinError(ctypes.get_last_error())

    gdi32.SelectObject(mdc, hbm)

    if not gdi32.BitBlt(mdc, 0, 0, sw, sh, sdc, 0, 0, SRCCOPY):
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        raise ctypes.WinError(ctypes.get_last_error())

    out = ctypes.string_at(bits, sw * sh * 4)

    user32.ReleaseDC(0, sdc)
    gdi32.DeleteDC(mdc)
    gdi32.DeleteObject(hbm)

    return out


def downsample(src: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    """Lanczos3 downsampling with unsharp mask for text clarity."""
    if (sw, sh) == (dw, dh):
        return src

    def lanczos_kernel(x: float, a: int = 3) -> float:
        if x == 0.0:
            return 1.0
        if abs(x) >= a:
            return 0.0
        pi_x = math.pi * x
        return a * math.sin(pi_x) * math.sin(pi_x / a) / (pi_x * pi_x)

    src_mv = memoryview(src)
    temp = bytearray(dw * dh * 4)

    x_scale = sw / dw
    y_scale = sh / dh
    radius = 3

    for dy in range(dh):
        cy = (dy + 0.5) * y_scale - 0.5
        y_start = max(0, int(cy - radius))
        y_end = min(sh, int(cy + radius) + 1)

        for dx in range(dw):
            cx = (dx + 0.5) * x_scale - 0.5
            x_start = max(0, int(cx - radius))
            x_end = min(sw, int(cx + radius) + 1)

            accum = [0.0, 0.0, 0.0, 0.0]
            weight_sum = 0.0

            for sy in range(y_start, y_end):
                wy = lanczos_kernel((sy - cy) / y_scale)
                for sx in range(x_start, x_end):
                    wx = lanczos_kernel((sx - cx) / x_scale)
                    w = wx * wy

                    si = (sy * sw + sx) * 4
                    for c in range(4):
                        accum[c] += src_mv[si + c] * w
                    weight_sum += w

            di = (dy * dw + dx) * 4
            if weight_sum > 0.0:
                for c in range(4):
                    val = int(accum[c] / weight_sum)
                    temp[di + c] = max(0, min(255, val))

    dst = bytearray(dw * dh * 4)
    amount = 0.8
    threshold = 10

    for dy in range(dh):
        for dx in range(dw):
            di = (dy * dw + dx) * 4

            blur = [0.0, 0.0, 0.0]
            count = 0.0

            for oy in range(-1, 2):
                for ox in range(-1, 2):
                    ny = max(0, min(dh - 1, dy + oy))
                    nx = max(0, min(dw - 1, dx + ox))
                    ni = (ny * dw + nx) * 4

                    kernel_weight = 1.0 if (ox == 0 and oy == 0) else 0.5
                    for c in range(3):
                        blur[c] += temp[ni + c] * kernel_weight
                    count += kernel_weight

            for c in range(3):
                blur[c] /= count

            for c in range(3):
                original = temp[di + c]
                detail = original - blur[c]

                if abs(detail) > threshold:
                    sharpened = original + detail * amount
                    dst[di + c] = max(0, min(255, int(sharpened)))
                else:
                    dst[di + c] = temp[di + c]

            dst[di + 3] = temp[di + 3]

    return bytes(dst)


def encode_png(bgra: bytes, width: int, height: int) -> bytes:
    raw = bytearray((width * 3 + 1) * height)

    for y in range(height):
        raw[y * (width * 3 + 1)] = 0
        row_offset = y * width * 4
        row = bgra[row_offset : row_offset + width * 4]
        di = y * (width * 3 + 1) + 1

        for x in range(width):
            raw[di + x * 3] = row[x * 4 + 2]
            raw[di + x * 3 + 1] = row[x * 4 + 1]
            raw[di + x * 3 + 2] = row[x * 4]

    comp = zlib.compress(bytes(raw), 6)
    ihdr = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")


def call_vlm(png: bytes, mode: int) -> tuple[str, dict[str, Any]]:
    """Call VLM with mode-aware sampling parameters."""
    sampling = get_sampling_config(mode)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"},
                    }
                ],
            },
        ],
        "tools": TOOLS,
        "tool_choice": "required",
        **sampling,
    }

    req = urllib.request.Request(
        API_URL,
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data: dict[str, Any] = json.load(resp)

    message = data["choices"][0]["message"]
    tool_calls = message["tool_calls"]
    tc = tool_calls[0]

    name: str = tc["function"]["name"]
    args_raw = tc["function"]["arguments"]
    args: dict[str, Any] = json.loads(args_raw) if isinstance(args_raw, str) else args_raw

    return name, args


# ----------------------------- HUD -----------------------------


@dataclass(slots=True)
class HUD:
    hwnd: w.HWND | None = None
    edit_hwnd: w.HWND | None = None
    button_hwnd: w.HWND | None = None
    slider_hwnd: w.HWND | None = None
    mode_hwnd: w.HWND | None = None

    mode: int = field(default=MODE_OBSERVATION)
    _mode_brush_obs: w.HBRUSH | None = None
    _mode_brush_exec: w.HBRUSH | None = None

    thread: threading.Thread | None = None
    ready_event: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)

    paused: bool = field(default=True)
    pause_event: threading.Event = field(default_factory=threading.Event)

    _wndproc_ref: Any = None

    _BTN_ID = 1001
    _SLIDER_ID = 1002
    _MODE_ID = 1003

    def _mode_text(self) -> str:
        return "EXECUTION" if self.mode == MODE_EXECUTION else "OBSERVATION"

    def set_mode(self, new_mode: int) -> None:
        self.mode = int(new_mode)
        if self.mode_hwnd:
            user32.SetWindowTextW(self.mode_hwnd, self._mode_text())
            user32.InvalidateRect(self.mode_hwnd, None, True)

    def _layout_controls(self) -> None:
        if not self.hwnd or not self.edit_hwnd:
            return

        rc = w.RECT()
        if not user32.GetClientRect(self.hwnd, ctypes.byref(rc)):
            return

        width = rc.right - rc.left
        height = rc.bottom - rc.top

        margin = 10
        btn_h = 35
        slider_h = 40

        bottom_area = margin + btn_h + margin + slider_h + margin
        edit_h = max(50, height - bottom_area)

        user32.MoveWindow(self.edit_hwnd, margin, margin, max(50, width - 2 * margin), edit_h, True)

        row_y = margin + edit_h + margin
        if self.button_hwnd:
            user32.MoveWindow(self.button_hwnd, margin, row_y, 150, btn_h, True)

        mode_w = 260
        if self.mode_hwnd:
            user32.MoveWindow(self.mode_hwnd, max(margin, width - margin - mode_w), row_y, mode_w, btn_h, True)

        if self.slider_hwnd:
            user32.MoveWindow(
                self.slider_hwnd,
                margin,
                row_y + btn_h + margin,
                max(50, width - 2 * margin),
                slider_h,
                True,
            )

    def _set_paused_ui(self, paused: bool) -> None:
        self.paused = paused

        if self.button_hwnd:
            user32.SetWindowTextW(self.button_hwnd, "RESUME" if paused else "PAUSE")

        if self.edit_hwnd:
            user32.SendMessageW(self.edit_hwnd, EM_SETREADONLY, 0 if paused else 1, 0)

        if paused:
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def _wndproc(self, hwnd: w.HWND, msg: int, wparam: w.WPARAM, lparam: w.LPARAM) -> w.LPARAM:
        try:
            if msg == WM_COMMAND:
                cmd_id = int(wparam) & 0xFFFF
                notify = (int(wparam) >> 16) & 0xFFFF
                if cmd_id == self._BTN_ID:
                    self._set_paused_ui(not self.paused)
                    return 0
                if cmd_id == self._MODE_ID and notify in (STN_CLICKED, STN_DBLCLK):
                    self.set_mode(MODE_EXECUTION if self.mode == MODE_OBSERVATION else MODE_OBSERVATION)
                    return 0

            elif msg == WM_CTLCOLORSTATIC:
                if self.mode_hwnd and int(lparam) == int(self.mode_hwnd):
                    hdc = w.HDC(wparam)
                    WHITE = w.COLORREF(0x00FFFFFF)
                    RED = w.COLORREF(0x000000FF)
                    BLUE = w.COLORREF(0x00FF0000)
                    gdi32.SetTextColor(hdc, WHITE)
                    if self.mode == MODE_EXECUTION:
                        gdi32.SetBkColor(hdc, BLUE)
                        return int(self._mode_brush_exec) if self._mode_brush_exec else 0
                    gdi32.SetBkColor(hdc, RED)
                    return int(self._mode_brush_obs) if self._mode_brush_obs else 0

            elif msg == WM_HSCROLL:
                if self.slider_hwnd and int(lparam) == int(self.slider_hwnd):
                    pos = int(user32.SendMessageW(self.slider_hwnd, TBM_GETPOS, 0, 0))
                    pos = max(20, min(255, pos))
                    user32.SetLayeredWindowAttributes(self.hwnd, 0, ctypes.c_ubyte(pos), LWA_ALPHA)
                    return 0

            elif msg == WM_SIZE:
                self._layout_controls()
                return 0

            elif msg == WM_CLOSE:
                self.stop_event.set()
                user32.DestroyWindow(hwnd)
                return 0

            elif msg == WM_DESTROY:
                self.stop_event.set()
                self.pause_event.set()
                try:
                    if self._mode_brush_obs:
                        gdi32.DeleteObject(w.HGDIOBJ(self._mode_brush_obs))
                        self._mode_brush_obs = None
                    if self._mode_brush_exec:
                        gdi32.DeleteObject(w.HGDIOBJ(self._mode_brush_exec))
                        self._mode_brush_exec = None
                except Exception:
                    pass
                return 0

        except Exception as e:
            try:
                print(f"[HUD wndproc error] {e}")
            except Exception:
                pass

        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _window_thread(self) -> None:
        hinst = kernel32.GetModuleHandleW(None)

        icc = INITCOMMONCONTROLSEX(dwSize=ctypes.sizeof(INITCOMMONCONTROLSEX), dwICC=ICC_BAR_CLASSES)
        comctl32.InitCommonControlsEx(ctypes.byref(icc))

        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)

        win_w, win_h = (sw // 4, sh // 4) if HUD_SIZE == 0 else (480, 650)
        win_x, win_y = (500, 500) if HUD_SIZE == 0 else (1400, 200)

        self._wndproc_ref = WNDPROC(self._wndproc)

        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.style = CS_HREDRAW | CS_VREDRAW
        wc.lpfnWndProc = self._wndproc_ref
        wc.cbClsExtra = 0
        wc.cbWndExtra = 0
        wc.hInstance = hinst
        wc.hIcon = None
        wc.hCursor = user32.LoadCursorW(None, MAKEINTRESOURCEW(IDC_ARROW))
        wc.hbrBackground = w.HBRUSH(COLOR_WINDOW + 1)
        wc.lpszMenuName = None
        wc.lpszClassName = "FRANZWindowClass"
        wc.hIconSm = None

        atom = user32.RegisterClassExW(ctypes.byref(wc))
        if not atom:
            err = ctypes.get_last_error()
            ERROR_CLASS_ALREADY_EXISTS = 1410
            if err != ERROR_CLASS_ALREADY_EXISTS:
                self.ready_event.set()
                return

        self.hwnd = user32.CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_LAYERED,
            "FRANZWindowClass",
            "FRANZ MEMORY",
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_VISIBLE,
            win_x, win_y, win_w, win_h,
            None, None, hinst, None,
        )

        if not self.hwnd:
            self.ready_event.set()
            return

        user32.SetLayeredWindowAttributes(self.hwnd, 0, ctypes.c_ubyte(255), LWA_ALPHA)

        mono_font = gdi32.CreateFontW(-HUD_FONT_HEIGHT, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 0, 0, "Consolas")
        ui_font = gdi32.CreateFontW(-14, 0, 0, 0, 700, 0, 0, 0, 1, 0, 0, 0, 0, "Segoe UI")

        self.edit_hwnd = user32.CreateWindowExW(
            0, "RICHEDIT50W", "",
            WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL,
            10, 10, win_w - 40, win_h - 140,
            self.hwnd, None, hinst, None,
        )

        if self.edit_hwnd and mono_font:
            user32.SendMessageW(self.edit_hwnd, WM_SETFONT, mono_font, 1)

        CYAN = 0x00FFFF00
        user32.SendMessageW(self.edit_hwnd, EM_SETBKGNDCOLOR, 0, CYAN)
        user32.SetWindowTextW(self.edit_hwnd, INITIAL_STORY)

        self.button_hwnd = user32.CreateWindowExW(
            0, "BUTTON", "RESUME",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, win_h - 120, 150, 35,
            self.hwnd, w.HMENU(self._BTN_ID), hinst, None,
        )

        if self.button_hwnd and ui_font:
            user32.SendMessageW(self.button_hwnd, WM_SETFONT, ui_font, 1)

        RED = w.COLORREF(0x000000FF)
        BLUE = w.COLORREF(0x00FF0000)
        self._mode_brush_obs = gdi32.CreateSolidBrush(RED)
        self._mode_brush_exec = gdi32.CreateSolidBrush(BLUE)

        self.mode_hwnd = user32.CreateWindowExW(
            0, "STATIC", "OBSERVATION",
            WS_CHILD | WS_VISIBLE | SS_CENTER | SS_NOTIFY | WS_BORDER,
            0, 0, 0, 0,
            self.hwnd, w.HMENU(self._MODE_ID), hinst, None,
        )
        if self.mode_hwnd and ui_font:
            user32.SendMessageW(self.mode_hwnd, WM_SETFONT, ui_font, 1)

        self.set_mode(self.mode)

        self.slider_hwnd = user32.CreateWindowExW(
            0, "msctls_trackbar32", "",
            WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_AUTOTICKS,
            10, win_h - 70, win_w - 40, 40,
            self.hwnd, w.HMENU(self._SLIDER_ID), hinst, None,
        )

        if self.slider_hwnd:
            user32.SendMessageW(self.slider_hwnd, TBM_SETRANGE, 1, (255 << 16) | 20)
            user32.SendMessageW(self.slider_hwnd, TBM_SETPOS, 1, 255)

        self._set_paused_ui(True)
        self._layout_controls()

        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
        user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)

        self.ready_event.set()

        msg = MSG()
        while not self.stop_event.is_set():
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret == 0 or ret == -1:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        self.stop_event.set()
        self.pause_event.set()

    def __enter__(self) -> "HUD":
        self.ready_event.clear()
        self.stop_event.clear()
        self.pause_event.clear()
        self.paused = True

        self.thread = threading.Thread(target=self._window_thread, daemon=True)
        self.thread.start()
        self.ready_event.wait(timeout=2.0)
        time.sleep(0.2)
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        self.pause_event.set()
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        if self.thread:
            self.thread.join(timeout=1.0)

    def update(self, report: str) -> None:
        if self.edit_hwnd:
            user32.SetWindowTextW(self.edit_hwnd, report)
        if self.hwnd:
            user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)

    def wait_for_resume(self) -> None:
        while self.paused and not self.stop_event.is_set():
            self.pause_event.wait(timeout=0.1)


# ----------------------------- Tool execution / testing -----------------------------


def execute_tool_action(tool: str, args: dict[str, Any], conv: Coord) -> None:
    match tool:
        case "click":
            x, y = conv.to_screen(float(args["x"]), float(args["y"]))
            mouse_click(x, y, conv)
        case "right_click":
            x, y = conv.to_screen(float(args["x"]), float(args["y"]))
            mouse_right_click(x, y, conv)
        case "double_click":
            x, y = conv.to_screen(float(args["x"]), float(args["y"]))
            mouse_double_click(x, y, conv)
        case "drag":
            x1, y1 = conv.to_screen(float(args["x1"]), float(args["y1"]))
            x2, y2 = conv.to_screen(float(args["x2"]), float(args["y2"]))
            mouse_drag(x1, y1, x2, y2, conv)
        case "type":
            type_text(str(args["text"]))
        case "scroll":
            scroll(float(args["dy"]))


def test_tool(tool: str, **kwargs) -> None:
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    conv = Coord(sw=sw, sh=sh)

    print(f"Testing tool: {tool}")
    print(f"Parameters: {kwargs}")

    args = {**kwargs, "report": f"Testing {tool} tool execution with parameters {kwargs}. Verifying SendInput injection and coordinate conversion. Monitoring for successful interaction completion."}

    with HUD() as hud:
        hud.update(args["report"])
        hud._set_paused_ui(False)
        time.sleep(1.0)
        execute_tool_action(tool, args, conv)
        time.sleep(1.0)
        print(f"Tool {tool} executed")


def prompt_with_default(prompt: str, default: Any) -> str:
    response = input(f"{prompt} [{default}]: ").strip()
    return response if response else str(default)


# ----------------------------- Main -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FRANZ - Stateless action-biased Windows desktop agent")
    parser.add_argument("--test", choices=["click", "right_click", "double_click", "drag", "type", "scroll"], help="Test a specific tool")

    args = parser.parse_args()

    if args.test:
        params: dict[str, Any] = {}
        if args.test in ["click", "right_click", "double_click"]:
            params = {
                "x": float(prompt_with_default("X coordinate (0-1000)", 500)),
                "y": float(prompt_with_default("Y coordinate (0-1000)", 500)),
            }
        elif args.test == "drag":
            params = {
                "x1": float(prompt_with_default("Start X (0-1000)", 400)),
                "y1": float(prompt_with_default("Start Y (0-1000)", 400)),
                "x2": float(prompt_with_default("End X (0-1000)", 600)),
                "y2": float(prompt_with_default("End Y (0-1000)", 600)),
            }
        elif args.test == "type":
            params = {"text": prompt_with_default("Text to type", "Hello FRANZ")}
        elif args.test == "scroll":
            params = {"dy": float(prompt_with_default("Scroll amount (positive=up, negative=down)", 240))}

        test_tool(args.test, **params)
        return

    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    conv = Coord(sw=sw, sh=sh)

    dump_dir = DUMP_FOLDER / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    print(f"FRANZ awakens | Physical: {sw}x{sh} | Perception: {SCREEN_W}x{SCREEN_H}")
    print(f"Quality: {SCREENSHOT_QUALITY} | Adaptive sampling: OBS temp=1.8, EXEC temp=0.8")
    print(f"Dump: {dump_dir}")
    print("Starting PAUSED - Click RESUME to begin\n")

    with HUD() as hud:
        step = 0

        hud.wait_for_resume()
        if hud.stop_event.is_set():
            return

        while not hud.stop_event.is_set():
            hud.wait_for_resume()
            if hud.stop_event.is_set():
                break

            time.sleep(0.02)
            step += 1
            ts = datetime.now().strftime("%H:%M:%S")

            bgra = capture_screen(sw, sh)
            down = downsample(bgra, sw, sh, SCREEN_W, SCREEN_H)
            png = encode_png(down, SCREEN_W, SCREEN_H)
            (dump_dir / f"step{step:03d}.png").write_bytes(png)

            try:
                tool, args2 = call_vlm(png, hud.mode)
                report = args2.get("report", "")

                print(f"\n[{ts}] {step:03d} | {tool}")
                print(f"{report}\n")

                hud.update(report)
                time.sleep(0.1)

                if tool != "observe":
                    execute_tool_action(tool, args2, conv)
                    time.sleep(0.5)
                else:
                    time.sleep(0.2)

            except Exception as e:
                print(f"[{ts}] Error: {e}")
                time.sleep(2.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nFRANZ sleeps.")
