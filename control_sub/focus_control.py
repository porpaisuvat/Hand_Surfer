import win32gui
import win32con
from pynput.keyboard import Key, Controller
import time

keyboard = Controller()

def bring_bluestacks_to_front():
    hwnd = win32gui.FindWindow(None, "BlueStacks App Player")
    if hwnd:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        
    else:
        print("Could not find the BlueStacks window")

def left():
    bring_bluestacks_to_front()
    keyboard.press(Key.left)
    keyboard.release(Key.left)

def right():
    bring_bluestacks_to_front()
    keyboard.press(Key.right)
    keyboard.release(Key.right)

def jump():
    bring_bluestacks_to_front()
    keyboard.press(Key.up)
    keyboard.release(Key.up)

def roll():
    bring_bluestacks_to_front()
    keyboard.press(Key.down)
    keyboard.release(Key.down)

def space():
    bring_bluestacks_to_front()
    keyboard.press(Key.space)
    keyboard.release(Key.space)
    time.sleep(0.100)
    keyboard.press(Key.space)
    keyboard.release(Key.space)

if __name__ == "__main__":
    jump()