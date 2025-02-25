import win32gui

def list_windows():
    """
    Enumerate all visible windows and print their titles.
    """
    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if window_text:
                print(f"HWND: {hwnd}, Title: '{window_text}'")

    win32gui.EnumWindows(callback, None)

if __name__ == "__main__":
    list_windows()