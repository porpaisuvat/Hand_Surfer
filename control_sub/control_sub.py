from pynput.keyboard import Key, Controller

keyboard = Controller()

# Press and release space
keyboard.press(Key.left )
keyboard.release(Key.left )
keyboard.press(Key.space)
keyboard.release(Key.space)


