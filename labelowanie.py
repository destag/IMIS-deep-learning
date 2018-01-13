import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd

directory_path = 'D:/samochody/'
coords = [0, 0, 0, 0]
width, height = 800, 600
starting_number = 1
max_number = 566
make_new_csv = False if starting_number > 1 else True
gen = (i for i in range(starting_number, max_number + 1))

def callback_left(event):
    coords[0] = event.x
    coords[1] = event.y
    canvas.coords(rect, *coords)
    canvas.coords(point, coords[0], coords[1], coords[0], coords[1])

def callback_right(event):
    coords[2] = event.x
    coords[3] = event.y
    canvas.coords(rect, *coords)

def callback_button():
    global img, make_new_csv
    license_plate = entry.get().upper()
    entry.delete(0, 'end')
    df = pd.DataFrame(columns=('license_plate', 'x', 'y', 'w', 'h'),
                      data={'license_plate': [license_plate],
                            'x': coords[0],
                            'y': coords[1],
                            'w': coords[2] - coords[0],
                            'h': coords[3] - coords[1]})
    if make_new_csv:
        df.to_csv(directory_path + 'IMG0.csv', encoding='utf-8', index=False)
        make_new_csv = False
    else:
        df.to_csv(directory_path + 'IMG0.csv', encoding='utf-8', index=False, mode='a', header=False)

    try:
        n = str(next(gen))
    except StopIteration:
        root.destroy()
        print('...')
        return

    root.title('IMG' + n)
    img = Image.open(directory_path + 'IMG' + n + '.jpg').convert('L').resize((width, height), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    canvas.itemconfigure(image_on_canvas, image=img)

root = tk.Tk()
n =  str(next(gen))
root.title('IMG' + n)
img = Image.open(directory_path + 'IMG' + n + '.jpg').convert('L').resize((width, height), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)

main_panel = tk.PanedWindow()
main_panel.pack()

canvas = tk.Canvas(main_panel)
canvas.pack()
canvas.bind('<B1-Motion>', callback_left)
canvas.bind('<B3-Motion>', callback_right)
main_panel.add(canvas, width=width, height=height)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=img)
rect = canvas.create_rectangle(0, 0, 0, 0, outline='yellow')
point = canvas.create_oval(0, 0, 0, 0, width=3, outline='blue')

right_panel = tk.PanedWindow(orient=tk.VERTICAL)
right_panel.add(tk.Label(right_panel, text='numer rejestracyjny'))
entry = tk.Entry(right_panel)
right_panel.add(entry)

button = tk.Button(right_panel, text='nastepne', command=callback_button)
right_panel.add(button)

right_panel.add(tk.Label(right_panel, text='v0.2'))
main_panel.add(right_panel)

root.mainloop()
