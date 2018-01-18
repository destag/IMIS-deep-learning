import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd

directory_path = 'D:/samochody/'
coords = [0, 0, 0, 0, 0, 0, 0, 0]
width, height = 800, 600
starting_number = 1
max_number = 566
make_new_csv = False if starting_number > 1 else True
gen = (i for i in range(starting_number, max_number + 1))
calls = 0
points = [0, 0, 0, 0]
n = starting_number

def callback_left(event):
    global calls, img, make_new_csv, n
    coords[2 * calls] = event.x
    coords[2 * calls + 1] = event.y
    # canvas.coords(rect, *coords)
    canvas.coords(points[calls],
                  coords[2 * calls],
                  coords[2 * calls + 1],
                  coords[2 * calls],
                  coords[2 * calls + 1])
    calls += 1
    if calls > 3:
        calls = 0
        for point in points:
            canvas.coords(point, 0, 0, 0, 0)

        df = pd.DataFrame(columns=('x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'),
                          data={'x1': [coords[0]],
                                'y1': [coords[1]],
                                'x2': [coords[2]],
                                'y2': [coords[3]],
                                'x3': [coords[4]],
                                'y3': [coords[5]],
                                'x4': [coords[6]],
                                'y4': [coords[7]]}, index=[n])
        if make_new_csv:
            df.to_csv(directory_path + 'IMG04p.csv', encoding='utf-8', index_label='idx')
            make_new_csv = False
        else:
            df.to_csv(directory_path + 'IMG04p.csv', encoding='utf-8', mode='a', header=False)

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
canvas.bind('<Button-1>', callback_left)
main_panel.add(canvas, width=width, height=height)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=img)
points[0] = canvas.create_oval(0, 0, 0, 0, width=3, outline='blue')
points[1] = canvas.create_oval(0, 0, 0, 0, width=3, outline='blue')
points[2] = canvas.create_oval(0, 0, 0, 0, width=3, outline='blue')
points[3] = canvas.create_oval(0, 0, 0, 0, width=3, outline='blue')

right_panel = tk.PanedWindow(orient=tk.VERTICAL)

right_panel.add(tk.Label(right_panel, text='v0.2'))
main_panel.add(right_panel)

root.mainloop()
