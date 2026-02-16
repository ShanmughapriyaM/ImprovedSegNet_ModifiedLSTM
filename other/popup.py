import PySimpleGUI as sg


def popup(full, pre):

    layout = [[sg.Button(f'{" " * 7}Full-Analysis+Plots{" " * 7}'), sg.Button(f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}')]]

    event, values = sg.Window('', layout).read(close=True)
    if event == f'{" " * 7}Full-Analysis+Plots{" " * 7}':
        db = int(input('Enter the database you want execute:1 or 2:'))
        while db > 2:
            print('Error : Incorrect Database Number')
            db = int(input('Enter the database from 1 0rt 2:'))
        full(db)
    elif event == f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}':
        db = int(input('Enter the database you want execute:1 or 2 :'))
        while db > 2:
            print('Error : Incorrect Database Number')
            db = int(input('Enter the database from 1 to 2:'))
        pre(db)
