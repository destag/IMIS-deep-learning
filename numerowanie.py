from os import listdir, rename
directory_path = input('Pelna sciezka katalogu: ')
starting_number = int(input('Numer poczatkowy: '))
n = starting_number

for filename in listdir(directory_path):
    if not filename.startswith('IMG'):
        rename(directory_path + '\\' + filename, directory_path + '\\IMG' + str(n) + '.jpg')
        n += 1

print('Zmieniono', n - starting_number, 'plikow.')
