from rembg import remove

inputim='./input/1.png'
outputim='./output/1.png'
def t1():
    with open(inputim, 'rb') as f:
        with open(outputim, 'wb') as ff:
            input=f.read()
            output=remove(input, only_mask=True)
            ff.write(output)


if __name__ == '__main__':
    t1()