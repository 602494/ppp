from loader import load, binary_encode, col_merge
from ppp import pcal


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "<path>"
    data, columns = load(path)
    print(data[0])
    data, idx, tags = binary_encode(data)
    print(data[0])
    ppp = pcal(data, columns, idx)
    print(ppp)
    data = col_merge(data, idx, tags)
    print(data[0])
