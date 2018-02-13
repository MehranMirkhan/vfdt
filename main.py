
# import vfdt.dataset as ds


def f1():
    a = [[(2, 4), (1, 6)], [(5, 3), (3, 2)], [(6, 1), (4, 3)]]
    b = [list(zip(*q)) for q in a]
    print(b)
    c = [(sum(q[0]), sum(q[1])) for q in b]
    print(c)


def main():
    # file_path = './data/randomrbf.csv'
    # # dataset = ds.DatasetCSV(['Numerical']*10, 2, file_path)
    # dataset = ds.DatasetCSVChunky(['Numerical']*10, 2,
    #                               file_path, chunksize=10)
    # counter = 0
    # for instance, label in dataset.get_generator():
    #     continue
    f1()

if __name__ == '__main__':
    main()
