
def f1():
    import pandas as pd
    import vfdt.dataset as ds
    # from sklearn.model_selection import train_test_split
    file_path = './data/randomrbf.csv'
    df = pd.read_csv(file_path, nrows=4, skiprows=4)
    print(df)
    for instance, label in ds.data_frame_iterator(df):
        print('instance: {} - label: {}'.format(instance, label))


def f2():
    a = [[(2, 4), (1, 6)], [(5, 3), (3, 2)], [(6, 1), (4, 3)]]
    b = [list(zip(*q)) for q in a]
    print(b)
    c = [(sum(q[0]), sum(q[1])) for q in b]
    print(c)


def main():
    f1()

if __name__ == '__main__':
    main()
