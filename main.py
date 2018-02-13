
import vfdt.dataset as ds


def main():
    file_path = './data/randomrbf.csv'
    dataset = ds.DatasetCSV(['Numerical']*10, 2, file_path)
    counter = 0
    for instance, label in dataset.get_generator():
        print('{}/{}'.format(instance, label))
        counter += 1
        if counter > 2:
            break

if __name__ == '__main__':
    main()
