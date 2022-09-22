import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = args.root
    csv_list = os.listdir(root)
    print(csv_list)
    for csv in csv_list:
        csv_path = os.path.join(root, csv)
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        with open(csv_path, 'w') as f:
            f.write('loss,train_acc,time,test_acc\n')
            for line in lines[1:]:
                line = line.replace('\n', '')
                line = line.replace(' ', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = line.rstrip(',')
                line = line.split(',')
                line = list(filter(lambda x: x != 'time', line))
                line = ','.join(line) + '\n'
                f.write(line)