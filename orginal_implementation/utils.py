# utils.py

import os
import csv


def readcsvfile(filename):
    with open(filename, 'r') as f:
        content = []
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            content.append(row)
    f.close()
    return content


def readtextfile(filename):
    with open(filename) as f:
        content = f.readlines()
    f.close()
    return content


def writetextfile(data, filename):
    with open(filename, 'w') as f:
        f.writelines(data)
    f.close()


def delete_file(filename):
    if os.path.isfile(filename) is True:
        os.remove(filename)


def eformat(f, prec, exp_digits):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def saveargs(args):
    path = args.logs
    if os.path.isdir(path) is False:
        os.makedirs(path)
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(arg + ' ' + str(getattr(args, arg)) + '\n')


def file_exists(filename):
    return os.path.isfile(filename)
