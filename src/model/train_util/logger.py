import csv
import os


class CSVLogger():
    def __init__(self, save_csv_name, fieldnames, overwrite=True):
        self.save_csv_name = save_csv_name
        self.overwrite = overwrite
        self.fieldnames = fieldnames
        self.writeheader(fieldnames)

    def writeheader(self, fieldnames):
        row_list = self.read_file(self.overwrite)
        if len(row_list) == 0:
            row_list.append(fieldnames)
            self.open_file()
            self.writer.writerows(row_list)
            self.close_file()
        else:
            pass

    def writerow(self, row):
        row_list = self.read_file()
        row_list.append(row)
        self.open_file()
        self.writer.writerows(row_list)
        self.close_file()

    def read_file(self, overwrite=False):
        self.check_file(overwrite)
        row_list = []
        if os.path.exists(self.save_csv_name):
            self.csvfile = open(self.save_csv_name, 'r', newline='')
            reader = csv.reader(self.csvfile)
            for row in reader:
                row_list.append(row)
            self.close_file()
        return row_list

    def open_file(self):
        self.csvfile = open(self.save_csv_name, 'w', newline='')
        self.writer = csv.writer(self.csvfile)

    def close_file(self):
        self.csvfile.close()

    def check_file(self, overwrite=False):
        if overwrite:
            print(f"{self.save_csv_name} check exist...")
            if os.path.exists(self.save_csv_name):
                os.remove(self.save_csv_name)
                print(f"{self.save_csv_name} exist.")
                print(f"{self.save_csv_name} has been deleted.")
            else:
                print(f"{self.save_csv_name} does not exist.")
