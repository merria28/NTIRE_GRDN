import os
import cv2
import csv


def load_BGR(filepath):
    img_BGR = cv2.imread(filepath, cv2.IMREAD_COLOR)
    return img_BGR


def load_grayscale(filepath):
    img_grayscale = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img_grayscale


def make_dirs(path):
    """
    경로(폴더) 가 있음을 확인하고 없으면 새로 생성한다.
    :param path: 확인할 경로
    :return: path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class LogCSV(object):
    def __init__(self, log_dir):
        """
        :param log_dir: log(csv 파일) 가 저장될 dir
        """
        self.log_dir = log_dir
        f = open(self.log_dir, 'a')
        f.close()

    def make_head(self, header):
        with open(self.log_dir, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(header)

    def __call__(self, log):
        """
        :param log: header 의 각 항목에 해당하는 값들의 list
        """
        with open(self.log_dir, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(log)