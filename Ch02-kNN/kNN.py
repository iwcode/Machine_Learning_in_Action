#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @FileName: kNN.py
# @Software: PyCharm
# @Author: iwcode
# @Time: 7/21/2020 12:54 AM

import numpy as np
import operator
from os import listdir


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # np.tile(a,(2))函数的作用就是将a沿着X轴扩大两倍，[2,3,4]变成[2,3,4,2,3,4]
    # np.tile(a,(2,3))第一个参数为Y轴扩大倍数,第二个为X轴扩大倍数
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    # 简单的理解就是axis=1表示行向量相加，axis=0表示列向量相加，
    # axis=2是对三维数组进行操作，没有参数就是所有向量相加
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()  # 排序后，原来的索引数组
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(),  # python3中iteritems()已经废除了
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file_to_matrix(file_name):
    fr = open(file_name)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    fr.close()
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    min_values = data_set.min(0)  # 返回每一列的最小值，min(1)则返回每一行最小值
    max_values = data_set.max(0)
    ranges = max_values - min_values
    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_values, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_values


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file_to_matrix('datingTestSet.txt')
    norm_mat, ranges, min_values = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file_to_matrix('datingTestSet.txt')
    norm_mat, ranges, min_values = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_values) / ranges, norm_mat, dating_labels, 3)
    print("You will probably like this person:", result_list[classifier_result - 1])


def img_to_vector(file_name):
    return_vect = np.zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    fr.close()
    return return_vect


def hand_writing_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img_to_vector('trainingDigits/%s' % file_name_str)
    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img_to_vector('testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 50)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total error rate is: %f" % (error_count / float(m_test)))

    