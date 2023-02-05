import numpy as np
import os
import shutil
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import time
import math
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import torch
import argparse
from collections import Counter


def analysis_features():
    m = loadmat('duke' + '_pytorch_target_result.mat')
    print('type(m) = %s' % type(m))
    print(m.keys())
    target_features = m['train_f']
    data_num = len(target_features)

    np.random.seed(100)

    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    # savemat(os.path.join(dst_path, 'target_features.mat'), {'features': target_features})

    target_labels = m['train_label'][0]
    # print('target_labels = %s' % (target_labels))
    # print('unique lable = %s' % np.unique(np.sort(target_labels)))
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))

    target_names = m['train_name']
    print(sorted(Counter(target_labels).values())[:10])
    print(sorted(Counter(target_labels).values())[-10:])

    indices = np.random.permutation(len(target_labels))
    target_labels = target_labels[indices]
    target_features = target_features[indices]

    same_dist = []
    diff_dist = []
    same_max = []
    diff_min = []
    same_avg = []
    diff_avg = []

    for i in np.arange(len(target_features)):
        dist = ((target_features[i] - target_features) * (target_features[i] - target_features)).sum(1)
        same_sub = [dist[j] for j in np.arange(len(target_features)) if target_labels[j] == target_labels[i]]
        diff_sub = [dist[j] for j in np.arange(len(target_features)) if target_labels[j] != target_labels[i]]
        same_dist.append(same_sub)
        diff_dist.append(diff_sub[: len(same_sub)])
        same_max.append(np.max(same_sub))
        diff_min.append(np.min(diff_sub))
        same_avg.append(np.sum(same_sub) / (len(same_sub) - 1 + 1e-6))
        diff_avg.append(np.mean(diff_sub))
        if i % 200 == 0:
            print('i = %3d' % i)
        if i > 1000:
            break

    cnt = np.sum(np.array(same_max) < np.array(diff_min))
    ratio = cnt / len(same_max)
    print(ratio)
    print('avg same_max = %.3f    avg diff_min = %.3f' % (np.mean(same_max), np.mean(diff_min)))
    print('avg same = %.3f    avg diff = %.3f' % (np.mean(same_avg), np.mean(diff_avg)))


##############################################################################
# Compute DBSCAN

def generate_cluster(cluster_result_path, dist=None, eps=0.8, min_samples=10, data_dir=None, flag='did'):
    m = loadmat(str(0) + '_' + flag + '_' + data_dir + '_pytorch_target_result.mat')
    target_features = m['train_f']
    data_num = len(target_features)
    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    target_labels = m['train_label'][0]
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))
    target_names = m['train_name']

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    process_num = m['train_label'][0].shape[0]
    if dist is None:
        X = target_features[:process_num]
    else:
        X = dist[:process_num]
    labels_true = target_labels[:process_num]
    names = target_names[:process_num]
    print('DBSCAN starting ......')
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(sorted(Counter(labels).values())[:10])
    print(sorted(Counter(labels).values())[-10:])

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    n_node = np.sum(labels != -1)
    print('Estimated number of nodes: %d' % n_node)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(n_clusters_):
        files = names[np.where(labels == i)]
        if len(files) > np.max((2, min_samples - 1)):
            dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
            os.mkdir(dir_path)
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
        dir_cnt += 1
        image_cnt += len(files)

    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))

def generate_cluster_kmeans(cluster_result_path, dist=None, eps=0.8, min_samples=10, data_dir=None, flag='did'):
    m = loadmat(str(0) + '_' + flag + '_' + data_dir + '_pytorch_target_result.mat')
    target_features = m['train_f']
    data_num = len(target_features)
    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    target_labels = m['train_label'][0]
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))
    target_names = m['train_name']

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    process_num = m['train_label'][0].shape[0]
    if dist is None:
        X = target_features[:process_num]
    else:
        X = dist[:process_num]
    labels_true = target_labels[:process_num]
    names = target_names[:process_num]
    print('KMeans starting ......')
    db = KMeans(n_clusters=702).fit(X)
    labels = db.labels_
    print(sorted(Counter(labels).values())[:10])
    print(sorted(Counter(labels).values())[-10:])

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    n_node = np.sum(labels != -1)
    print('Estimated number of nodes: %d' % n_node)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(n_clusters_):
        files = names[np.where(labels == i)]
        if len(files) > np.max((2, min_samples - 1)):
            dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
            os.mkdir(dir_path)
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
        dir_cnt += 1
        image_cnt += len(files)

    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))

def generate_cluster_with_semantic_clean(cluster_result_path, dist=None, eps=0.8, min_samples=10, domain_num=5, data_dir=None, flag='all'):
    info_mat = []
    target_features_all = []
    target_labels_all = []
    target_names_all = []
    target_cluster_result_all = []
    target_cluster_num_all = []
    target_nodes_all = []
    if 'did' in flag:
        clean_list = ['did']
    elif 'sid' in flag:
        clean_list = ['sid']
    else:
        clean_list = ['all', 'did', 'sid']
    for cl in clean_list:
        m = loadmat(str(0) + '_' + cl + '_' + data_dir + '_pytorch_target_result.mat')
        target_features = m['train_f']
        target_labels = m['train_label'][0]
        target_names = m['train_name']
        process_num = m['train_label'][0].shape[0]
        if dist is None:
            X = target_features[:process_num]
        else:
            X = dist[cl][:process_num]
        labels_true = target_labels[:process_num]
        names = target_names[:process_num]
        print('Semantic %s  DBSCAN starting ......' % cl)
        db = DBSCAN(eps=eps[cl], min_samples=min_samples).fit(X)
        labels = db.labels_
        print(sorted(Counter(labels).values())[:10])
        print(sorted(Counter(labels).values())[-10:])
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels != -1)
        print('Estimated number of nodes: %d' % n_node)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

        target_features_all.append(target_features)
        target_labels_all.append(labels_true)
        target_names_all.append(target_names)
        target_cluster_result_all.append(labels)
        target_cluster_num_all.append(n_clusters_)
        sub_nodes = []
        for j in np.arange(n_clusters_):
            nodes = names[np.where(labels == j)]
            sub_nodes.append(nodes)
        target_nodes_all.append(sub_nodes)

    print('Semantic Cluster Data Cleaning ......')
    cnt = 0
    for i in np.arange(1, len(clean_list)):
        for j in np.arange(target_cluster_num_all[0]):
            iou_max = 0.0
            index = -1
            for k in np.arange(target_cluster_num_all[i]):
                iou = len(np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][k])) / len(
                    np.union1d(target_nodes_all[0][j], target_nodes_all[i][k]))
                if iou > 1e-6:
                    # print('cnt = %3d  iou = %.3f' % (cnt, iou))
                    cnt += 1
                if iou > iou_max:
                    iou_max = iou
                    index = k
            if index != -1:
                disabled_nodes = np.setdiff1d(target_nodes_all[0][j], target_nodes_all[i][index])
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][index])
            else:
                disabled_nodes = target_nodes_all[0][j]
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = []

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(len(target_nodes_all[0])):
        dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
        if len(target_nodes_all[0][i]) >= 4:
            os.mkdir(dir_path)
            files = target_nodes_all[0][i]
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
            dir_cnt += 1
            image_cnt += len(files)
    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(target_labels_all[0], target_cluster_result_all[0]))
    print("Completeness: %0.3f" % metrics.completeness_score(target_labels_all[0], target_cluster_result_all[0]))

def generate_cluster_with_semantic_kmeans_clean(cluster_result_path, dist=None, eps=0.8, min_samples=10, domain_num=5, data_dir=None, flag='all'):
    info_mat = []
    target_features_all = []
    target_labels_all = []
    target_names_all = []
    target_cluster_result_all = []
    target_cluster_num_all = []
    target_nodes_all = []
    if 'did' in flag:
        clean_list = ['did']
    elif 'sid' in flag:
        clean_list = ['sid']
    else:
        clean_list = ['all', 'did', 'sid']
    cluster_num = 0
    indices = 0
    for cl in clean_list:
        if 'all' in cl:
            m = loadmat(str(0) + '_' + cl + '_' + data_dir + '_pytorch_target_result.mat')
            target_features = m['train_f']
            target_labels = m['train_label'][0]
            target_names = m['train_name']
            process_num = m['train_label'][0].shape[0]
            if dist is None:
                X = target_features[:process_num]
            else:
                X = dist[cl][:process_num]
            labels_true = target_labels[:process_num]
            names = target_names[:process_num]
            print('Semantic  %s  DBSCAN  cluster starting ......' % cl)
            db = DBSCAN(eps=eps[cl], min_samples=min_samples).fit(X)
            labels = db.labels_
            db_labels = labels
            indices = np.where(labels != -1)[0]
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_num = n_clusters_
        else:
            m = loadmat(str(0) + '_' + cl + '_' + data_dir + '_pytorch_target_result.mat')
            target_features = m['train_f']
            target_labels = m['train_label'][0]
            target_names = m['train_name']
            process_num = m['train_label'][0].shape[0]
            if dist is None:
                X = target_features[:process_num]
            else:
                X = dist[cl][:process_num]
            labels_true = target_labels[:process_num]
            names = target_names[:process_num]
            print('Semantic  %s  KMeans  cluster starting ......' % cl)
            cluster_center = np.zeros((cluster_num, X.shape[1]))
            for i in np.arange(cluster_num):
                cluster_center[i] = np.average(X[np.where(db_labels==i)[0]], 0)
            db = KMeans(n_clusters=cluster_num, init=cluster_center).fit(X)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print(sorted(Counter(labels).values())[:10])
        print(sorted(Counter(labels).values())[-10:])
        print('Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels != -1)
        print('Estimated number of nodes: %d' % n_node)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        target_features_all.append(target_features)
        target_labels_all.append(labels_true)
        target_names_all.append(target_names)
        target_cluster_result_all.append(labels)
        target_cluster_num_all.append(n_clusters_)
        sub_nodes = []
        for j in np.arange(n_clusters_):
            nodes = names[np.where(labels == j)]
            sub_nodes.append(nodes)
        target_nodes_all.append(sub_nodes)

    print('Semantic Cluster Data Cleaning ......')
    cnt = 0
    for i in np.arange(1, len(clean_list)):
        for j in np.arange(target_cluster_num_all[0]):
            iou_max = 0.0
            index = -1
            for k in np.arange(target_cluster_num_all[i]):
                iou = len(np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][k])) / len(
                    np.union1d(target_nodes_all[0][j], target_nodes_all[i][k]))
                if iou > 1e-6:
                    # print('cnt = %3d  iou = %.3f' % (cnt, iou))
                    cnt += 1
                if iou > iou_max:
                    iou_max = iou
                    index = k
            if index != -1:
                disabled_nodes = np.setdiff1d(target_nodes_all[0][j], target_nodes_all[i][index])
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][index])
            else:
                disabled_nodes = target_nodes_all[0][j]
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = []

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(len(target_nodes_all[0])):
        dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
        if len(target_nodes_all[0][i]) >= 4:
            os.mkdir(dir_path)
            files = target_nodes_all[0][i]
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
            dir_cnt += 1
            image_cnt += len(files)
    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(target_labels_all[0], target_cluster_result_all[0]))
    print("Completeness: %0.3f" % metrics.completeness_score(target_labels_all[0], target_cluster_result_all[0]))

def generate_cluster_with_domain_clean(cluster_result_path, dist=None, eps=0.8, min_samples=10, domain_num=5, data_dir=None, flag='did'):
    info_mat = []
    target_features_all = []
    target_labels_all = []
    target_names_all = []
    target_cluster_result_all = []
    target_cluster_num_all = []
    target_nodes_all = []
    for i in np.arange(domain_num):
        m = loadmat(str(i) + '_' + flag + '_' + data_dir + '_pytorch_target_result.mat')
        target_features = m['train_f']
        target_labels = m['train_label'][0]
        target_names = m['train_name']

        process_num = m['train_label'][0].shape[0]
        if dist is None:
            X = target_features[:process_num]
        else:
            X = dist[i][:process_num]
        labels_true = target_labels[:process_num]
        names = target_names[:process_num]
        print('Domain %d  DBSCAN starting ......' % i)
        db = DBSCAN(eps=eps[i], min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print(sorted(Counter(labels).values())[:10])
        print(sorted(Counter(labels).values())[-10:])
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels != -1)
        print('Estimated number of nodes: %d' % n_node)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

        target_features_all.append(target_features)
        target_labels_all.append(labels_true)
        target_names_all.append(target_names)
        target_cluster_result_all.append(labels)
        target_cluster_num_all.append(n_clusters_)
        sub_nodes = []
        for j in np.arange(n_clusters_):
            nodes = names[np.where(labels == j)]
            sub_nodes.append(nodes)
        target_nodes_all.append(sub_nodes)

    cnt = 0
    for i in np.arange(1, domain_num):
        for j in np.arange(target_cluster_num_all[0]):
            iou_max = 0.0
            index = -1
            for k in np.arange(target_cluster_num_all[i]):
                iou = len(np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][k])) / len(
                    np.union1d(target_nodes_all[0][j], target_nodes_all[i][k]))
                if iou > 1e-6:
                    # print('cnt = %3d  iou = %.3f' % (cnt, iou))
                    cnt += 1
                if iou > iou_max:
                    iou_max = iou
                    index = k
            if index != -1:
                disabled_nodes = np.setdiff1d(target_nodes_all[0][j], target_nodes_all[i][index])
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][index])
            else:
                disabled_nodes = target_nodes_all[0][j]
                ind = [np.where(target_names_all[0] == x)[0][0] for x in disabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = -1
                target_nodes_all[0][j] = []

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(len(target_nodes_all[0])):
        dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
        if len(target_nodes_all[0][i]) >= np.max((3, min_samples - 1)):
            os.mkdir(dir_path)
            files = target_nodes_all[0][i]
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
            dir_cnt += 1
            image_cnt += len(files)
    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(target_labels_all[0], target_cluster_result_all[0]))
    print("Completeness: %0.3f" % metrics.completeness_score(target_labels_all[0], target_cluster_result_all[0]))


def generate_cluster_with_clean_union(cluster_result_path, eps=0.5, min_samples=5, domain_num=2):
    info_mat = []
    target_features_all = []
    target_labels_all = []
    target_names_all = []
    target_cluster_result_all = []
    target_cluster_num_all = []
    target_nodes_all = []
    for i in np.arange(domain_num):
        m = loadmat(str(i) + '_duke' + '_pytorch_target_result.mat')
        target_features = m['train_f']
        target_labels = m['train_label'][0]
        target_names = m['train_name']

        process_num = m['train_label'][0].shape[0]
        X = target_features[:process_num]
        labels_true = target_labels[:process_num]
        names = target_names[:process_num]
        print('Domain %d  DBSCAN starting ......' % i)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print(sorted(Counter(labels).values())[:10])
        print(sorted(Counter(labels).values())[-10:])
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels != -1)
        print('Estimated number of nodes: %d' % n_node)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

        target_features_all.append(target_features)
        target_labels_all.append(labels_true)
        target_names_all.append(target_names)
        target_cluster_result_all.append(labels)
        target_cluster_num_all.append(n_clusters_)
        sub_nodes = []
        for j in np.arange(n_clusters_):
            nodes = names[np.where(labels == j)]
            sub_nodes.append(nodes)
        target_nodes_all.append(sub_nodes)

    cnt = 0
    for i in np.arange(1, domain_num):
        for j in np.arange(target_cluster_num_all[0]):
            iou_max = 0.0
            index = -1
            for k in np.arange(target_cluster_num_all[i]):
                iou = len(np.intersect1d(target_nodes_all[0][j], target_nodes_all[i][k])) / len(
                    np.union1d(target_nodes_all[0][j], target_nodes_all[i][k]))
                if iou > 1e-6:
                    # print('cnt = %3d  iou = %.3f' % (cnt, iou))
                    cnt += 1
                if iou > iou_max:
                    iou_max = iou
                    index = k
            if index != -1:
                enabled_nodes = np.setdiff1d(target_nodes_all[i][index], target_nodes_all[0][j])
                ind = [np.where(target_names_all[0] == x)[0][0] for x in enabled_nodes]
                for _ind in ind:
                    target_cluster_result_all[0][_ind] = target_cluster_result_all[0][j]
                target_nodes_all[0][j] = np.union1d(target_nodes_all[0][j], target_nodes_all[i][index])

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(len(target_nodes_all[0])):
        dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
        if len(target_nodes_all[0][i]) >= np.max((2, min_samples - 1)):
            os.mkdir(dir_path)
            files = target_nodes_all[0][i]
            for file in files:
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
            dir_cnt += 1
            image_cnt += len(files)
    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(target_labels_all[0], target_cluster_result_all[0]))
    print("Completeness: %0.3f" % metrics.completeness_score(target_labels_all[0], target_cluster_result_all[0]))


if __name__ == '__main__':
    # analysis_features()
    # cluster()
    # generate_cluster_data('data/duke/pytorch/train_all_cluster')
    generate_cluster_data_with_clean('data/duke/pytorch/train_all_cluster')
