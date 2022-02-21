import numpy as np


def cal_hamming_dis(b1, b2):
    k = b2.shape[1]  # length of hash code
    dis = 0.5 * (k - np.dot(b1, b2.transpose()))
    return dis


def cal_map(retrieval_binary, query_binary, retrieval_label, query_label, top_k):
    """
    Calculate MAP (Mean Average Precision)
    :param retrieval_binary: binary code of database
    :param query_binary: binary code of query sample
    :param retrieval_label:
    :param query_label:
    :param top_k:
    :return:
    """
    query_number = query_label.shape[0]
    top_k_map = 0

    for query_index in range(query_number):
        ground_truth = (np.dot(query_label[query_index, :], retrieval_label.transpose()) > 0).astype(
            np.float32)  # (1, N)
        hamming_dis = cal_hamming_dis(query_binary[query_index, :], retrieval_binary)  # (1, N)

        # sort hamming distance
        sort_index = np.argsort(hamming_dis)

        # resort ground truth
        ground_truth = ground_truth[sort_index]

        # get top K ground truth
        top_k_gnd = ground_truth[0:top_k]
        top_k_sum = np.sum(top_k_gnd).astype(int)  # the number of correct retrieval in top K
        if top_k_sum == 0:
            continue
        count = np.linspace(1, top_k_sum, top_k_sum)

        top_k_index = np.asarray(np.where(top_k_gnd == 1)) + 1.0
        top_k_map += np.mean(count / top_k_index)  # average precision of per class

    return top_k_map / query_number  # mean of average precision of all class