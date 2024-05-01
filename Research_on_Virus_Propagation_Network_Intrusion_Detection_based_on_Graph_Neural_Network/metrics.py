from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix

def binary_evaluate(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    TP = matrix[0, 0]
    FN = matrix[0, 1]
    FP = matrix[1, 0]
    TN = matrix[1, 1]

    try: 
        acc = (TP + TN) / (TP + FP + FN + TN)
    except BaseException:
        acc = 0

    try:
        fpr = FP / (FP + TN)
    except BaseException:
        fpr = 0

    try:
        precision = TP / (TP + FP)
    except BaseException: 
        precision = 0

    try:
        recall = TP / (TP + FN)
    except BaseException:
        recall = 0

    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except BaseException:
        f1_score = 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fpr': fpr,
        'acc': acc
    }

def multi_evaluate(y_test, y_pred, idx2label): 
    acc_name = 'acc'
    fpr_name = 'fpr'
    p_name = 'precision'
    r_name = 'recall'
    f1_name = 'f1_score'

    multi_matrix = multilabel_confusion_matrix(y_test, y_pred)
    total_num = multi_matrix[0].sum()
    total_result = dict()
    for i in range(len(multi_matrix)):
        try:
            acc_tmp = (multi_matrix[i][0,0] + multi_matrix[i][1,1]) / total_num
        except BaseException:
            acc_tmp = 0

        try:
            fpr_tmp = multi_matrix[i][1,0] / (multi_matrix[i][1,0] + multi_matrix[i][1,1])
        except BaseException:
            fpr_tmp = 0

        try:
            p_tmp = multi_matrix[i][0,0] / (multi_matrix[i][0,0] + multi_matrix[i][1,0])
        except BaseException:
            p_tmp = 0
            
        try:    
            r_tmp = multi_matrix[i][0,0] / (multi_matrix[i][0,0] + multi_matrix[i][0,1])
        except BaseException:
            r_tmp = 0
        
        try:
            f1_tmp = 2 * p_tmp * r_tmp / (p_tmp + r_tmp)
        except BaseException:
            f1_tmp = 0

        result = {
            acc_name: acc_tmp,
            fpr_name: fpr_tmp,
            p_name: p_tmp,
            r_name: r_tmp,
            f1_name: f1_tmp
        }
        total_result[idx2label[i]] = result
    return total_result