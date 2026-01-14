'''
* @name: metric.py
* @description: Evaluation metrics. Note: The code source from MMSA (https://github.com/thuiar/MMSA/tree/master).
'''

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


__all__ = ['MetricsTop', 'AVEMetric']


class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
            'MOSI': self.__eval_mosi_regression,
            'MOSEI': self.__eval_mosei_regression,
            'SIMS': self.__eval_sims_regression,
            'SIMS2': self.__eval_sims2_regression
        }
    

    def __multiclass_acc(self, y_pred, y_true):
        """
        计算多类准确率相对于真实值
        :param preds: 表示预测值的浮点数组，维度 (N,)
        :param truths: 表示真实类别的浮点/整数数组，维度 (N,)
        :return: 分类准确率
        """

        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        # -3 -2 -1 0 1 2 3
        # [-3, -2.5)、[-2.5, -1.5]、(-1.5, -0.5)、[-0.5, 0.5]、(0.5, 1.5)、[1.5, 2.5]、(2.5, 3]
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))   
        # np.round() 四舍五入，为什么要round呢？ 因为预测值和真实值都是浮点数，所以需要四舍五入
    

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()  
        # view()函数作用是将一个多行的Tensor拼接成一行,cpu()函数将数据从GPU转到CPU
        # detach()函数返回一个新的没有连接的Tensor，numpy()函数将Tensor转换为numpy数组
        test_truth = y_true.view(-1).cpu().detach().numpy()

        # np.clip()函数将数组中的元素限制在一个范围内
        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)


        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths, 预测值和真实值之间的平均绝对误差
        corr = np.corrcoef(test_preds, test_truth)[0][1]  # Correlation Coefficient, 皮尔逊相关系数
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_preds, non_zeros_binary_truth, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_preds, binary_truth, average='weighted')
        
        eval_results = {
            "Has0_acc_2":  round(acc2, 4),  # 保留四位小数
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results


    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_preds_a2, test_truth_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": round(mult_a2, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }

        return eval_results
    
    def __eval_sims2_regression(self, y_pred, y_true):
        return self.__eval_sims_regression(y_pred, y_true)
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]






class AVEMetric:
    def __init__(self, topk=None):
        self.topk = topk
        self.reset()

    def reset(self):
        """清空缓存（每个 epoch 调一次）"""
        self.y_pred = []
        self.y_true = []
        self.y_pred_probs = []  
        
    def update(self, y_pred, y_true, y_pred_probs=None):
        """
        流式更新
        :param y_pred: numpy array, (B,)
        :param y_true: numpy array, (B,)
        :param y_pred_probs: optional numpy array, (B, C)
        """
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

        if y_pred_probs is not None:
            self.y_pred_probs.append(y_pred_probs)

    def compute(self):
        """epoch 结束后统一计算指标"""
        y_pred = np.concatenate(self.y_pred, axis=0)
        y_true = np.concatenate(self.y_true, axis=0)

        results = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred, average='weighted'), 4)
        }

        if len(self.y_pred_probs) > 0:
            y_pred_probs = np.concatenate(self.y_pred_probs, axis=0)
            for k in self.topk:
                try:
                    acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)
                except Exception:
                    acc = 0.0
                results[f"top{k}_acc"] = round(acc, 4)

        return results





