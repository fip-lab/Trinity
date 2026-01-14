'''
* @name: utils.py
* @description: Other functions.
'''


import os
import random
import numpy as np
import torch


class AverageMeter(object): 
    def __init__(self):    
        self.value = 0      
        self.value_avg = 0  
        self.value_sum = 0  
        self.count = 0      

    def reset(self):     
        self.value = 0      
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count): 

        self.value = value
        self.value_sum += value * count  
        self.count += count         
        self.value_avg = self.value_sum / self.count 


def setup_seed(seed):
    torch.manual_seed(seed)        
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)       
    random.seed(seed)            
    torch.backends.cudnn.deterministic = True 



def save_model(save_path, epoch, model, optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'epoch_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)


def best_result(test_results_list):

    best_results = {}
    for key in test_results_list[0].keys():
        # best_value = max(test_results_list, key=lambda x: x[key]) if key != 'MAE' else min(test_results_list, key=lambda x: x[key])
        best_value = max(
            enumerate(test_results_list), 
            key=lambda x: (x[1][key], x[0]) if key != 'MAE' else (-x[1][key], x[0])
        )[1] if key != 'MAE' else min(
            enumerate(test_results_list), 
            key=lambda x: (x[1][key], -x[0])
        )[1]

        best_epoch = test_results_list.index(best_value) + 1
        best_results[key] = (best_value[key], best_epoch)


    return best_results


def best_result_val(val_results_list, test_results_list):

    best_results = {}
    for key in val_results_list[0].keys():
        best_epoch_val = max(
            enumerate(val_results_list),
            key=lambda x: (x[1][key], x[0]) if key != 'MAE' else (-x[1][key], x[0])
        )[1] if key != 'MAE' else min(
            enumerate(val_results_list),
            key=lambda x: (x[1][key], -x[0])
        )[1]

        best_epoch = val_results_list.index(best_epoch_val) + 1
        best_value = test_results_list[best_epoch - 1][key]
        best_results[key] = (best_value, best_epoch)

        # print(best_epoch, best_epoch_val, best_value)
        # print(best_results, '\n')
    return best_results





def save_metrics_to_txt_aligned(file_path, metrics_dict_base_val, metrics_dict_base_test, title=None, has_val=True):
    """
    Save metrics dictionary to a txt file with names in the first line
    and values in the second line, aligned in columns.
    """

    def format_value(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)


    keys_val = list(metrics_dict_base_val.keys())
    keys_test = list(metrics_dict_base_test.keys())

    values_val = [format_value(metrics_dict_base_val[key]) for key in keys_val]
    values_test = [format_value(metrics_dict_base_test[key]) for key in keys_test]

    col_widths_val = [max(len(k), len(v)) + 5 for k, v in zip(keys_val, values_val)]
    col_widths_test = [max(len(k), len(v)) + 5 for k, v in zip(keys_test, values_test)]


    with open(file_path, "a") as f:
        if title:
            f.write(f"{title}\n")

        # 第一行：指标名称
        for key, width in zip(keys_test, col_widths_test):
            f.write(key.ljust(width))
        f.write("\n")

        if has_val:
            # 第二行：指标值 val
            for value, width in zip(values_val, col_widths_val):
                f.write(value.ljust(width))
            f.write(f"{title}==base_val.")
            f.write("\n")

        # 第二行：指标值 test
        for value, width in zip(values_test, col_widths_test):
            f.write(value.ljust(width))
        f.write(f"{title}==base_test.")
        f.write("\n\n\n")





        

        