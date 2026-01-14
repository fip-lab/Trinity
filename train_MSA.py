import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from opts import *
from core.dataset_msa import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed, best_result, best_result_val
from tensorboardX import SummaryWriter
import importlib
from core.metric import MetricsTop
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



val_results_list = []
test_results_list = []

val_global_acc2_best = {'epoch': 0, 'val_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0, 'MAE':10000}, 'test_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0, 'MAE':10000}}


class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()


def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
    return res


def main(opt, device):

    print("\n=================== 打印其他信息 ===========================")
    print("device:  {}".format(device))
    print("Datsest: ", opt.datasetName)
    print("note:    ", opt.note)
    print("model:   ", opt.model_name)
    print("token_len:   ", opt.token_len)
    print("truncated:   ", opt.need_truncated)
    print("text_bert:   ", opt.text_bert)
    print("\n")

    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))

    # tensorboardX
    log_path = os.path.join(".", "log", opt.project_name)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    print("log_path :", log_path)
    

    # model save path
    save_name = f"{opt.datasetName}---{opt.vision_feats}_{opt.audio_feats}"
    save_path = os.path.join(opt.models_save_root, save_name, f"{opt.model_name}---{opt.note}")
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    print("model_save_path :", save_path)


    print("datapath: {}".format(opt.dataPath))
    if  os.path.exists(opt.dataPath):
        print("Data path exists!")


    time.sleep(0.1)
    model_module = importlib.import_module("models.Trinity")
    model_class = getattr(model_module, opt.model_name)

    model = model_class(dataset=opt.datasetName,
                        fusion_layer_depth=opt.fusion_layer_depth,
                        num_experts=opt.num_experts,
                        top_k=opt.top_k,
                        capacity_factor=opt.capacity_factor,
                        dropout=opt.dropout,
                        cls_num=opt.cls_num,
                        token_len=opt.token_len,
                        vision_feats=opt.vision_feats,
                        audio_feats=opt.audio_feats).to(device)

    print("Number of trainable parameters:", count_parameters(model))
    print(model.__class__.__name__)


    dataLoader = MMDataLoader(opt)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)
    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)

    writer = SummaryWriter(logdir=log_path)


    print("\nStarting training...", opt.epochs, "epochs")
    epoch_pbar = tqdm(range(1, opt.epochs+1))

    per_train_acc2_best = {'epoch': 0, 'val_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0, 'MAE':10000}, 'test_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0, 'MAE':10000}}

    for epoch in epoch_pbar:
        train_metric = train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer, metrics)
        eval_metric = evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, writer, save_path, metrics)
        if opt.is_test is not None:
            test_metric = test(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, metrics, save_path)
        scheduler_warmup.step()


        acc2_val = 'Has0_acc_2' if 'Has0_acc_2' in eval_metric else 'Mult_acc_2'
        # Update the best results for per_train_acc2_best
        if eval_metric[acc2_val] >= per_train_acc2_best['val_result'][acc2_val]:
            per_train_acc2_best['epoch'] = epoch
            per_train_acc2_best['val_result'] = eval_metric
            per_train_acc2_best['test_result'] = test_metric



        # Update the best results for val_global_acc2_best
        if eval_metric[acc2_val] >= val_global_acc2_best['val_result'][acc2_val]:
            val_global_acc2_best['epoch'] = epoch
            val_global_acc2_best['val_result'] = eval_metric
            val_global_acc2_best['test_result'] = test_metric

            # Check if the save path exists, if it does, delete the model
            if os.path.exists(save_path):
                for file in os.listdir(save_path):
                    if file.endswith('.pth'):
                        os.remove(os.path.join(save_path, file))
            save_model(save_path, epoch, model, optimizer)

        
        epoch_pbar.set_description(opt.datasetName) 
        epoch_pbar.set_postfix({
            f"Epoch": per_train_acc2_best['epoch'],
            f"Bval_test": per_train_acc2_best['test_result']
        })



    # # write results to txt file
    # with open(opt.result_txtPath, 'a') as f:  
    #     f.write("======:   ")
    #     for key, value in per_train_acc2_best['test_result'].items():
    #         f.write(f"{value:<15.4f}")
    #     f.write(f"{opt.datasetName:<8}【{opt.model_name}__{opt.note}】{per_train_acc2_best['epoch']}\n")

    # print('\nper_train_acc2_best epoch: ', per_train_acc2_best['epoch'], per_train_acc2_best['test_result'])
    # writer.close()






def train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics):    
    # train_pbar = enumerate(train_loader)

    losses = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    #  text, audio, visual, summary_clue, audio_clue, visual_clue
    # for cur_iter, data in train_pbar:
    for data in tqdm(train_loader, desc="Training", leave=False):

        text, audio, img, audio_clue, visual_clue = data['text'].to(device), data['audio'].to(device), data['vision'].to(device), data['audio_clue_bert'].to(device), data['visual_clue_bert'].to(device)

        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        output = model(text=text, audio=audio, visual=img, audio_clue=audio_clue, visual_clue=visual_clue)

        loss = loss_fn(output, label)
        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)

    writer.add_scalar('train/loss', losses.value_avg, epoch)
    return train_results





def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path, metrics):
    # eval_pbar = enumerate(eval_loader)

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        # for cur_iter, data in eval_pbar:
        for data in tqdm(eval_loader, desc="Evaluating", leave=False):
            text, audio, img, audio_clue, visual_clue = data['text'].to(device), data['audio'].to(device), data['vision'].to(device), data['audio_clue_bert'].to(device), data['visual_clue_bert'].to(device)

            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(text=text, audio=audio, visual=img, audio_clue=audio_clue, visual_clue=visual_clue)  # 前向传播

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())


        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = metrics(pred, true)
        val_results_list.append(eval_results)

        writer.add_scalar('evaluate/loss', losses.value_avg, epoch)

    return eval_results





def test(model, test_loader, optimizer, loss_fn, epoch, writer, metrics, save_path):
    # test_pbar = enumerate(test_loader)

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        # for cur_iter, data in test_pbar:
        for data in tqdm(test_loader, desc="Testing", leave=False):
            # img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            text, audio, img, audio_clue, visual_clue = data['text'].to(device), data['audio'].to(device), data['vision'].to(device), data['audio_clue_bert'].to(device), data['visual_clue_bert'].to(device)

            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            # output = model(img, audio, text)
            output = model(text=text, audio=audio, visual=img, audio_clue=audio_clue, visual_clue=visual_clue)

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        test_results_list.append(test_results)

        writer.add_scalar('test/loss', losses.value_avg, epoch)
    
    return test_results





                ###################################### main ######################################
                ###################################### main ######################################
                ###################################### main ######################################





if __name__ == '__main__':
    
    dataPath = {
            'mosei': './datasets/mosei_unaligned_50_reason.pkl',
    }

    SMoE_layers = {
        'mosei': 2,
    }


    result_txtPath = {
        'mosei': './best_MOSEI.txt',
    }


    for data in dataPath.keys():

        opt = parse_opts()
        print("############################################################################### {} ###############################################################################".format(data))
        opt.use_bert = True
        # opt.need_truncated = False
        opt.need_data_aligned = True
        opt.need_normalize = False
        opt.datasetName = data
        opt.dataPath = dataPath[data]
        opt.CUDA_VISIBLE_DEVICES = opt.device
        opt.is_test = 1
        # opt.batch_size = 64
        opt.result_txtPath = result_txtPath[data]
        opt.fusion_layer_depth = SMoE_layers[data]

        opt.cls_num = 1
        opt.text_bert = "text_residual_bert"

        opt.seq_lens = [50, 50, 50]

        device = torch.device('cuda:{}'.format(opt.CUDA_VISIBLE_DEVICES) if torch.cuda.is_available() else 'cpu')


        log_path = f"./log/{opt.datasetName}---{opt.vision_feats}_{opt.audio_feats}"
        os.makedirs(log_path, exist_ok=True)
        model_name = f"{opt.model_name}---{opt.note}"
        sys.stdout = Logger(os.path.join(log_path, model_name + '.log'))
        print(f"Log file name (modelname): {model_name}.log\n")


        run_times = 1
        print(f"\n=================== Running {run_times} Experiments ===================")
        for i in range(run_times):

            print(f"\n=================== Experiment run {i+1} ===================")

            main(opt, device)


            bast_results_base_val = best_result_val(val_results_list, test_results_list)

            # with open(opt.result_txtPath, 'a') as f:
            #     f.write("Metrrr:   ")
            #     for key, (value, epoch) in bast_results_base_val.items():
            #         f.write(f"{value:<5.4f} -- {epoch:<5}")
            #     f.write(f"{opt.datasetName:<8}【{opt.model_name}__{opt.note}】")
            #     f.write("\n")

            
            # with open(opt.result_txtPath, 'a') as f:
            #     f.write("Resuuu:   ")
            #     for key, value in val_global_acc2_best['test_result'].items():
            #         f.write(f"{value:<15.4f}")
            #     f.write(f"{opt.datasetName:<8}【{opt.model_name}__{opt.note}】{val_global_acc2_best['epoch']}\n\n")

            
            print('val_global_acc2_best epoch: ', val_global_acc2_best['epoch'], val_global_acc2_best['test_result'], '\n\n')

            val_results_list = []
            test_results_list = []





        with open(opt.result_txtPath, 'a') as f:
            f.write("########################################## {} \n".format(data))
            f.write("Metric:   ")
            for key in bast_results_base_val.keys():
                f.write(f"{key:<15}")
            f.write("\n")


        with open(opt.result_txtPath, 'a') as f:
            f.write("Result:   ")
            for key, value in val_global_acc2_best['test_result'].items():
                f.write(f"{value:<15.4f}")
            f.write(f"{opt.datasetName:<8}【{opt.model_name}__{opt.note}】{val_global_acc2_best['epoch']}\n\n\n\n\n")

        
        print('\nval_global_acc2_best epoch: ', val_global_acc2_best['epoch'], val_global_acc2_best['test_result'], '\n\n\n\n\n\n')


        val_global_acc2_best = {'epoch': 0, 'val_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0, 'MAE':10000}, 'test_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0, 'MAE':10000}}
