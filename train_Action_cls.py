"""
* @name: AVE_train.py
* @description: Training script for AVE multi-modal emotion recognition dataset
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from opts import *
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from core.dataset_ave import AVEDataLoader
from core.dataset_ks import KSDataLoader
from core.dataset_ucf51 import UCF51DataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed, save_metrics_to_txt_aligned
import importlib
from core.metric import AVEMetric
import logging

import psutil

def get_cpu_mem_gb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss 
    return mem_bytes / (1024 ** 3)


# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For storing best metrics
best_val_global = {'epoch': 0, 'val': 0.0, 'test': 0.0, 'F1': 0.0, 'top1': 0, 'top3': 0, 'top5': 0, 'val_loss': float('inf')}
best_test_global = {'epoch': 0, 'val': 0.0, 'test': 0.0, 'F1': 0.0, 'top1': 0, 'top3': 0, 'top5': 0, 'test_loss': float('inf')}


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




def train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics):

    losses = AverageMeter()
    model.train()

    all_preds = []
    all_targets = []

    for data in tqdm(train_loader, desc="Training", leave=False):

        text = data['time_text_bert'].to(device)
        audio = data['audio'].to(device)
        vision = data['vision'].to(device)
        audio_clue = data['audio_clue_bert'].to(device)
        visual_clue = data['visual_clue_bert'].to(device)
        label = data['label'].to(device)             # (B,)


        logits = model(
            text=text, audio=audio, visual=vision,
            audio_clue=audio_clue, visual_clue=visual_clue
        )  # (B, C)

        loss = loss_fn(logits, label)
        losses.update(loss.item(), label.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(label.detach().cpu().numpy())


    metrics.reset()
    metrics.update(
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0)
    )
    results = metrics.compute()
    results['loss'] = losses.value_avg

    writer.add_scalar('train/loss', losses.value_avg, epoch)
    return results





def test(model, test_loader, optimizer, loss_fn, epoch, writer, metrics, save_path):

    losses = AverageMeter()
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", leave=False):

            text = data['time_text_bert'].to(device)
            audio = data['audio'].to(device)
            vision = data['vision'].to(device)
            audio_clue = data['audio_clue_bert'].to(device)
            visual_clue = data['visual_clue_bert'].to(device)
            label = data['label'].to(device)             # (B,)


            logits = model(text=text, audio=audio, visual=vision, 
                           audio_clue=audio_clue, visual_clue=visual_clue)

            loss = loss_fn(logits, label)
            losses.update(loss.item(), label.size(0))


            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(label.detach().cpu().numpy())


        metrics.reset()
        metrics.update(
            np.concatenate(all_preds, axis=0),
            np.concatenate(all_targets, axis=0)
        )
        results = metrics.compute()
        results['loss'] = losses.value_avg

        writer.add_scalar('test/loss', losses.value_avg, epoch)
    return results





def main(opt, device):

    print("\n=================== 打印其他信息 ===========================")
    print("device:  {}".format(device))
    print("Datsest: ", opt.datasetName)
    print("note:    ", opt.note)
    print("model:   ", opt.model_name)
    print("token_len:   ", opt.token_len)
    print("truncated:   ", opt.need_truncated)
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


    # check data path
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


    # Load data
    if opt.datasetName == "AVE":
        dataLoader = AVEDataLoader(opt)
        opt.cls_num = 28
        opt.result_txtPath = "best_AVE_28.txt"
    elif opt.datasetName == "KS":
        dataLoader = KSDataLoader(opt)
        opt.cls_num = 31
        opt.result_txtPath = "best_KS_31.txt"
    elif opt.datasetName == "UCF51":
        dataLoader = UCF51DataLoader(opt)
        opt.cls_num = 51
        opt.result_txtPath = "best_UCF_51.txt"
    else:
        raise ValueError("Unsupported dataset: {}".format(opt.datasetName))


    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=opt.lr, 
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = AVEMetric(topk=(1,3,5))

    writer = SummaryWriter(logdir=log_path)


    print("\nStarting training...", opt.epochs, "epochs")
    epoch_pbar = tqdm(range(1, opt.epochs+1))
    for epoch in epoch_pbar:
        train_results = train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer, metric)
        if opt.datasetName == "AVE":
            val_results = test(model, dataLoader['valid'], optimizer, loss_fn, epoch, writer, metric, save_path)
        else:
            val_results = {'accuracy': 0.0, 'f1_score': 0.0, 'loss': 0.0}  # For KS and UCF51, we do not have a validation set.
        test_results = test(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, metric, save_path)
        scheduler_warmup.step()

        epoch_pbar.set_description(opt.datasetName)


        if val_results.get('loss', 0) <= val_best['val_loss']:
            val_best['val'] = val_results.get('accuracy', 0)
            val_best['val_loss'] = val_results.get('loss', 0)
            val_best['epoch'] = epoch
            val_best['F1'] = test_results.get('f1_score', 0)
            val_best['test'] = test_results.get('accuracy', 0)

        
        if test_results.get('loss', 0) <= test_best['test_loss']:
            test_best['test'] = test_results.get('accuracy', 0)
            test_best['test_loss'] = test_results.get('loss', 0)
            test_best['epoch'] = epoch
            test_best['val'] = val_results.get('accuracy', 0)
            test_best['F1'] = test_results.get('f1_score', 0)




        if val_results.get('loss', 0) <= best_val_global['val_loss']:
            best_val_global['val'] = val_results.get('accuracy', 0)
            best_val_global['val_loss'] = val_results.get('loss', 0)
            best_val_global['epoch'] = epoch
            best_val_global['F1'] = test_results.get('f1_score', 0)
            best_val_global['test'] = test_results.get('accuracy', 0)

            if opt.datasetName == "AVE":
                if os.path.exists(save_path):
                    for file in os.listdir(save_path):
                        if file.endswith('.pth'):
                            os.remove(os.path.join(save_path, file))
                save_model(save_path, epoch, model, optimizer)


        if test_results.get('loss', 0) <= best_test_global['test_loss']:
            best_test_global['test'] = test_results.get('accuracy', 0)
            best_test_global['test_loss'] = test_results.get('loss', 0)
            best_test_global['epoch'] = epoch
            best_test_global['val'] = val_results.get('accuracy', 0)
            best_test_global['F1'] = test_results.get('f1_score', 0)

            if opt.datasetName != "AVE": 
                if os.path.exists(save_path):
                    for file in os.listdir(save_path):
                        if file.endswith('.pth'):
                            os.remove(os.path.join(save_path, file))
                save_model(save_path, epoch, model, optimizer)


        
        epoch_pbar.set_postfix({
            'val_acc': f"{val_results['accuracy']:.4f}", # per epoch validation accuracy
            'test_acc': f"{test_results['accuracy']:.4f}",
            # 'test_f1': f"{test_results['f1_score']:.4f}",
            'vBest_acc': f"{val_best['test']:.4f}", # per training best validation accuracy
            'tBest_acc': f"{test_best['test']:.4f}",
        })


    writer.close()
   
    txt_path = f'./{opt.result_txtPath}'
    save_metrics_to_txt_aligned(txt_path, val_best, test_best, title=f"{opt.datasetName}_{opt.vision_feats}_{opt.audio_feats}_____{opt.model_name}_____{opt.note}", has_val=True if opt.datasetName=="AVE" else False)







if __name__ == "__main__":
    opt = parse_opts()

    # opt.datasetName = "AVE"
    # opt.dataPath = ""
    # opt.result_txtPath = ""
    # opt.batch_size = 64
    # opt.lr = 1e-4
    # opt.weight_decay = 1e-5
    opt.CUDA_VISIBLE_DEVICES = opt.device
    opt.num_experts = 5
    opt.fusion_layer_depth = 2
    opt.top_k = 3
    opt.capacity_factor = 1.0
    # opt.dropout = 0.0
    opt.use_bert = True
    # opt.need_truncated = False
    opt.need_data_aligned = True
    opt.need_normalize = False
    # opt.cls_num = 29


    # if opt.token_len == 50:
    #     opt.seq_lens = [50, 50, 50]
    # elif opt.token_len == 128:
    #     opt.seq_lens = [128, 128, 128]

    opt.seq_lens = [50, 50, 50]


    # opt.device = 3
    # opt.epochs = 30

    # opt.vision_feats = "resnet2plus1D"
    # opt.audio_feats = "librosa"
    # opt.model_name = "AVE_TEXT_fullyS"



    # Load data
    if opt.datasetName == "AVE":
        opt.cls_num = 28
        opt.result_txtPath = "best_AVE_28.txt"
    elif opt.datasetName == "KS":
        opt.cls_num = 31
        opt.result_txtPath = "best_KS_31.txt"
    elif opt.datasetName == "UCF51":
        opt.cls_num = 51
        opt.result_txtPath = "best_UCF_51.txt"
    else:
        raise ValueError("Unsupported dataset: {}".format(opt.datasetName))


    device = torch.device('cuda:{}'.format(opt.CUDA_VISIBLE_DEVICES) if torch.cuda.is_available() else 'cpu')



    log_path = f"./log/{opt.datasetName}---{opt.vision_feats}_{opt.audio_feats}"
    os.makedirs(log_path, exist_ok=True)
    model_name = f"{opt.model_name}---{opt.note}"
    sys.stdout = Logger(os.path.join(log_path, model_name + '.log'))
    print(f"Log file name (modelname): {model_name}.log\n")



    run_times = 1
    avg_BaseTest_test_acc = 0.0
    avg_BaseVal_test_acc = 0.0
    print(f"\n=================== Running {run_times} Experiments ===================")

    for i in range(run_times):
            
        val_best = {'epoch': 0, 'val': 0.0, 'test': 0.0, 'F1': 0.0, 'top1': 0, 'top3': 0, 'top5': 0, 'val_loss': float('inf')}
        test_best = {'epoch': 0, 'val': 0.0, 'test': 0.0, 'F1': 0.0, 'top1': 0, 'top3': 0, 'top5': 0, 'test_loss': float('inf')}


        print(f"\n\n=================== Experiment run {i+1} ===================")
        main(opt, device)

        avg_BaseTest_test_acc += test_best['test']
        avg_BaseVal_test_acc += val_best['test']


        print("\n=================== Best Results for This Run ===================")

        print("\nbest_val :", val_best['epoch'], 
            "Val Acc:", val_best['val'], 
            "              Test Acc:", val_best['test'],
            "Test F1:", val_best['F1'],
            "val_loss:", val_best['val_loss'],
            "Test Top-1:", val_best['top1'],
            "Test Top-3:", val_best['top3'],
            "Test Top-5:", val_best['top5'])
        
        print("best_test: ", test_best['epoch'], 
            "Val Acc:", test_best['val'], 
            "              Test Acc:", test_best['test'],
            "Test F1:", test_best['F1'],
            "test_loss:", test_best['test_loss'],
            "Test Top-1:", test_best['top1'],
            "Test Top-3:", test_best['top3'],
            "Test Top-5:", test_best['top5'])


        print("\n=================== Final Best Results ===================")
        print("\nbest_val_global :", best_val_global['epoch'], 
            "Val Acc:", best_val_global['val'], 
            "              Test Acc:", best_val_global['test'],
            "Test F1:", best_val_global['F1'],
            "val_loss:", best_val_global['val_loss'],
            "Test Top-1:", best_val_global['top1'],
            "Test Top-3:", best_val_global['top3'],
            "Test Top-5:", best_val_global['top5'])
        
        print("best_test_global: ", best_test_global['epoch'], 
            "Val Acc:", best_test_global['val'], 
            "              Test Acc:", best_test_global['test'],
            "Test F1:", best_test_global['F1'],
            "test_loss:", best_test_global['test_loss'],
            "Test Top-1:", best_test_global['top1'],
            "Test Top-3:", best_test_global['top3'],
            "Test Top-5:", best_test_global['top5'])
        

