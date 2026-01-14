import torch
from opts import *
import importlib
from tqdm import tqdm
import numpy as np

from core.dataset_ave import AVEDataLoader
from core.dataset_ks import KSDataLoader
from core.dataset_ucf51 import UCF51DataLoader
from core.utils import AverageMeter, setup_seed
from core.metric import AVEMetric


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# Dataset & Checkpoint
# =========================
dataPath = {
    'KS': './datasets/KS_rms_k16_last.pkl',
    'AVE': './datasets/AVE_rms_k16_last.pkl',
    'UCF51': './datasets/UCF51_rms_k16_last.pkl'
}


checkpointPath = {
    'KS': '',
    'AVE': '',
    'UCF51': '',
}



# =========================
# Test
# =========================
for dataset in checkpointPath.keys():

    opt = parse_opts()
    print("\n" + "#" * 120)
    print(f"Testing on {dataset}")
    print("#" * 120)

    # ===== basic config =====
    opt.datasetName = dataset
    opt.dataPath = dataPath[dataset]
    opt.test_checkpoint = checkpointPath[dataset]
    opt.is_test = 1

    opt.use_bert = True
    opt.need_truncated = False
    opt.need_data_aligned = True
    opt.need_normalize = False

    opt.seq_lens = [128, 128, 128]
    opt.CUDA_VISIBLE_DEVICES = opt.device

    if dataset == "AVE":
        opt.cls_num = 28
    elif dataset == "KS":
        opt.cls_num = 31
    elif dataset == "UCF51":
        opt.cls_num = 51

    device = torch.device(f'cuda:{opt.device}')
    print("Device:", device)

    if opt.seed is not None:
        setup_seed(opt.seed)
    print("Seed:", opt.seed)

    print("Model:", opt.model_name)
    print("Checkpoint:", opt.test_checkpoint)
    print("Batch size:", opt.batch_size)

    # =========================
    # Load model
    # =========================
    model_module = importlib.import_module("models.Trinity")
    model_class = getattr(model_module, opt.model_name)

    model = model_class(
        dataset=opt.datasetName,
        fusion_layer_depth=opt.fusion_layer_depth,
        num_experts=opt.num_experts,
        top_k=opt.top_k,
        capacity_factor=opt.capacity_factor,
        dropout=opt.dropout,
        cls_num=opt.cls_num,
        token_len=opt.token_len,
        vision_feats=opt.vision_feats,
        audio_feats=opt.audio_feats
    ).to(device)

    print("Trainable parameters:", count_parameters(model))

    checkpoint = torch.load(opt.test_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # =========================
    # DataLoader
    # =========================
    if dataset == "AVE":
        dataLoader = AVEDataLoader(opt, mode='test')
    elif dataset == "KS":
        dataLoader = KSDataLoader(opt, mode='test')
    elif dataset == "UCF51":
        dataLoader = UCF51DataLoader(opt, mode='test')
    else:
        raise ValueError("Unsupported dataset")

    test_loader = dataLoader['test']

    # =========================
    # Metric
    # =========================
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = AVEMetric(topk=(1, 3, 5))
    losses = AverageMeter()

    all_preds, all_targets = [], []

    # =========================
    # Testing loop
    # =========================
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing on {dataset}"):

            text = data['time_text_bert'].to(device)
            audio = data['audio'].to(device)
            vision = data['vision'].to(device)
            audio_clue = data['audio_clue_bert'].to(device)
            visual_clue = data['visual_clue_bert'].to(device)
            label = data['label'].to(device)   # (B,)

            logits = model(
                text=text,
                audio=audio,
                visual=vision,
                audio_clue=audio_clue,
                visual_clue=visual_clue
            )

            loss = loss_fn(logits, label)
            losses.update(loss.item(), label.size(0))

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(label.cpu().numpy())

    # =========================
    # Metric compute
    # =========================
    metric.reset()
    metric.update(
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0)
    )
    results = metric.compute()
    results['loss'] = losses.value_avg

    # =========================
    # Print results
    # =========================
    print("\n=================== Test Results ===================")
    for k, v in results.items():
        print(f"{k:<12}: {v:.4f}")
