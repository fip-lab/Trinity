import torch
from opts import *
from core.dataset_msa import MMDataset, MMDataLoader
from core.utils import AverageMeter, best_result, setup_seed
import importlib
from core.metric import MetricsTop
from tqdm import tqdm


def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
    return res


dataPath = {
        'mosei': './datasets/mosei_unaligned_50_reason.pkl',
}

SMoE_layers = {
        'mosei': 2,
}

checjpointPath = {
        'mosei': '',
}



test_results_list = []

for data in checjpointPath.keys():
    opt = parse_opts()
    print("################################################################ {} ################################################################".format(data))
    opt.use_bert = True
    opt.need_truncated = False
    opt.need_data_aligned = True
    opt.need_normalize = False
    opt.datasetName = data
    opt.dataPath = dataPath[data]
    opt.CUDA_VISIBLE_DEVICES = opt.device
    opt.seq_lens = [50, 50, 50]
    opt.is_test = 1
    # opt.batch_size = 64

    opt.fusion_layer_depth = SMoE_layers[data]
    opt.text_bert = 'text_residual_bert'
    opt.test_checkpoint = checjpointPath[data]


    device = torch.device('cuda:{}'.format(opt.device))
    print(device)

    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))


    print(f"model_name: ", opt.model_name)
    print(f"note: ", opt.note)
    print(f"dataset: ", opt.datasetName)
    print(f"dataPath: ", opt.dataPath)
    print(f"test_checkpoint: ", opt.test_checkpoint)
    print(f"fusion_layer_depth: ", opt.fusion_layer_depth)
    print(f"batch_size: ", opt.batch_size)


    # print("======================= 打印参数信息 =======================")
    # for key, value in vars(opt).items():
    #     print(key, value)


    # 动态导入模型类
    model_module = importlib.import_module("models.Trinity")
    model_class = getattr(model_module, opt.model_name)

    # 动态初始化模型
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

    checkpoint = torch.load(opt.test_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    # print(model)

    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)
    # dataset = MMDataset(opt, mode='test')
    dataLoader = MMDataLoader(opt, mode='test')
    dataset = dataLoader['test']


    test_pbar = enumerate(dataset)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        # for cur_iter, data in test_pbar:
        for cur_iter, data in tqdm(test_pbar, total=len(dataset), desc="Testing on {}".format(opt.datasetName)):

            text, audio, img, audio_clue, visual_clue = data['text'].to(device), data['audio'].to(device), data['vision'].to(device), data['audio_clue_bert'].to(device), data['visual_clue_bert'].to(device)

            # text, audio, img, audio_clue, visual_clue = text.unsqueeze(0), audio.unsqueeze(0), img.unsqueeze(0), audio_clue.unsqueeze(0), visual_clue.unsqueeze(0)

            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]


            output = model(text=text, audio=audio, visual=img, audio_clue=audio_clue, visual_clue=visual_clue)
            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())


        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        test_results_list.append(test_results)



    print("\n =================== print result =================== \n")
    best_results = best_result(test_results_list)

    print("指标：   ", end="")
    for key in best_results.keys():
        print(f"{key:<15}", end="")
    print()

    print("val_best_acc2:    ", end="")
    for key, (value, epoch) in best_results.items():
        print(f"{value:<15.4f}", end="")
    print(f"{opt.datasetName}  -- {opt.note}")
    print(f"\n\n\n")

    test_results_list = []
