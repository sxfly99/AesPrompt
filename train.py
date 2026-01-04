import os
import datetime
import torch.optim
from torch.utils.data import DataLoader
import warnings
import nni
from nni.utils import merge_parameter
warnings.filterwarnings("ignore")

from model import ITC_model, AesPrompt
from iaa_datasets.dataset_ava import AVA_zs_cap, AVA_level_cap
from iaa_datasets.dataset_aadb import *
from iaa_datasets.dataset_eva import *
from iaa_datasets.dataset_tad66k import *
from iaa_datasets.dataset_sac import *
from iaa_datasets.dataset_apdd import *
from utils import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, mean_squared_error

def init():
    parser = argparse.ArgumentParser(description="AesPrompt")

    parser.add_argument('--path_to_images_ava', type=str, default='/data/Database/AVA/images',
                        help='directory to AVA images')
    parser.add_argument('--path_to_images_aadb', type=str, default='/data/Database/AADB/datasetImages_originalSize',
                        help='directory to AADB images')
    parser.add_argument('--path_to_images_tad', type=str, default='/data/Database/TAD66K',
                        help='directory to TAD66K images')
    # art image
    parser.add_argument('--path_to_images_apdd', type=str, default='/data/Database/APDD/imgs',
                        help='directory to APDD images')
    # aigc image
    parser.add_argument('--path_to_images_sac', type=str,
                        default='/data/Database/sac/home/jdp/simulacra-aesthetic-captions',
                        help='directory to SAC images')

    parser.add_argument('--path_to_save_csv_pretrained', type=str, default="data/level/llama3_mistral_qwen",
                        help='directory to pretrained csv folder')
    parser.add_argument('--path_to_save_csv_ava', type=str, default="data/label/AVA",
                        help='directory to AVA csv folder')
    parser.add_argument('--path_to_save_csv_aadb', type=str, default="data/label/AADB",
                        help='directory to AADB csv folder')
    parser.add_argument('--path_to_save_csv_tad', type=str, default="data/label/TAD66K",
                        help='directory to TAD66K csv folder')
    parser.add_argument('--path_to_save_csv_apdd', type=str, default="data/label/APDD",
                        help='directory to APDD csv folder')

    # SAC specific
    parser.add_argument('--path_to_db_sac', type=str, default="data/label/SAC/sac_public_2022_06_29.sqlite",
                        help='path to sac db file')
    parser.add_argument('--path_to_split_sac', type=str, default='data/label/SAC/sac_split.txt',
                        help='path to sac split file')

    parser.add_argument('--path_to_ava_train_cap', type=str,
                        default='data/critiques/data/ava_train.json',
                        help='path to ava train captions json')
    parser.add_argument('--path_to_ava_test_cap', type=str,
                        default='data/critiques/data/ava_test.json',
                        help='path to ava test captions json')
    parser.add_argument('--path_to_aadb_cap', type=str,
                        default='data/critiques/data/aadb_test.json',
                        help='path to aadb captions json')
    parser.add_argument('--path_to_eva_cap', type=str,
                        default='data/critiques/data/eva_test.json',
                        help='path to eva captions json')
    parser.add_argument('--path_to_tad_cap', type=str,
                        default='data/critiques/data/tad_test.json',
                        help='path to tad captions json')
    parser.add_argument('--path_to_apdd_cap', type=str,
                        default='data/critiques/data/apdd_test.json',
                        help='path to apdd captions json')
    parser.add_argument('--path_to_sac_cap', type=str,
                        default='data/critiques/sac_test.json',
                        help='path to sac captions json')

    parser.add_argument('--experiment_dir_name', type=str, default='result/',
                        help='directory to save experiment results')

    parser.add_argument('--path_to_model', type=str, default='pretrained_ckpt/itc_model.pth',
                        help='path to pretrained model checkpoint')

    # Hyperparameters
    parser.add_argument('--init_lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--num_comments', type=int, default=4, help='num of aesthetics comments')

    parser.add_argument('--temperature', default=0.07, type=float, nargs='+',
                        help='temperature for contrastive learning')
    parser.add_argument('--n_ctx', default=6, type=int,
                        help='length of context')

    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.8685, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--step_size', default=1, type=float,
                        help='Step size for SGD')
    parser.add_argument("--num_epoch", default=6, type=int,
                        help="epochs of training")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size of training")
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')

    args = parser.parse_args()
    return args


def adjust_learning_rate(params, optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = params['init_lr'] * (0.5 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_score(opt, y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.cuda()

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part(opt):
    # TODO: pretrained data
    train_csv_path = os.path.join(opt['path_to_save_csv_pretrained'], 'distribution.csv')
    train_ds = AVA_level_cap(train_csv_path, opt['path_to_images_ava'], if_train=True, json_file=opt['path_to_ava_train_cap'])
    train_loader = DataLoader(train_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)

    # ava
    test_csv_path_ava = os.path.join(opt['path_to_save_csv_ava'], 'test.csv')
    ava_zs_ds = AVA_zs_cap(test_csv_path_ava, opt['path_to_images_ava'], if_train=False, json_file=opt['path_to_ava_test_cap'])
    ava_zs_loader = DataLoader(ava_zs_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    # aadb
    test_csv_path_aadb = os.path.join(opt['path_to_save_csv_aadb'], 'test.csv')
    aadb_zs_ds = AADB_cap(test_csv_path_aadb, opt['path_to_images_aadb'], if_train=False, json_file=opt['path_to_aadb_cap'])
    aadb_zs_loader = DataLoader(aadb_zs_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True, collate_fn=my_collate, drop_last=False)

    # tad66k
    test_csv_path_tad = os.path.join(opt['path_to_save_csv_tad'], 'test.csv')
    tad_zs_ds = TAD66K_cap(test_csv_path_tad, opt['path_to_images_tad'], if_train=False, json_file=opt['path_to_tad_cap'])
    tad_zs_loader = DataLoader(tad_zs_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    # apdd
    test_csv_path_apdd = os.path.join(opt['path_to_save_csv_apdd'], 'test.csv')
    apdd_zs_ds = APDD_cap(test_csv_path_apdd, opt['path_to_images_apdd'], if_train=False,json_file=opt['path_to_apdd_cap'])
    apdd_zs_loader = DataLoader(apdd_zs_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    # sac
    sac_zs_ds = SAC_cap(opt['path_to_images_sac'], opt['path_to_db_sac'], if_train=False, split='test',
                                    split_file=opt['path_to_split_sac'], json_file=opt['path_to_sac_cap'])
    sac_zs_loader = DataLoader(sac_zs_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, ava_zs_loader, aadb_zs_loader, tad_zs_loader, apdd_zs_loader, sac_zs_loader


def train(opt, epoch, model, loader, optimizer, criterion):
    model.train()
    model.cuda()
    train_losses_i = AverageMeter()
    # Freeze all parameters in the CLIP model
    for param in model.clip_model.parameters():
        param.requires_grad = False
    for param in model.prompt_learner.bert_model.parameters():
        param.requires_grad = False
    for param in model.prompt_learner.sem_model.parameters():
        param.requires_grad = False
    for param in model.prompt_learner.emo_model.parameters():
        param.requires_grad = False
    param_num = 0
    for param in model.parameters():
        if param.requires_grad == True:
            param_num += int(np.prod(param.shape))
    print('Trainable params: %.4f million' % (param_num / 1e6))

    for idx, (x, y, caps) in enumerate(tqdm(loader)):
        x = x.cuda()
        y = y.cuda()

        _, all_similarities = model(x, caps)
        loss = criterion(all_similarities, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_losses_i.avg

def validate_zs(opt, model, loader, database):
    model.eval()
    model.cuda()

    true_score_list = []
    pred_score_list = []

    for idx, data in enumerate(tqdm(loader)):
        if data is None:
            continue
        try:
            if database == 'ava':
                x, y, caps = data
                x = x.cuda()
                y = y.type(torch.FloatTensor).cuda()
                with torch.no_grad():
                    y_pred, _ = model(x, caps)
                tscore, tscore_np = get_score(opt, y)
                pscore_np = [item.item() * 10 for item in y_pred]
                true_score_list += tscore_np.tolist()
                pred_score_list += pscore_np
            else:
                if database in ['aadb', 'apdd']:
                    x = data['image']
                    y = data['label']
                    caps = data['caps']
                else:
                    x, y, caps = data
                    # print('ok')
                x = x.cuda()
                y = y.cuda()
                with torch.no_grad():
                    y_pred, _ = model(x, caps)
                if isinstance(y_pred, list):
                    y_pred = torch.tensor(y_pred)
                pred_score = torch.squeeze(y_pred)
                labels = torch.squeeze(y)
                predicts_np = pred_score.data.cpu().numpy() * 10
                ratings_np = labels.data.cpu().numpy() * 10
                pred_score_list += predicts_np.tolist()
                true_score_list += ratings_np.tolist()
        except FileNotFoundError as e:
            print(f"Error opening image: {e}")
            continue
        except Exception as e:
            print(f"Error processing data: {e}")
            continue

    pred_score_list = np.array(pred_score_list)
    true_score_list = np.array(true_score_list)
    lcc_mean = pearsonr(pred_score_list, true_score_list)
    srcc_mean = spearmanr(pred_score_list, true_score_list)

    true_score = np.array(true_score_list)
    pred_score = np.array(pred_score_list)

    if database == 'para':
        true_score_label = np.where(true_score <= 3.00, 0, 1)
        pred_score_label = np.where(pred_score <= 3.00, 0, 1)
    else:
        true_score_label = np.where(true_score <= 5.00, 0, 1)
        pred_score_label = np.where(pred_score <= 5.00, 0, 1)

    acc = accuracy_score(true_score_label, pred_score_label)
    mse = mean_squared_error(pred_score, true_score)

    return acc, lcc_mean[0], srcc_mean[0], mse

def zero_shot(opt):
    train_loader, ava_zs_loader, aadb_zs_loader, tad_zs_loader, apdd_zs_loader, sac_zs_loader = create_data_part(opt)
    pretrained_model = ITC_model(clip_name='ViT-B/16')
    pretrained_model.load_state_dict(torch.load(opt['path_to_model']))
    model = AesPrompt(pretrained_model)
    model = model.cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=opt['init_lr'],
                                  weight_decay=0.01)
    criterion = EDMLoss()
    datasets = {
        'AVA': ava_zs_loader,
        'AADB': aadb_zs_loader,
        'TAD66K': tad_zs_loader,
        'APDD': apdd_zs_loader,
        'SAC': sac_zs_loader
    }
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    log_name = 'final_model' + '.txt'
    log_file = os.path.join(opt['experiment_dir_name'], log_name)

    dataset_count = len(datasets)
    with open(log_file, 'a') as f:
        f.write(f"Zero Shot Results\n")
        f.write(f"Run Time: {current_time}\n")
        f.write("=" * 60 + "\n")

    best_results = {
        'epoch': -1,
        'avg_acc': 0,
        'avg_lcc': 0,
        'avg_srcc': 0,
        'avg_mse': float('inf'),
        'score': 0
    }

    for e in range(opt['num_epoch']):
        total_acc = 0
        total_lcc = 0
        total_srcc = 0
        total_mse = 0
        optimizer = adjust_learning_rate(opt, optimizer, e)
        print("*******************************************************************************************************")
        print("第%d个epoch的学习率：%f" % (1 + e, optimizer.param_groups[0]['lr']))

        train_loss = train(opt, epoch=e, model=model, loader=train_loader, optimizer=optimizer,
                           criterion=criterion)
        with open(log_file, 'a') as f:
            f.write(f"Epoch: {e}\n")
            f.write("=" * 60 + "\n")

            for name, loader in datasets.items():
                print(f"\n{'=' * 20} Results for {name} dataset {'=' * 20}\n")
                f.write(f"\n{'=' * 20} Results for {name} dataset {'=' * 20}\n")

                acc, lcc, srcc, mse = validate_zs(opt, model=model, loader=loader, database=name.lower())

                if acc is not None and lcc is not None and srcc is not None and mse is not None:
                    result_str = (f"PLCC: {lcc:.4f}\n"
                                  f"SRCC: {srcc:.4f}\n"
                                  f"ACC: {acc:.4f}\n"
                                  f"MSE: {mse:.4f}\n")
                    # 累加性能指标
                    total_acc += acc
                    total_lcc += lcc
                    total_srcc += srcc
                    total_mse += mse
                else:
                    result_str = "No valid predictions or true scores to evaluate.\n"

                print(result_str)
                f.write(result_str)

                print("\n" + "=" * 60 + "\n")
                f.write("\n" + "=" * 60 + "\n")

            avg_acc = total_acc / dataset_count
            avg_lcc = total_lcc / dataset_count
            avg_srcc = total_srcc / dataset_count
            avg_mse = total_mse / dataset_count

            avg_result_str = (f"\n{'=' * 20} Average Results {'=' * 20}\n"
                              f"Average PLCC: {avg_lcc:.4f}\n"
                              f"Average SRCC: {avg_srcc:.4f}\n"
                              f"Average ACC: {avg_acc:.4f}\n"
                              f"Average MSE: {avg_mse:.4f}\n")

            print(avg_result_str)
            f.write(avg_result_str)

            print("\n" + "=" * 60 + "\n")
            f.write("\n" + "=" * 60 + "\n")

            score = (avg_lcc + avg_srcc) / 2
            if score > best_results['score']:
                best_results = {
                    'epoch': e,
                    'avg_acc': avg_acc,
                    'avg_lcc': avg_lcc,
                    'avg_srcc': avg_srcc,
                    'avg_mse': avg_mse,
                    'score': score
                }
                best_model_path = os.path.join(opt['experiment_dir_name'], 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)

    with open(log_file, 'a') as f:
        best_result_str = (f"\n{'=' * 20} Best Results {'=' * 20}\n"
                           f"Best Epoch: {best_results['epoch']}\n"
                           f"Best Average PLCC: {best_results['avg_lcc']:.4f}\n"
                           f"Best Average SRCC: {best_results['avg_srcc']:.4f}\n"
                           f"Best Average ACC: {best_results['avg_acc']:.4f}\n"
                           f"Best Average MSE: {best_results['avg_mse']:.4f}\n"
                           f"Best Score (PLCC+SRCC)/2: {best_results['score']:.4f}\n")

        print(best_result_str)
        f.write(best_result_str)

if __name__ == "__main__":
    opt = init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings.filterwarnings('ignore')
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(opt, tuner_params))
    print(params)
    zero_shot(params)

