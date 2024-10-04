import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

import torch
from torch.optim import Adam
from transformers import AutoTokenizer

from module.module import Model
from module.trainer import Trainer
from module.dataloader import CNSDataLoader
from module.utils import set_deterministic_training, print_config


def define_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_path", default="./save/sample.pt", type=str)
    parser.add_argument('--load_path', default=None)
    parser.add_argument("--train_data_path", default="./data/final_matched2.train.csv", type=str)
    parser.add_argument("--test_data_path", default="./data/final_matched2.test.csv", type=str)

    # hyperparameters
    parser.add_argument('--huggingface_model_name', default='KETI-AIR-Downstream/long-ke-t5-base-summarization', type=str, help='KETI-AIR/long-ke-t5-small, KETI-AIR/ke-t5-large')
    parser.add_argument('--max_input_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=50, type=int)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--prefix', default='summarize: ', type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_stage', default=0, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument('--lr_decay_factor', default=2, type=int)
    parser.add_argument('--early_stopping_round', default=3, type=int)
    parser.add_argument('--early_stopping_criterion', default='bleu', type=str, help='bleu')
    parser.add_argument('--num_beams', default=1, type=int)

    # copy mechanism
    parser.add_argument('--use_copy', default=True, type=bool)
    parser.add_argument('--gen_loss_lambda', type=float, default=.7)
    parser.add_argument('--coverage', default=True, type=bool)
    parser.add_argument('--coverage_version', type=int, default=3)

    # distributed training  
    parser.add_argument("--prefetch_n_workers", default=4, type=int)

    # etc
    parser.add_argument('--print_example', action='store_true', help='print decoded result')
    parser.add_argument('--print_test_result', action='store_true', help='only for benchmark dataset')
    parser.add_argument('--save_result_to_xlsx', action='store_true', help='save result to xlsx format')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--train_n_data', default=1e8, type=int)
    parser.add_argument('--test_n_data', default=1e8, type=int)
    parser.add_argument('--fp16', action='store_true', help='using automatic mixed precision')
    parser.add_argument('--tokenizing_strategy', default='else', help='space/mecab(depacreted)/else(recommended)')
    parser.add_argument('--do_pruning', action='store_true')
    parser.add_argument('--prune_ratio', default=.3, type=float)

     # do not touch
    parser.add_argument('--seed', default=42, type=int) #42
    parser.add_argument('--local_rank', default=0, type=int)

    config = parser.parse_args()

    return config

def get_optimizer(model, config):
    no_decay = ['bias', 'LayerNorm.weight']
    params = []
    modules = [model.generator, model.attention, model.p_gen] if config.use_copy else [model.generator]

    for mod in modules:
        params += [
            {
                'params': [p for n, p in mod.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in mod.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
    optimizer = Adam(params, lr=config.lr)
     
    return optimizer

def load(model, config, optimizer, save_path, local_rank, return_prev_config=False):
    checkpoint = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(local_rank))
    model.generator.load_state_dict(checkpoint["generator"])

    if config.use_copy:
        model.attention.load_state_dict(checkpoint["attention"])
        model.p_gen.load_state_dict(checkpoint["p_gen"])
    # if 'pos' in config.train_data_path or 'pos' in config.valid_data_path:
    #     model.p_emb.load_state_dict(checkpoint['p_emb'])
    optimizer.load_state_dict(checkpoint["optimizer"])

    ret = (checkpoint["epoch"], checkpoint["score"], checkpoint["config"]) if return_prev_config else (checkpoint["epoch"], checkpoint["score"])
    return ret

if __name__ == '__main__':
    config = define_argparser()

    set_deterministic_training(config)
    device = torch.device(('cuda:{}'.format(config.local_rank)) if torch.cuda.is_available() else torch.device('cpu'))
    print('[Device] {}'.format(device))
    print_config(config)

    model = Model(config)
    model.init()

    tokenizer = AutoTokenizer.from_pretrained(config.huggingface_model_name)
    dataloader = CNSDataLoader(config, tokenizer)
    train_loader, test_loader = dataloader.get_dataloader()
    optim = get_optimizer(model, config)
    scheduler = None
    trainer = Trainer(model, tokenizer, optim, scheduler, train_loader, test_loader, config, device)

    global_epoch = 0
    try:
        global_epoch, score = load(model, config, optim, config.load_path, config.local_rank)
        print('load_from {} (epoch {})'.format(config.load_path, global_epoch))
        trainer.best_loss = score
    except:
        pass

    trainer.train(global_epoch)