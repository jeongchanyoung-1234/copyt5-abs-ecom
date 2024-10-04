import torch
from transformers import AutoTokenizer

from train import define_argparser, get_optimizer, load
from module.trainer import Trainer
from module.utils import print_config, compare_config
from module.dataloader import CNSDataLoader
from module.module import Model


    
if __name__ == '__main__':
    config = define_argparser()

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

    _, _, prev_config = load(model, config, optim, config.load_path, config.local_rank, return_prev_config=True)
    config = compare_config(config, prev_config)

    global_epoch, score = load(model, config, optim, config.load_path, config.local_rank)
    trainer = Trainer(model, tokenizer, optim, scheduler, train_loader, test_loader, config, device)

    trainer.validate(global_epoch, is_test=True)