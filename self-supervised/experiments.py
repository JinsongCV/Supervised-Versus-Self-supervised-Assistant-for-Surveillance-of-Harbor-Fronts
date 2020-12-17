from train import train
from test import test
from evaluate import evaluate

if __name__ == "__main__":
    #groupings = ['normal','abnormal']
    groupings = ['normal','entering','abnormal']

    cfg = {
           'experiment': 'normal',
           'train_folder': 'data/train1715/normal/',
           'test_folder': 'data/test500/',
           'val_folder': 'data/val143/',
           'image_size': (64,192),
           'max_epoch': 200,
           'gpus': 1,
           'lr': 0.0005,
           'batch_size': 32,
           'nc': 1,
           'nz': 8,
           'nfe': 32,
           'nfd': 32,
           'device': 'cuda',
          }

    #train(cfg)
    test(cfg,dataset='test',groupings=groupings,save=False)
    test(cfg,dataset='val', groupings=groupings)
    cfg['train_folder'] = 'data/train1715/'
    test(cfg,dataset='train', groupings=groupings)
    th = evaluate(cfg, dataset='train', groupings=groupings) # find a suitable threshold using training set
    evaluate(cfg, dataset='test', threshold=th, groupings=groupings)


    cfg = {
           'experiment': 'all',
           'train_folder': 'data/train1715/',
           'test_folder': 'data/test500/',
           'val_folder': 'data/val143/',
           'image_size': (64,192),
           'max_epoch': 200,
           'gpus': 1,
           'lr': 0.0005,
           'batch_size': 32,
           'nc': 1,
           'nz': 8,
           'nfe': 32,
           'nfd': 32,
           'device': 'cuda',
          }

    #train(cfg)
    test(cfg,dataset='test', groupings=groupings)
    test(cfg,dataset='val', groupings=groupings)
    test(cfg,dataset='train', groupings=groupings)
    th = evaluate(cfg, dataset='train', groupings=groupings)
    evaluate(cfg, dataset='test', threshold=th, groupings=groupings)


    '''
    #
    cfg = {
           'experiment': 'extensive',
           'train_folder': 'data/train1715/',
           'test_folder': 'data/test500/',
           'val_folder': 'data/val143/',
           'image_size': (64,192),
           'max_epoch': 50,
           'gpus': 1,
           'lr': 0.0005,
           'batch_size': 32,
           'nc': 1,
           'nz': 8,
           'nfe': 32,
           'nfd': 32,
           'device': 'cuda',
          }

    #train(cfg)
    test(cfg,dataset='test', groupings=groupings)
    test(cfg,dataset='val', groupings=groupings)
    #test(cfg,dataset='extensive', groupings=groupings)
    test(cfg,dataset='train', groupings=groupings)
    th = evaluate(cfg, dataset='train', groupings=groupings)
    evaluate(cfg, dataset='test', threshold=th, groupings=groupings)
    '''
