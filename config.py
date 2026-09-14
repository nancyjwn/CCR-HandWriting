config = {
    'exp_name': '【0001】FULL DATASET TRAIN',
    'epoch': 100,
    'lr': 1.0,
    'mode': 'stroke',
    'batch': 64,
    'val_frequency': 2000,
    'test_only': False,
    'resume': '',                      # KOSONG - karena training dari awal
    'train_dataset': './data/mydata/train_1000',
    'test_dataset': './data/mydata/test_1000',
    'weight_decay': False,
    'schedule_frequency': 5000,
    'image_size': 32,
    'alphabet': 3755,
}