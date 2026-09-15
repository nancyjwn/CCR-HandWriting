config = {
    'exp_name': '【0001】FULL DATASET TRAIN',
    'epoch': 30,                    # cukup untuk konvergensi awal di data sebesar ini, realistis dari segi waktu
    'lr': 1.0,                      # default asli Fudan, dirancang untuk skala data besar
    'mode': 'stroke',
    'batch': 64,                    # dinaikkan dari default 32, mempercepat training di data besar
    'val_frequency': 3000,          # ~6250 iterasi/epoch, validasi ~2x per epoch
    'test_only': False,
    'resume': '',                   # kosong untuk training dari awal
    'train_dataset': './data/Trainmdb/Trainmdb',
    'test_dataset': './data/Testmdb/Testmdb',
    'weight_decay': False,
    'schedule_frequency': 5000,     # aktifkan LR decay bertahap, penting untuk training jangka panjang
    'image_size': 32,
    'alphabet': 3755,
}