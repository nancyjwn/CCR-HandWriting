import os
import sys
import time
import torch
import datetime
import itertools
import torch.nn as nn
from config import config
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from model.transformer import Transformer
from util import get_dataloader, get_data_package, converter, tensor2str, \
    saver, get_alphabet, rectify, is_correct, must_in_screen, confusing_feature_stroke, \
    character_to_strokelist, confusing_character_340, get_support_sample_feature_stroke, \
    to_gray_image_zero_one


# =========================================================
# TENSORBOARD
# =========================================================

writer = SummaryWriter(
    'history/{}'.format(config['exp_name'])
)


# =========================================================
# CONFIGURATION
# =========================================================

mode = config['mode']  # character / stroke

saver()
must_in_screen()

alphabet = get_alphabet(mode)

print('Alphabet : {}'.format(alphabet))


# =========================================================
# BATAS WAKTU TRAINING
# =========================================================

MAX_TRAINING_SECONDS = 10 * 60 * 60
TEST_BUFFER_SECONDS = 2 * 60 * 60

training_start_time = time.time()


# =========================================================
# MODEL
# =========================================================

model = Transformer(mode).cuda()
model = nn.DataParallel(model)


# =========================================================
# LOAD MODEL + RESUME POSITION
# =========================================================

start_epoch = 0
start_iter = 0

BASE_SEED = 12345


if config['resume'].strip() != '':

    print('Loading model dari:')
    print(config['resume'])

    model.load_state_dict(
        torch.load(
            config['resume'],
            map_location='cuda'
        )
    )

    print('Model berhasil dimuat.')

    # -----------------------------------------------------
    # LOAD LAST EPOCH
    # -----------------------------------------------------

    epoch_file = './history/{}/last_epoch.txt'.format(
        config['exp_name']
    )

    if os.path.exists(epoch_file):

        with open(epoch_file, 'r') as f:
            start_epoch = int(
                f.read().strip()
            )

        print(
            'Melanjutkan dari epoch: {}'.format(
                start_epoch
            )
        )

    else:

        print(
            'last_epoch.txt tidak ditemukan.'
        )


    # -----------------------------------------------------
    # LOAD LAST ITERATION
    # -----------------------------------------------------

    iter_file = './history/{}/last_iter.txt'.format(
        config['exp_name']
    )

    if os.path.exists(iter_file):

        with open(iter_file, 'r') as f:
            saved_iter = int(
                f.read().strip()
            )

        if saved_iter >= 0:

            start_iter = saved_iter + 1

            print(
                'Melanjutkan dari iterasi: {}'.format(
                    start_iter
                )
            )

        else:

            start_iter = 0

            print(
                'last_iter.txt = -1, mulai dari awal epoch.'
            )

    else:

        print(
            'last_iter.txt tidak ditemukan.'
        )


# =========================================================
# OPTIMIZER
# =========================================================

if config['weight_decay']:

    optimizer = optim.Adadelta(
        model.parameters(),
        lr=config['lr'],
        rho=0.9,
        weight_decay=1e-4
    )

else:

    optimizer = optim.Adadelta(
        model.parameters(),
        lr=config['lr'],
        rho=0.9
    )


# =========================================================
# LOSS
# =========================================================

criterion = torch.nn.CrossEntropyLoss().cuda()

mse_loss = torch.nn.MSELoss()


# =========================================================
# DATASET
# =========================================================

train_loader, test_loader = get_data_package()


# =========================================================
# SUPPORT SAMPLE
# =========================================================

class SupportSample(Dataset):

    def __init__(self, pair):

        self.samples = pair

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return self.samples[index]


# =========================================================
# FAST RESUME DATALOADER
# =========================================================

def create_resume_loader(
    train_loader,
    start_iter,
    epoch
):
    """
    Membuat DataLoader yang langsung mengambil data
    mulai dari iterasi start_iter.

    Tidak menggunakan:
        for _ in range(start_iter):
            next(dataloader)

    sehingga resume dari iterasi tinggi jauh lebih cepat.
    """

    if start_iter <= 0:

        return train_loader


    dataset = train_loader.dataset

    batch_size = train_loader.batch_size


    if batch_size is None:

        raise ValueError(
            'train_loader.batch_size tidak boleh None.'
        )


    # -----------------------------------------------------
    # Seed harus sama dengan seed ketika epoch tersebut
    # pertama kali dijalankan.
    # -----------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        BASE_SEED + epoch
    )


    # -----------------------------------------------------
    # Buat urutan data yang sama
    # -----------------------------------------------------

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()


    # -----------------------------------------------------
    # Hitung data yang sudah selesai
    # -----------------------------------------------------

    start_index = start_iter * batch_size


    # Jangan sampai index melebihi dataset

    if start_index >= len(indices):

        print(
            'start_index >= jumlah dataset. '
            'Epoch dianggap sudah selesai.'
        )

        remaining_indices = []

    else:

        remaining_indices = indices[start_index:]


    # -----------------------------------------------------
    # Dataset sisa
    # -----------------------------------------------------

    resume_dataset = Subset(
        dataset,
        remaining_indices
    )


    # -----------------------------------------------------
    # DataLoader baru
    # -----------------------------------------------------

    resume_loader = DataLoader(
        resume_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        collate_fn=train_loader.collate_fn,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last
    )


    print(
        'Fast resume aktif.'
    )

    print(
        'Epoch              : {}'.format(epoch)
    )

    print(
        'Mulai iterasi      : {}'.format(start_iter)
    )

    print(
        'Data yang dilewati : {}'.format(start_index)
    )

    print(
        'Sisa batch         : {}'.format(
            len(resume_loader)
        )
    )


    return resume_loader


# =========================================================
# TRAINING VARIABLES
# =========================================================

times = 0

best_acc = -1

confusing_dict = None

gallery_combine = None


# =========================================================
# SAVE CHECKPOINT
# =========================================================

def save_checkpoint_now(
    epoch,
    iteration
):
    """
    Menyimpan checkpoint darurat ketika batas waktu
    training tercapai.
    """

    history_path = './history/{}'.format(
        config['exp_name']
    )


    # -----------------------------------------------------
    # Simpan model
    # -----------------------------------------------------

    torch.save(
        model.state_dict(),
        '{}/model.pth'.format(history_path)
    )


    # -----------------------------------------------------
    # Simpan epoch
    # -----------------------------------------------------

    with open(
        '{}/last_epoch.txt'.format(history_path),
        'w'
    ) as f:

        f.write(
            str(epoch)
        )


    # -----------------------------------------------------
    # Simpan iterasi
    # -----------------------------------------------------

    with open(
        '{}/last_iter.txt'.format(history_path),
        'w'
    ) as f:

        f.write(
            str(iteration)
        )


    print(
        'Checkpoint darurat tersimpan '
        'di epoch {}, iterasi {}.'.format(
            epoch,
            iteration
        )
    )


# =========================================================
# TRAIN
# =========================================================

def train(
    epoch,
    iteration,
    image,
    length,
    text_input,
    text_gt,
    character_level_label
):

    global times
    global confusing_dict
    global gallery_combine


    model.train()


    optimizer.zero_grad()


    result = model(
        image,
        length,
        text_input
    )


    text_pred = result['pred']


    loss = criterion(
        text_pred,
        text_gt
    )


    loss.backward()


    optimizer.step()


    print(
        'epoch : {} | iter : {}/{} | loss : {}'.format(
            epoch,
            iteration,
            len(train_loader),
            loss
        )
    )


    writer.add_scalar(
        'loss',
        loss,
        times
    )


    times += 1


# =========================================================
# TEST / VALIDATION
# =========================================================

@torch.no_grad()
def test(epoch):

    torch.cuda.empty_cache()


    history_path = './history/{}'.format(
        config['exp_name']
    )


    # -----------------------------------------------------
    # Save current model
    # -----------------------------------------------------

    torch.save(
        model.state_dict(),
        '{}/model.pth'.format(history_path)
    )


    # -----------------------------------------------------
    # Accuracy record
    # -----------------------------------------------------

    result_file = open(
        '{}/accuracy_record.txt'.format(history_path),
        'w+',
        encoding='utf-8'
    )


    print('Start Eval!')


    model.eval()


    dataloader = iter(test_loader)

    test_loader_len = len(test_loader)


    correct = 0

    total = 0


    if config['mode'] == 'stroke':

        max_length = 30

    elif config['mode'] == 'character':

        max_length = 2

    else:

        raise ValueError(
            'Mode tidak dikenal: {}'.format(
                config['mode']
            )
        )


    clean_cache = False


    # =====================================================
    # TEST LOOP
    # =====================================================

    for iteration in range(test_loader_len):

        data = next(dataloader)


        image, label = data


        image = torch.nn.functional.interpolate(
            image,
            size=(
                config['image_size'],
                config['image_size']
            )
        )


        length, text_input, text_gt, character_level_label = \
            converter(
                mode,
                label
            )


        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------

        text_gt_list = []

        start = 0


        for i in length:

            text_gt_list.append(
                text_gt[
                    start:start + i
                ]
            )

            start += i


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        batch = image.shape[0]


        pred = torch.zeros(
            batch,
            1
        ).long().cuda()


        image_features = None


        prob = torch.zeros(
            batch,
            max_length
        ).float()


        # -------------------------------------------------
        # Character / stroke prediction
        # -------------------------------------------------

        for i in range(max_length):

            length = torch.zeros(
                batch
            ).long().cuda() + i + 1


            result = model(
                image,
                length,
                pred,
                conv_feature=image_features,
                test=True
            )


            prediction = result['pred']


            now_pred = torch.max(
                torch.softmax(
                    prediction,
                    2
                ),
                2
            )[1]


            prob[:, i] = torch.max(
                torch.softmax(
                    prediction,
                    2
                ),
                2
            )[0][:, -1]


            pred = torch.cat(
                (
                    pred,
                    now_pred[:, -1].view(
                        -1,
                        1
                    )
                ),
                1
            )


            image_features = result['conv']


        # -------------------------------------------------
        # Prediction list
        # -------------------------------------------------

        text_pred_list = []

        text_prob_list = []


        for i in range(batch):

            now_pred = []


            for j in range(max_length):

                if pred[i][j] != len(alphabet) - 1:

                    now_pred.append(
                        pred[i][j]
                    )

                else:

                    now_pred.append(
                        pred[i][j]
                    )

                    break


            text_pred_list.append(
                torch.Tensor(
                    now_pred
                )[1:].long().cuda()
            )


            overall_prob = 1.0


            for j in range(
                len(now_pred) - 1
            ):

                overall_prob *= prob[i][j]


            text_prob_list.append(
                overall_prob
            )


        # -------------------------------------------------
        # Compare prediction vs ground truth
        # -------------------------------------------------

        start = 0


        for i in range(batch):

            state = False


            pred_origin = tensor2str(
                mode,
                text_pred_list[i]
            ).replace(
                '$',
                ''
            )


            pred = rectify(
                mode,
                pred_origin
            )


            gt = tensor2str(
                mode,
                text_gt_list[i]
            ).replace(
                '$',
                ''
            )


            word_label = label[i].replace(
                '$',
                ''
            )


            if iteration == 0 and i == 0:

                clean_cache = True

            else:

                clean_cache = False


            whether_is_correct = is_correct(
                epoch,
                model,
                mode,
                image_features[i],
                pred,
                gt,
                word_label,
                clean_cache
            )


            if whether_is_correct['correct']:

                correct += 1

                state = True


            start += i

            total += 1


            print(
                '{} | {} | {} | {} | {} | {} | {}'.format(
                    total,
                    pred,
                    gt,
                    state,
                    text_prob_list[i],
                    correct / total,
                    pred_origin
                )
            )


            result_file.write(
                '{} | {} | {} | {} | {} | {}\n'.format(
                    total,
                    pred,
                    gt,
                    state,
                    text_prob_list[i],
                    pred_origin
                )
            )


    # =====================================================
    # ACCURACY
    # =====================================================

    acc = correct / total


    print(
        'ACC : {}'.format(acc)
    )


    global best_acc


    if acc > best_acc:

        best_acc = acc


        torch.save(
            model.state_dict(),
            '{}/best_model.pth'.format(history_path)
        )


    # -----------------------------------------------------
    # Record
    # -----------------------------------------------------

    f = open(
        '{}/record.txt'.format(history_path),
        'a+',
        encoding='utf-8'
    )


    f.write(
        'Epoch : {} | ACC : {}\n'.format(
            epoch,
            acc
        )
    )


    f.close()


    result_file.close()


    # -----------------------------------------------------
    # Test only
    # -----------------------------------------------------

    if config['test_only']:

        print(
            'Finish testing'
        )

        exit(0)


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':


    # -----------------------------------------------------
    # TEST ONLY
    # -----------------------------------------------------

    if config['test_only']:

        test(-1)


    stop_training = False


    # =====================================================
    # EPOCH LOOP
    # =====================================================

    for epoch in range(
        start_epoch,
        config['epoch']
    ):


        if stop_training:

            break


        # -------------------------------------------------
        # Save model sebelum epoch dimulai
        # -------------------------------------------------

        torch.save(
            model.state_dict(),
            './history/{}/model.pth'.format(
                config['exp_name']
            )
        )


        # -------------------------------------------------
        # Save current epoch
        # -------------------------------------------------

        with open(
            './history/{}/last_epoch.txt'.format(
                config['exp_name']
            ),
            'w'
        ) as f:

            f.write(
                str(epoch)
            )


        # -------------------------------------------------
        # Seed
        # -------------------------------------------------

        torch.manual_seed(
            BASE_SEED + epoch
        )


        # =================================================
        # FAST RESUME
        # =================================================

        resume_iter = (
            start_iter
            if epoch == start_epoch
            else 0
        )


        if resume_iter > 0:

            current_loader = create_resume_loader(
                train_loader,
                resume_iter,
                epoch
            )


            iteration_start = resume_iter


        else:

            current_loader = train_loader


            iteration_start = 0


        dataloader = iter(
            current_loader
        )


        train_loader_len = len(
            train_loader
        )


        print(
            'Training epoch {} dimulai dari iterasi {}.'.format(
                epoch,
                iteration_start
            )
        )


        # =================================================
        # ITERATION LOOP
        # =================================================

        for iteration in range(
            iteration_start,
            train_loader_len
        ):


            # -------------------------------------------------
            # Check batas waktu
            # -------------------------------------------------

            elapsed = (
                time.time()
                - training_start_time
            )


            if elapsed >= MAX_TRAINING_SECONDS:

                print(
                    'Batas 10 jam tercapai. '
                    'Menyimpan checkpoint dan '
                    'menghentikan training dengan aman...'
                )


                prev_iter = (
                    iteration - 1
                    if iteration > 0
                    else -1
                )


                save_checkpoint_now(
                    epoch,
                    prev_iter
                )


                stop_training = True

                break


            # -------------------------------------------------
            # Load batch
            # -------------------------------------------------

            data = next(
                dataloader
            )


            image, label = data


            image = torch.nn.functional.interpolate(
                image,
                size=(
                    config['image_size'],
                    config['image_size']
                )
            )


            length, text_input, text_gt, character_level_label = \
                converter(
                    mode,
                    label
                )


            # -------------------------------------------------
            # TRAIN
            # -------------------------------------------------

            train(
                epoch,
                iteration,
                image,
                length,
                text_input,
                text_gt,
                character_level_label
            )


            # -------------------------------------------------
            # SAVE ITERATION
            # -------------------------------------------------

            with open(
                './history/{}/last_iter.txt'.format(
                    config['exp_name']
                ),
                'w'
            ) as f:

                f.write(
                    str(iteration)
                )


            # =================================================
            # VALIDATION
            # =================================================

            if (
                iteration + 1
            ) % config['val_frequency'] == 0:


                elapsed_now = (
                    time.time()
                    - training_start_time
                )


                remaining = (
                    MAX_TRAINING_SECONDS
                    - elapsed_now
                )


                if remaining < TEST_BUFFER_SECONDS:

                    print(
                        'Sisa waktu tidak cukup untuk '
                        'evaluasi penuh ({:.0f} menit tersisa). '
                        'Melewati evaluasi, menyimpan checkpoint, '
                        'dan berhenti...'.format(
                            remaining / 60
                        )
                    )


                    save_checkpoint_now(
                        epoch,
                        iteration
                    )


                    stop_training = True

                    break


                else:

                    torch.cuda.empty_cache()

                    test(
                        epoch
                    )


        # =================================================
        # JIKA TRAINING BERHENTI
        # =================================================

        if stop_training:

            break


        # =================================================
        # EPOCH SELESAI
        # =================================================

        with open(
            './history/{}/last_iter.txt'.format(
                config['exp_name']
            ),
            'w'
        ) as f:

            f.write(
                '-1'
            )


        # -------------------------------------------------
        # Reset start_iter supaya epoch berikutnya
        # mulai dari 0
        # -------------------------------------------------

        start_iter = 0


        # =================================================
        # LEARNING RATE SCHEDULE
        # =================================================

        if (
            (epoch + 1)
            % config['schedule_frequency']
            == 0
        ):

            for p in optimizer.param_groups:

                p['lr'] *= 0.1


    # =====================================================
    # FINISH
    # =====================================================

    if stop_training:

        print(
            'Training dihentikan otomatis.'
        )

        print(
            'Jalankan ulang dengan resume '
            'untuk melanjutkan.'
        )

    else:

        print(
            'Training selesai penuh sampai epoch {}.'.format(
                config['epoch']
            )
        )