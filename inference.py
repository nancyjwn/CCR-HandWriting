import sys
import json
import argparse
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# 1. LOKASI PROJECT DAN EKSPERIMEN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

EXPERIMENT_DIR = ROOT_DIR / "history" / "【0005】FULLDATASET TRAIN"

CHECKPOINT_PATH = EXPERIMENT_DIR / "best_model.pth"

SIMSUN_PATH = ROOT_DIR / "data" / "print_font_template" / "simsun.pkl"
SIMFANG_PATH = ROOT_DIR / "data" / "print_font_template" / "simfang.pkl"

sys.path.insert(0, str(EXPERIMENT_DIR))


# ============================================================
# 2. KELAS UNTUK MEMBACA TEMPLATE PICKLE
# ============================================================

class SupportSample(Dataset):

    def __init__(self, pair):
        self.samples = pair

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


# ============================================================
# 3. IMPORT MODEL DAN UTILITAS EKSPERIMEN
# ============================================================

from model.transformer import Transformer
import util


# ============================================================
# 4. KELAS INFERENSI
# ============================================================

class HandwritingRecognizer:

    def __init__(self):

        self.device = torch.device("cpu")

        print("Device:", self.device)
        print("Memuat model AI...")

        self.model = Transformer("stroke").to(self.device)

        state_dict = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True
        )

        # Menghapus prefix module. dari checkpoint DataParallel
        if all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
            }

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.alphabet = util.get_alphabet("stroke")

        self.template_simsun = None
        self.template_simfang = None

        print("Model berhasil dimuat.")

    # --------------------------------------------------------
    # PREPROCESSING GAMBAR
    # --------------------------------------------------------

    def preprocess_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = image.resize(
            (32, 32),
            Image.Resampling.BILINEAR
        )

        image_tensor = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32
        )

        image_tensor = image_tensor.reshape(32, 32, 3)
        image_tensor = image_tensor.permute(2, 0, 1)

        image_tensor = image_tensor / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5

        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    # --------------------------------------------------------
    # MEMUAT TEMPLATE SIMSUN DAN SIMFANG
    # --------------------------------------------------------

    def load_templates(self):

        if self.template_simsun is None:

            print("Memuat template SimSun...")

            self.template_simsun = (
                util.get_support_sample_feature_stroke(
                    self.model,
                    str(SIMSUN_PATH)
                )
            )

            print("Template SimSun berhasil dimuat.")

        if self.template_simfang is None:

            print("Memuat template SimFang...")

            self.template_simfang = (
                util.get_support_sample_feature_stroke(
                    self.model,
                    str(SIMFANG_PATH)
                )
            )

            print("Template SimFang berhasil dimuat.")

    # --------------------------------------------------------
    # PREDIKSI URUTAN STROKE
    # --------------------------------------------------------

    @torch.no_grad()
    def predict_stroke(self, image):

        pred = torch.zeros(
            (1, 1),
            dtype=torch.long,
            device=self.device
        )

        image_features = None
        probe_feature = None

        max_length = 30

        for i in range(max_length):

            length = torch.full(
                (1,),
                i + 1,
                dtype=torch.long,
                device=self.device
            )

            result = self.model(
                image,
                length,
                pred,
                conv_feature=image_features,
                test=True
            )

            prediction = result["pred"]

            now_pred = torch.argmax(
                torch.softmax(prediction, dim=2),
                dim=2
            )

            next_token = now_pred[:, -1:]

            pred = torch.cat(
                (pred, next_token),
                dim=1
            )

            image_features = result["conv"]
            probe_feature = result["conv"]

            predicted_index = next_token.item()

            # Token terakhir alphabet adalah $
            if predicted_index == len(self.alphabet) - 1:
                break

        # Menghilangkan token awal < dan token akhir $
        token_indices = pred[0, 1:].tolist()

        stroke_sequence = ""

        for index in token_indices:

            token = self.alphabet[index]

            if token == "$":
                break

            stroke_sequence += token

        return stroke_sequence, probe_feature

    # --------------------------------------------------------
    # MENENTUKAN KARAKTER DARI STROKE SEQUENCE
    # --------------------------------------------------------

    @torch.no_grad()
    def identify_character(self, stroke_sequence, probe_feature):

        candidates = util.confusing_stroke_dic.get(
            stroke_sequence
        )

        if not candidates:
            raise ValueError(
                "Stroke sequence tidak ditemukan dalam kamus model: "
                + stroke_sequence
            )

        # Jika hanya ada satu karakter dengan urutan stroke tersebut
        if len(candidates) == 1:
            return candidates[0]

        # Jika terdapat beberapa karakter, gunakan template
        self.load_templates()

        confusing_dict1, gallery1 = self.template_simsun
        confusing_dict2, gallery2 = self.template_simfang

        if stroke_sequence not in confusing_dict1:
            raise ValueError(
                "Stroke sequence tidak ditemukan dalam template SimSun."
            )

        confusing_indices = confusing_dict1[stroke_sequence]

        if stroke_sequence not in confusing_dict2:
            raise ValueError(
                "Stroke sequence tidak ditemukan dalam template SimFang."
            )

        if confusing_indices != confusing_dict2[stroke_sequence]:
            raise ValueError(
                "Daftar kandidat SimSun dan SimFang tidak cocok."
            )

        indices = torch.tensor(
            confusing_indices,
            dtype=torch.long
        )

        gallery_feature1 = gallery1[indices]
        gallery_feature2 = gallery2[indices]

        probe = probe_feature.squeeze(0)

        scores = []

        for i in range(len(confusing_indices)):

            score1 = F.mse_loss(
                probe,
                gallery_feature1[i]
            )

            score2 = F.mse_loss(
                probe,
                gallery_feature2[i]
            )

            average_score = (score1 + score2) / 2

            scores.append(average_score)

        scores = torch.stack(scores)

        best_position = scores.argmin().item()

        best_index = confusing_indices[best_position]

        character = util.confusing_character_340[best_index]

        return character

    # --------------------------------------------------------
    # FUNGSI UTAMA PENGENALAN TULISAN
    # --------------------------------------------------------

    def recognize(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Gambar tidak ditemukan: {image_path}"
            )

        image = self.preprocess_image(image_path)

        raw_stroke, probe_feature = self.predict_stroke(image)

        if not raw_stroke:
            raise ValueError(
                "Model tidak menghasilkan urutan stroke."
            )

        corrected_stroke = util.rectify(
            "stroke",
            raw_stroke
        )

        character = self.identify_character(
            corrected_stroke,
            probe_feature
        )

        return {
            "character": character,
            "stroke_sequence": corrected_stroke,
            "raw_stroke_sequence": raw_stroke
        }


# ============================================================
# 5. MENJALANKAN INFERENSI DARI TERMINAL
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pengenalan tulisan tangan Mandarin"
    )

    parser.add_argument(
        "image",
        help="Lokasi gambar tulisan tangan Mandarin"
    )

    args = parser.parse_args()

    recognizer = HandwritingRecognizer()

    result = recognizer.recognize(args.image)

    print("\n========== HASIL PREDIKSI ==========")

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4
        )
    )