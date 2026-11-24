
import pickle
import sys



# Class harus tersedia dengan nama __main__.SupportSample
class SupportSample:

    def __init__(self, pair):
        self.samples = pair

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


files = [
    "data/print_font_template/simsun.pkl",
    "data/print_font_template/simfang.pkl"
]

for path in files:

    print("\n" + "=" * 60)
    print("FILE:", path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    print("Tipe data:", type(data))
    print("Jumlah elemen:", len(data))

    confusing_dict, support_dataset = data

    print("\nConfusing dict:")
    print("Tipe:", type(confusing_dict))
    print("Jumlah key:", len(confusing_dict))

    for i, (key, value) in enumerate(confusing_dict.items()):
        print("Contoh", i + 1)
        print("Stroke:", key)
        print("Indeks karakter:", value)

        if i >= 4:
            break

    print("\nSupport dataset:")
    print("Tipe:", type(support_dataset))
    print("Jumlah data:", len(support_dataset))

    if len(support_dataset) > 0:
        sample = support_dataset[0]

        print("Tipe sampel:", type(sample))

        if isinstance(sample, (tuple, list)):
            print("Jumlah elemen sampel:", len(sample))

            for j, item in enumerate(sample):
                if hasattr(item, "shape"):
                    print("Elemen", j, "shape:", item.shape)
                else:
                    print("Elemen", j, "tipe:", type(item))
