Download the dataset into this folder. This should contain two subfolders `features`, `class-split` and `triplet_split`. The directory structure should be as mentioned below.

```bash
|-- features
|   |-- audio
|   |   |-- trn
|   |   |   |-- dog.h5
|   |   |   |-- cat.h5
|   |   |   |-- ......
|   |-- text
|   |   |-- word_embeddings-dict-33.npy
|   |-- video
|   |   |-- trn
|   |   |   |-- dog.h5
|   |   |   |-- cat.h5
|   |   |   |-- ......
|   |-- val.h5
|-- class-split
|   |-- all_class.txt
|   |-- seen_class.txt
|   |-- unseen_class.txt
|-- triplet_split
|   |-- triplets_zeroshot_False.csv
```