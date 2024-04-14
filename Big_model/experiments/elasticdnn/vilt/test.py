from data import get_dataset


d = get_dataset('VQAv2_split1', '/data/zql/datasets/vqav2', 'train')
print(d[0])
