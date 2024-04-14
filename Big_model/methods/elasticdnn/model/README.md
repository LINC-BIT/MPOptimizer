## 20230527

only prune to_v (so should divide to_qkv to to_qkv and to_v, and only prune to_v)

attn (1, 12, 197, 197) * pruned_v (1, 12, 197, int(64\*s)) = out1 (1, 12, 197, int(64\*s))
out1 rearange to out2 (1, 197, int(768\*s))

out2 (1, 197, int(768\*s)) * pruned mlp_fc1 (int(768\*s), int(3072\*s))