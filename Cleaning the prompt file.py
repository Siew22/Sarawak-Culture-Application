import torch
import gc

torch.cuda.empty_cache()  # 清理未使用的显存
gc.collect()  # 强制进行垃圾回收
