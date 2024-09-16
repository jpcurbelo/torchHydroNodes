## https://discuss.pytorch.org/t/measuring-peak-memory-usage-tracemalloc-for-pytorch/34067/23

import torch
def consume_gpu_ram(n): return torch.ones((n, n)).cuda()
def consume_gpu_ram_256mb(): return consume_gpu_ram(2**13)

def b2mb(x): return int(x/2**20)
class TorchTracemalloc():

    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used   = b2mb(self.end-self.begin)
        self.peaked = b2mb(self.peak-self.begin)
        print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

# push the process' peak gauge high up and then release all the memory
# expecting 0 used / 1024 peaked
with TorchTracemalloc() as tt:
    z = [consume_gpu_ram_256mb() for i in range(4)] # 1GB
    del z
assert tt.used == 0 and tt.peaked == 1024

# allocate, allocate, release half
# expecting 256 used / 512 peaked
with TorchTracemalloc() as tt:
    # should be: 256 used, 512 peaked
    c1 = consume_gpu_ram_256mb()
    c2 = consume_gpu_ram_256mb()
    del c1
assert tt.used == 256 and tt.peaked == 512
del c2 # reset for next test

# allocate, allocate, release all
# expecting 0 used / 512 peaked
with TorchTracemalloc() as tt:
    # should be: 0 used, 512 peaked
    c1 = consume_gpu_ram_256mb()
    c2 = consume_gpu_ram_256mb()
    del c1, c2
assert tt.used == 0 and tt.peaked == 512

# allocate, don't release
# expecting 1536 used / 1536 peaked
with TorchTracemalloc() as tt:
    z = [consume_gpu_ram_256mb() for i in range(6)]
assert tt.used == 1536 and tt.peaked == 1536
del z # reset for next test