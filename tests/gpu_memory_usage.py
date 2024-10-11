import torch
import subprocess

def get_used_memory_by_gpu():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    memory_used = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return memory_used


def print_gpu_memory_usage(device_id=0):
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    # Get the device
    device = torch.device(f'cuda:{device_id}')
    
    # Get the memory allocated and reserved by PyTorch
    allocated_memory = torch.cuda.memory_allocated(device_id)
    reserved_memory = torch.cuda.memory_reserved(device_id)

    # Total memory on the GPU
    total_memory = torch.cuda.get_device_properties(device_id).total_memory

    # Calculate the available memory as total - reserved
    available_memory = total_memory - reserved_memory
    
    print(f"GPU Device: {device}")
    print(f"Total memory: {total_memory / 1024**2:.2f} MB")
    print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
    print(f"Reserved memory: {reserved_memory / 1024**2:.2f} MB")
    print(f"Available memory: {available_memory / 1024**2:.2f} MB")

def gpu_memory_usage():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    used_memory_by_gpu = get_used_memory_by_gpu()
    for i, memory in enumerate(used_memory_by_gpu):
        print(f"GPU-{i} used memory: {memory} MB")

    # Dynamically select a GPU that is available
    gpu_id = 0  # Default to 'cuda:0'
    device_count = torch.cuda.device_count()

    for i in range(device_count):
        if torch.cuda.memory_allocated(i) == 0:
            gpu_id = i
            break

    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using device: {device}")

    print_gpu_memory_usage(device_id=gpu_id)

    # Initial memory usage (should be zero or very low)
    print(f"Initial GPU memory used: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    
    # Allocate some tensors on the GPU
    x = torch.randn(1024*10, device=device)
    y = torch.randn(1024*10, device=device)
    print(f"GPU memory used after tensor allocation: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

    # Reset peak memory stats but it should still show the peak memory usage as nothing is deleted
    torch.cuda.reset_peak_memory_stats(device)
    print(f"Peak GPU memory used after reset (without deletion): {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

    # Delete one tensor and reset peak memory stats
    del y
    torch.cuda.reset_peak_memory_stats(device)
    print(f"Peak GPU memory used after deleting y tensor: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

    # Delete the remaining tensor
    del x
    torch.cuda.reset_peak_memory_stats(device)
    print(f"Peak GPU memory used after deleting x tensor: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

    # Final check, should be 0 MB or very low if everything is properly deleted
    print(f"Final GPU memory used after deleting all tensors: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")


if __name__ == "__main__":
    gpu_memory_usage()
