"""Quick speed comparison: baseline vs torch.compile + TF32.
Run in Docker: docker run --rm --gpus all wave-field-llm-s1-hybrid python scripts/speed_test.py
"""
import torch, time, os

os.environ['HYBRID_LAYERS'] = '3,7'
os.environ['SPECTRAL_GATE'] = '1'

torch.manual_seed(42)
B, N = 16, 512
device = 'cuda'

from src.wave_field_transformer import WaveFieldTransformer

def make_model():
    return WaveFieldTransformer(
        vocab_size=8000, embedding_dim=384, num_layers=8, num_heads=8,
        ffn_dim=1536, field_size=512, max_seq_len=514, dropout=0.1,
        hybrid_attention_layers=[3, 7], device=device
    ).to(device)

def bench(model, label, n_steps=5):
    optimizer = model.configure_optimizer(base_lr=2e-4)
    scaler = torch.amp.GradScaler(device)
    ids = torch.randint(0, 8000, (B, N), device=device)
    labels = torch.randint(0, 8000, (B, N), device=device)
    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            _, loss = model(ids, labels=labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    # Timed
    t0 = time.time()
    for _ in range(n_steps):
        optimizer.zero_grad()
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            _, loss = model(ids, labels=labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tps = n_steps * B * N / elapsed
    print(f"  {label}: {tps:,.0f} tok/s  ({elapsed:.2f}s / {n_steps} steps)")
    return tps

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Batch: {B}x{N} = {B*N:,} tokens/step\n")

# 1. Baseline (no compile, no TF32)
print("--- Baseline ---")
m1 = make_model()
t1 = bench(m1, "no compile, no TF32")
del m1; torch.cuda.empty_cache()

# 2. TF32 only
print("\n--- TF32 only ---")
torch.set_float32_matmul_precision('high')
m2 = make_model()
t2 = bench(m2, "TF32 only")
del m2; torch.cuda.empty_cache()

# 3. Compile + TF32
print("\n--- Compile + TF32 ---")
m3 = make_model()
m3.compile_model(mode='default')
t3 = bench(m3, "compile + TF32")
del m3; torch.cuda.empty_cache()

print(f"\n{'='*50}")
print(f"  TF32 speedup:          {t2/t1:.2f}x")
print(f"  Compile+TF32 speedup:  {t3/t1:.2f}x")
print(f"{'='*50}")
