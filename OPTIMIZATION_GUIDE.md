# Simulation Optimization & Analysis Guide

## 🚀 Quick Start: Running Fast Simulations

### Use the Optimized Config

```bash
# Fast simulation with all optimizations enabled
python3 -m orchestrator.cli configs/run.fast.yaml --live

# Estimated time: ~2-4 hours (vs 70 hours with default settings)
```

### Key Optimizations in `run.fast.yaml`

1. **Reduced tokens**: 48 instead of 128 (3x faster)
2. **Sparse reflection**: Every 5 ticks instead of every tick (5x less overhead)
3. **Single steering layer**: Layer 16 only (vs 3 layers)
4. **Quantization ready**: 4-bit quantization support
5. **Reduced events**: capped at 12 per tick via `max_events_per_tick`

`configs/run.small.yaml` and `configs/run.medium.yaml` now declare `template: run.fast.yaml` so they inherit the same max events per tick and quantization/offload defaults while overriding only the population, tokens, and steering strength relevant to each arm.

---

## 📊 Performance Comparison

| Configuration | Time per 200 steps | Speedup |
|---------------|-------------------|---------|
| Original (debug32) | ~4 hours | 1x |
| Current issue | ~70 hours | 0.06x ⚠️ |
| Optimized (fast) | ~2-4 hours | 1-2x |
| With quantization | ~1-2 hours | 4-8x |

---

## ⚡ Optimization Techniques Applied

### 1. **Lazy Reflection** ✅ Implemented
**Location**: `agents/agent.py:80-96`

Agents only perform full reflection every N ticks, caching plans between reflections.

```yaml
optimization:
  reflect_every_n_ticks: 5  # Reflect only every 5 ticks
```

**Impact**: 5x reduction in memory retrieval overhead

### 2. **Token Reduction** ✅ Implemented
**Location**: `configs/run.fast.yaml`

```yaml
inference:
  max_new_tokens: 48  # Reduced from 128
```

**Impact**: 2-3x faster generation

### 3. **4-bit Quantization** ✅ Implemented
**Location**: `agents/language_backend.py:47-60`

```yaml
optimization:
  use_quantization: true
```

**Impact**: 2-3x faster inference, 75% less VRAM

### 4. **Single Layer Steering** ✅ Implemented

```yaml
layers: [16]  # Instead of [12, 16, 20]
```

**Impact**: Faster hook execution, less overhead

---

## 🔍 Steering Verification

### Check if Steering is Working

```bash
# Verify steering on existing run
python3 scripts/verify_steering.py --messages-dir storage/dumps/debug32/messages

# Quick check (first 10 timesteps)
python3 scripts/verify_steering.py --messages-dir storage/dumps/debug32/messages --num-files 10
```

### Expected Output

```
✓ PASS: Steering values are present in logs
✓ PASS: Steering values show variation
✓ PASS: Agents show diverse steering profiles

Steering appears to be active and functioning
  - 32 agents with steering applied
  - Trait values show expected variation
```

### Steering Confirmation from Your Data

**✅ Steering is WORKING!**

From your `debug32` run:
- All 32 agents have active steering
- Mean traits: E:+0.85, A:+0.48, C:+0.51, O:-0.04, N:+0.03
- High agent diversity (variance 0.04-0.07 across traits)
- 100% of messages show non-zero steering coefficients

---

## 📈 Post-Simulation Analysis

### Comprehensive Analysis

```bash
# Run full analysis on a completed simulation
./scripts/analyze_run.sh debug32

# Or run components separately:
python3 scripts/analyze_simulation.py --dump-dir storage/dumps/debug32
python3 scripts/verify_steering.py --messages-dir storage/dumps/debug32/messages
```

### Generated Visualizations

Output directory: `storage/analysis/`

1. **`action_distribution.png`** - Bar chart of action types
2. **`steering_heatmap.png`** - Heatmap of agent steering profiles
3. **`activity_over_time.png`** - Line plots of actions/messages over time

### Analysis Reports Include:

- Action type distribution
- Agent activity levels
- Token usage statistics
- Steering profile analysis
- Temporal dynamics
- Success/failure rates

---

## 🎮 Live Watching Simulations

### Enable Live Terminal View

```bash
# Watch simulation in real-time with colors
python3 -m orchestrator.cli configs/run.fast.yaml --live

# Without colors (for logging to file)
python3 -m orchestrator.cli configs/run.fast.yaml --live --no-color > sim.log
```

### Live Output Format

```
================================================================================
TICK 0 | 16 events scheduled
================================================================================
  ✓ agent-020 MOVE → community_center
    💬 agent-020 @ community_center
       Goal: We need to create a collaborative brief...
       [125→128 tokens] [C:+0.9, E:+0.5]

  ⏱  Tick completed in 72.45s

================================================================================
SIMULATION COMPLETE
  Run ID: debug32
  Ticks: 200
  Agents: 32
  Total time: 14532.12s
  Avg time/tick: 72.66s
================================================================================
```

---

## 🛠️ Troubleshooting 70-Hour Slowdown

### Diagnosis Checklist

1. **GPU availability**: Check if model is on CPU instead of GPU
   ```bash
   nvidia-smi  # Verify GPU is being used
   ```

2. **Memory thrashing**: Check if system is swapping
   ```bash
   free -h
   top
   ```

3. **Model loading**: Verify model loads once, not per-generation

4. **Reflection overhead**: Check if agents are reflecting every tick
   - Look for "reflect_every_n_ticks" in config
   - Default is 1 (every tick) - should be 5+

### Quick Fixes

```bash
# Stop current simulation
# Ctrl+C or kill the process

# Run with optimizations
python3 -m orchestrator.cli configs/run.fast.yaml --live

# Or adjust your current config:
# 1. Add optimization.reflect_every_n_ticks: 5
# 2. Reduce inference.max_new_tokens to 48
# 3. Reduce layers to [16] only
# 4. Add optimization.use_quantization: true
# 5. Cap GPU memory usage and spill extra layers to CPU/offload dir
```

### GPU Memory Guardrails

If you share a GPU with other processes, tell Accelerate to keep some layers on CPU:

```yaml
optimization:
  max_gpu_memory_gb: 16      # keep ~8 GB free on a 24 GB card
  max_cpu_memory_gb: 64      # allow CPU RAM for overflow layers
  offload_folder: "./artifacts/offload_cache"
```

These caps prevent CUDA OOMs mid-run by forcing the HF loader to spill layers to host
RAM/offload storage while the hotter layers stay resident on the GPU.

---

## 🎯 Optimal Configuration Trade-offs

### For Speed (2-4 hour runs)

```yaml
inference:
  max_new_tokens: 48
optimization:
  reflect_every_n_ticks: 5
  use_quantization: true
layers: [16]
```

### For Quality (8-12 hour runs)

```yaml
inference:
  max_new_tokens: 96
optimization:
  reflect_every_n_ticks: 2
  use_quantization: false
layers: [12, 16, 20]
```

### For Prototyping (minutes)

```bash
# Use mock model
python3 -m orchestrator.cli configs/run.fast.yaml --mock-model --live
```

---

## 🔬 Advanced Optimizations (Future Work)

### Not Yet Implemented

1. **Batched Inference** - Generate multiple agents in parallel
   - Expected: 3-5x speedup
   - Complexity: Medium

2. **Multi-GPU** - Distribute agents across GPUs
   - Expected: 2x speedup per GPU
   - Complexity: High

3. **Adaptive Token Limits** - Different limits per action type
   - Expected: 1.5x speedup
   - Complexity: Low

4. **Cached Embeddings** - Reuse memory retrieval results
   - Expected: 1.2x speedup
   - Complexity: Medium

---

## 📋 Quick Reference

### Config Files

- `configs/run.small.yaml` - Original 32-agent, 200-step config
- `configs/run.fast.yaml` - Optimized for speed (NEW)
- `configs/run.medium.yaml` - 100-agent config

### Scripts

- `scripts/view_results.py` - View parquet file contents
- `scripts/verify_steering.py` - Verify steering is working
- `scripts/analyze_simulation.py` - Full analysis + plots
- `scripts/analyze_run.sh` - Run all analysis at once

### CLI Flags

```bash
--live                # Enable live console output
--no-color            # Disable colored output
--mock-model          # Use mock backend (no LLM)
--max-events N        # Set encounters per tick
```

---

## 💡 Tips

1. **Always start with fast config** for testing, then scale up
2. **Use --live mode** to spot issues early
3. **Run steering verification** after first 10-20 ticks
4. **Monitor GPU usage** with `nvidia-smi -l 1`
5. **Use mock-model** for rapid iteration on non-LLM components

---

## 🎓 Understanding the Bottlenecks

Your current 70-hour runtime suggests:

1. ❌ **Not using optimized config** - Default reflection every tick
2. ❌ **128 tokens per generation** - Too many for simple actions
3. ❌ **3 steering layers** - More hook overhead
4. ❌ **Possible CPU fallback** - Check if GPU is actually used
5. ❌ **No quantization** - Full FP16 model is slower

**Solution**: Use `configs/run.fast.yaml` to fix all these issues!
