"""
Scenario data for Stack Doctor.

Each scenario encodes a hidden root cause, the correct fix, an incident ticket,
hardware/model/backend context, log and code snippets, and specialist opinions
(some of which may be wrong).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field


ROOT_CAUSES = [
    "arch_guard",
    "backend_whitelist",
    "runtime_loader",
    "backend_selector",
    "model_config",
    "weight_layout",
    "memory_oom",
    "quantization_error",
    "distributed_comm",
    "driver_compat",
]

FIXES = [
    "relax_arch_check",
    "add_whitelist_entry",
    "fix_runtime_path",
    "switch_backend",
    "update_model_config",
    "fix_weight_mapping",
    "tune_memory_config",
    "fix_quantization",
    "fix_comm_config",
    "update_driver_config",
]

# 1:1 mapping
ROOT_CAUSE_TO_FIX = dict(zip(ROOT_CAUSES, FIXES))
FIX_TO_ROOT_CAUSE = {v: k for k, v in ROOT_CAUSE_TO_FIX.items()}

SPECIALISTS = ["runtime", "dispatch", "kernel", "loader"]

HARDWARE_OPTIONS = [
    "NVIDIA SM121 (DGX Spark)",
    "NVIDIA SM120 (GeForce RTX 5090)",
    "AMD MI300X",
    "AMD MI355X",
    "NVIDIA H100",
    "NVIDIA B200",
]

MODEL_OPTIONS = [
    "DeepSeek-V3-671B",
    "Llama-4-Maverick-17Bx128E",
    "Qwen3-235B-A22B",
    "Mistral-Large-2",
    "DeepSeek-R1-Distill-70B",
    "Llama-3.3-70B-Instruct",
]

BACKEND_OPTIONS = [
    "vLLM 0.8.x",
    "SGLang 0.5.x",
    "TensorRT-LLM 0.18",
    "FlashInfer 0.4",
    "Triton Inference Server",
]


@dataclass
class SpecialistOpinion:
    opinion: str
    confidence: float
    is_correct: bool


@dataclass
class InspectResult:
    logs: str
    config: str
    snippet: str
    metrics: str


@dataclass
class Scenario:
    id: str
    root_cause: str
    correct_fix: str
    incident_ticket: str
    hardware: str
    model_name: str
    backend: str
    initial_log: str
    initial_snippet: str
    specialist_opinions: dict[str, SpecialistOpinion]
    inspect_results: InspectResult
    # For ask_specialist follow-ups
    specialist_followups: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Seed scenarios
# ---------------------------------------------------------------------------

def _make_scenarios() -> list[Scenario]:
    scenarios = []

    # --- arch_guard scenarios ---
    scenarios.append(Scenario(
        id="arch_guard_01",
        root_cause="arch_guard",
        correct_fix="relax_arch_check",
        incident_ticket=(
            "INCIDENT: FlashInfer attention kernel fails to launch on newly provisioned "
            "DGX Spark nodes. Error: 'Unsupported GPU architecture sm_121'. "
            "Identical model config works on H100 nodes."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="DeepSeek-V3-671B",
        backend="FlashInfer 0.4",
        initial_log=(
            "[FlashInfer] Checking GPU capability... sm_121 detected\n"
            "[FlashInfer] ERROR: is_supported_arch() returned False for sm_121\n"
            "[FlashInfer] Falling back to... no fallback available\n"
            "RuntimeError: No compatible attention kernel for architecture sm_121"
        ),
        initial_snippet=(
            "# flashinfer/arch_check.py\n"
            "SUPPORTED_ARCHS = {70, 75, 80, 86, 89, 90}\n"
            "\n"
            "def is_supported_arch(cc: int) -> bool:\n"
            "    return cc in SUPPORTED_ARCHS"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime loaded successfully. No runtime issues detected.", 0.85, False
            ),
            "dispatch": SpecialistOpinion(
                "Architecture check is blocking kernel dispatch. The SM121 architecture "
                "is not in the supported set despite being SM90-compatible at the instruction level.", 0.92, True
            ),
            "kernel": SpecialistOpinion(
                "The HMMA m16n8k16 instructions used by the attention kernel are available on SM121. "
                "This looks like a capability check issue, not a kernel issue.", 0.88, True
            ),
            "loader": SpecialistOpinion(
                "Model weights loaded correctly. Weight layout is standard.", 0.80, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[FlashInfer] GPU: NVIDIA GH200 (sm_121)\n"
                "[FlashInfer] CUDA version: 13.0\n"
                "[FlashInfer] is_supported_arch(121) = False\n"
                "[FlashInfer] Architecture check FAILED\n"
                "[CUDA] All CUDA operations nominal\n"
                "[System] GPU memory: 96GB available"
            ),
            config=(
                "gpu_architecture: sm_121\n"
                "cuda_version: 13.0\n"
                "flashinfer_version: 0.4.1\n"
                "attention_backend: flashinfer\n"
                "supported_archs: [70, 75, 80, 86, 89, 90]"
            ),
            snippet=(
                "# The arch check function uses an exact match:\n"
                "def is_supported_arch(cc):\n"
                "    return cc in SUPPORTED_ARCHS  # misses sm_12x family\n\n"
                "# SM121 supports HMMA m16n8k16 (same as SM90)\n"
                "# but is not in the allowlist"
            ),
            metrics=(
                "kernel_launch_attempts: 47\n"
                "kernel_launch_failures: 47\n"
                "fallback_attempts: 47\n"
                "fallback_failures: 47\n"
                "gpu_utilization: 0%"
            ),
        ),
        specialist_followups={
            "runtime": "I confirmed CUDA 13.0 runtime is functional. All driver calls succeed. This isn't a runtime issue.",
            "dispatch": "The dispatch table maps arch -> kernel. SM121 has no entry. Adding sm_12x family to the arch check should fix it.",
            "kernel": "I inspected the PTX. The kernel only needs HMMA m16n8k16 which SM121 supports. The kernel itself is fine.",
            "loader": "Weights are in the expected layout. No loader issues.",
        },
    ))

    scenarios.append(Scenario(
        id="arch_guard_02",
        root_cause="arch_guard",
        correct_fix="relax_arch_check",
        incident_ticket=(
            "INCIDENT: MLA attention fails on GeForce RTX 5090. Error: "
            "'compute capability 120 not supported'. Customer reports RTX 4090 works fine."
        ),
        hardware="NVIDIA SM120 (GeForce RTX 5090)",
        model_name="DeepSeek-R1-Distill-70B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Detecting GPU... GeForce RTX 5090 (sm_120)\n"
            "[vLLM] FlashAttention: compute capability 120 not in supported list\n"
            "[vLLM] ERROR: Cannot initialize attention backend"
        ),
        initial_snippet=(
            "# vllm/attention/backends/flash_attn.py\n"
            "MIN_CC = 80\n"
            "MAX_CC = 90\n"
            "\n"
            "def is_supported(cc: int) -> bool:\n"
            "    return MIN_CC <= cc <= MAX_CC"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime is fine. CUDA 13 loaded.", 0.75, False),
            "dispatch": SpecialistOpinion(
                "The capability range check excludes SM120. Needs to include SM12x family.", 0.90, True
            ),
            "kernel": SpecialistOpinion(
                "Possible kernel incompatibility — SM120 lacks tcgen05 MMA.", 0.60, False
            ),
            "loader": SpecialistOpinion("Weights look fine.", 0.70, False),
        },
        inspect_results=InspectResult(
            logs="[vLLM] GPU cc=120 rejected by range [80,90]\n[vLLM] No fallback attention backend",
            config="compute_capability: 120\nmax_supported_cc: 90\nattention_backend: flash_attn",
            snippet="# Range check: MIN_CC(80) <= cc <= MAX_CC(90)\n# SM120 = 120 > 90, so rejected\n# Fix: add sm_12x family check",
            metrics="attention_init_failures: 1\nmodel_load_time: 0s (blocked at init)",
        ),
        specialist_followups={
            "runtime": "CUDA 13.0 runtime is healthy. Driver version matches.",
            "dispatch": "SM120 uses HMMA path (no warp specialization), same code path as SM86. Just need to update the arch range.",
            "kernel": "On closer inspection, SM120 does support the needed HMMA instructions. My earlier concern about tcgen05 was wrong — that's only needed for Hopper-style warp specialization.",
            "loader": "No weight issues detected.",
        },
    ))

    # --- backend_whitelist scenarios ---
    scenarios.append(Scenario(
        id="backend_whitelist_01",
        root_cause="backend_whitelist",
        correct_fix="add_whitelist_entry",
        incident_ticket=(
            "INCIDENT: Marlin quantized inference crashes on SM121 nodes. "
            "Error: 'Marlin kernel not available for current GPU'. "
            "FP16 inference works, only quantized (GPTQ/AWQ) path fails."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="Llama-3.3-70B-Instruct",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Loading GPTQ-quantized model...\n"
            "[vLLM] Checking Marlin kernel availability for sm_121\n"
            "[vLLM] WARNING: GPU sm_121 not in Marlin whitelist\n"
            "[vLLM] ERROR: No quantization kernel available"
        ),
        initial_snippet=(
            "# vllm/model_executor/layers/quantization/marlin.py\n"
            "MARLIN_SUPPORTED_GPUS = [\n"
            "    'A100', 'A10', 'H100', 'L40', 'RTX 4090',\n"
            "]\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA runtime OK. Libraries loaded.", 0.80, False),
            "dispatch": SpecialistOpinion(
                "Marlin whitelist doesn't include SM121 GPU names. Need to add the entry.", 0.91, True
            ),
            "kernel": SpecialistOpinion(
                "Marlin kernels use standard HMMA ops that SM121 supports. It's just not whitelisted.", 0.85, True
            ),
            "loader": SpecialistOpinion(
                "Quantized weights loaded but kernel never launches. Might be a weight format issue.", 0.55, False
            ),
        },
        inspect_results=InspectResult(
            logs="[Marlin] GPU name 'NVIDIA GH200' not in whitelist\n[Marlin] Whitelist: ['A100','A10','H100','L40','RTX 4090']",
            config="quantization: gptq\nmarlin_whitelist: [A100, A10, H100, L40, RTX 4090]\ngpu_name: NVIDIA GH200",
            snippet="# Whitelist check uses GPU product name string matching\n# GH200 / DGX Spark not in the list\n# Should use arch family check instead of name matching",
            metrics="quantized_kernel_attempts: 1\nquantized_kernel_failures: 1\nfp16_fallback: not_attempted",
        ),
        specialist_followups={
            "runtime": "All good on the runtime side.",
            "dispatch": "The whitelist is name-based, not arch-based. Adding 'GH200' or switching to family-level arch checks fixes this.",
            "kernel": "The Marlin FP8 GEMM dispatch works with SM121's MMA units. It's purely a whitelist gap.",
            "loader": "Actually, the weights loaded fine. I retract my earlier concern.",
        },
    ))

    scenarios.append(Scenario(
        id="backend_whitelist_02",
        root_cause="backend_whitelist",
        correct_fix="add_whitelist_entry",
        incident_ticket=(
            "INCIDENT: AWQ quantization backend refuses to initialize on MI300X. "
            "Error: 'GPU not supported for AWQ acceleration'. "
            "Other backends work fine on the same hardware."
        ),
        hardware="AMD MI300X",
        model_name="Qwen3-235B-A22B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Initializing AWQ backend...\n"
            "[vLLM] GPU: AMD Instinct MI300X\n"
            "[vLLM] AWQ: GPU not in supported devices list\n"
            "[vLLM] ERROR: AWQ acceleration unavailable"
        ),
        initial_snippet=(
            "# vllm/model_executor/layers/quantization/awq.py\n"
            "AWQ_SUPPORTED = {'A100', 'H100', 'RTX 4090', 'L40S'}\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("ROCm runtime healthy. HIP version matches.", 0.82, False),
            "dispatch": SpecialistOpinion(
                "AWQ whitelist is NVIDIA-only. MI300X needs to be added.", 0.93, True
            ),
            "kernel": SpecialistOpinion(
                "MI300X has MFMA instructions that can handle the AWQ GEMM. Not a kernel issue.", 0.87, True
            ),
            "loader": SpecialistOpinion("Weight format might not match AMD layout expectations.", 0.50, False),
        },
        inspect_results=InspectResult(
            logs="[AWQ] Device 'AMD Instinct MI300X' not in AWQ_SUPPORTED\n[AWQ] Supported: A100, H100, RTX 4090, L40S",
            config="quantization: awq\nawq_supported: [A100, H100, RTX 4090, L40S]\ngpu: AMD Instinct MI300X",
            snippet="# AWQ_SUPPORTED only lists NVIDIA GPUs\n# MI300X MFMA f32_32x32x8_f16 can handle AWQ ops\n# Need to add MI300X to whitelist",
            metrics="awq_init_failures: 1\nfallback_to_fp16: pending",
        ),
        specialist_followups={
            "runtime": "ROCm 6.3 loaded successfully. No runtime concerns.",
            "dispatch": "Simple whitelist gap. Adding MI300X resolves the issue.",
            "kernel": "Confirmed: MFMA ops on MI300X handle the AWQ GEMM pattern.",
            "loader": "I was wrong earlier — weights are fine. It's the whitelist.",
        },
    ))

    # --- runtime_loader scenarios ---
    scenarios.append(Scenario(
        id="runtime_loader_01",
        root_cause="runtime_loader",
        correct_fix="fix_runtime_path",
        incident_ticket=(
            "INCIDENT: SGLang server crashes on startup with CUDA 13 on DGX Spark. "
            "Error: 'libcudart.so.13: cannot open shared object file'. "
            "System has CUDA 13 installed but SGLang can't find it."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] Starting server...\n"
            "[SGLang] Loading CUDA runtime...\n"
            "[SGLang] ERROR: libcudart.so.13: cannot open shared object file\n"
            "[SGLang] LD_LIBRARY_PATH=/usr/local/cuda-12/lib64\n"
            "ImportError: CUDA runtime not found"
        ),
        initial_snippet=(
            "# sglang/startup.py\n"
            "CUDA_LIB_PATH = os.environ.get(\n"
            "    'CUDA_HOME', '/usr/local/cuda'\n"
            ") + '/lib64'\n"
            "# Hardcoded to cuda, not cuda-13\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA 13 is installed at /usr/local/cuda-13 but LD_LIBRARY_PATH points to cuda-12. "
                "The runtime path needs to be updated.", 0.95, True
            ),
            "dispatch": SpecialistOpinion("Can't tell — server never gets to dispatch phase.", 0.40, False),
            "kernel": SpecialistOpinion("No kernel issue — server crashes before kernel init.", 0.60, False),
            "loader": SpecialistOpinion(
                "The CUDA shared library loader can't find libcudart.so.13. Path issue.", 0.88, True
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[System] CUDA installations:\n"
                "  /usr/local/cuda-12 -> CUDA 12.4\n"
                "  /usr/local/cuda-13 -> CUDA 13.0\n"
                "  /usr/local/cuda -> symlink to cuda-12\n"
                "[SGLang] Trying to load libcudart.so.13 from /usr/local/cuda/lib64 -> NOT FOUND"
            ),
            config="CUDA_HOME=/usr/local/cuda\nLD_LIBRARY_PATH=/usr/local/cuda-12/lib64\ncuda_13_path=/usr/local/cuda-13",
            snippet="# /usr/local/cuda symlinks to cuda-12\n# Need: export CUDA_HOME=/usr/local/cuda-13\n# Or: update symlink",
            metrics="server_start_attempts: 3\nserver_start_failures: 3\nuptime: 0s",
        ),
        specialist_followups={
            "runtime": "Confirmed: /usr/local/cuda symlink targets cuda-12. CUDA 13 is at /usr/local/cuda-13. Fix the path.",
            "dispatch": "Server never started, so I can't diagnose dispatch.",
            "kernel": "Same — no kernel loaded.",
            "loader": "The dynamic linker searches LD_LIBRARY_PATH first. It needs /usr/local/cuda-13/lib64.",
        },
    ))

    scenarios.append(Scenario(
        id="runtime_loader_02",
        root_cause="runtime_loader",
        correct_fix="fix_runtime_path",
        incident_ticket=(
            "INCIDENT: ROCm HIP runtime fails to initialize on MI300X cluster. "
            "Error: 'hipErrorNoDevice' despite GPUs being visible in lspci. "
            "Worked yesterday before system update."
        ),
        hardware="AMD MI300X",
        model_name="DeepSeek-V3-671B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[HIP] Initializing runtime...\n"
            "[HIP] ERROR: hipErrorNoDevice (code 100)\n"
            "[System] lspci shows 8x AMD Instinct MI300X\n"
            "[System] /opt/rocm -> /opt/rocm-6.2 (outdated symlink)"
        ),
        initial_snippet=(
            "# environment setup\n"
            "ROCM_PATH=/opt/rocm  # symlinks to rocm-6.2\n"
            "# But rocm-6.3 installed at /opt/rocm-6.3\n"
            "# Driver expects rocm-6.3 runtime\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm path mismatch. /opt/rocm points to 6.2 but driver needs 6.3 runtime.", 0.94, True
            ),
            "dispatch": SpecialistOpinion("Not a dispatch issue — runtime doesn't initialize.", 0.70, False),
            "kernel": SpecialistOpinion("Might be a kernel module issue with the GPU driver.", 0.45, False),
            "loader": SpecialistOpinion("ROCm shared libraries at wrong version.", 0.80, True),
        },
        inspect_results=InspectResult(
            logs="[System] /opt/rocm -> /opt/rocm-6.2\n[System] Driver version: 6.3.0\n[HIP] Runtime version mismatch: expected 6.3, found 6.2",
            config="ROCM_PATH=/opt/rocm\nrocm_symlink_target=/opt/rocm-6.2\ninstalled_versions: [6.2, 6.3]\ndriver_version: 6.3.0",
            snippet="# The system was updated and ROCm 6.3 driver installed\n# But /opt/rocm symlink still points to 6.2\n# Fix: ln -sf /opt/rocm-6.3 /opt/rocm",
            metrics="gpu_init_failures: 8\ndriver_version: 6.3.0\nruntime_version: 6.2.0",
        ),
        specialist_followups={
            "runtime": "Classic version mismatch after system update. Fix the symlink to point to rocm-6.3.",
            "dispatch": "Can't assess dispatch without a working runtime.",
            "kernel": "I was wrong — it's not a kernel module issue. The GPU driver is fine, it's the userspace runtime path.",
            "loader": "The shared library loader finds rocm-6.2 libs but driver expects 6.3. Path fix needed.",
        },
    ))

    # --- backend_selector scenarios ---
    scenarios.append(Scenario(
        id="backend_selector_01",
        root_cause="backend_selector",
        correct_fix="switch_backend",
        incident_ticket=(
            "INCIDENT: Extreme latency (10x expected) on H100 serving Llama-3.3-70B. "
            "No errors, just very slow. GPU utilization looks low. "
            "Other models on the same node are fast."
        ),
        hardware="NVIDIA H100",
        model_name="Llama-3.3-70B-Instruct",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Selected attention backend: xformers\n"
            "[vLLM] WARNING: FlashAttention v2 not selected (override with VLLM_ATTENTION_BACKEND)\n"
            "[vLLM] Serving Llama-3.3-70B-Instruct...\n"
            "[vLLM] p99 latency: 4200ms (expected: ~400ms)"
        ),
        initial_snippet=(
            "# vllm/attention/selector.py\n"
            "def get_attention_backend(model_config):\n"
            "    if model_config.head_dim not in [64, 128]:\n"
            "        return 'xformers'  # fallback\n"
            "    return 'flash_attn'\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA runtime is fine. No errors.", 0.75, False),
            "dispatch": SpecialistOpinion(
                "Wrong attention backend selected. xformers is much slower than FlashAttention on H100. "
                "The backend selector has a bug in head_dim detection.", 0.94, True
            ),
            "kernel": SpecialistOpinion(
                "The xformers kernel is correct but suboptimal for H100. Should use flash_attn.", 0.82, True
            ),
            "loader": SpecialistOpinion("Model loaded correctly. Not a weight issue.", 0.80, False),
        },
        inspect_results=InspectResult(
            logs="[vLLM] head_dim=128, num_heads=64\n[vLLM] Backend selection: model reports head_dim=None (config missing) -> fallback to xformers",
            config="attention_backend: xformers (auto-selected)\nmodel_head_dim: null\nactual_head_dim: 128\ngpu: H100",
            snippet="# The model config doesn't explicitly set head_dim\n# Selector falls back to xformers when head_dim is None\n# Should infer head_dim from hidden_size / num_heads",
            metrics="p50_latency_ms: 3100\np99_latency_ms: 4200\ngpu_utilization: 12%\nexpected_gpu_util: 85%",
        ),
        specialist_followups={
            "runtime": "No runtime issues. The server is running, just slow.",
            "dispatch": "Backend selector bug: head_dim is None in model config, causing xformers fallback. Switch to flash_attn.",
            "kernel": "xformers works but doesn't use H100 TMA/warp specialization. flash_attn v2 would be 8-10x faster.",
            "loader": "Weights loaded correctly.",
        },
    ))

    scenarios.append(Scenario(
        id="backend_selector_02",
        root_cause="backend_selector",
        correct_fix="switch_backend",
        incident_ticket=(
            "INCIDENT: FP8 inference on MI300X producing garbage output. "
            "Model loads, tokens generate, but output is nonsensical. "
            "BF16 inference on same hardware works perfectly."
        ),
        hardware="AMD MI300X",
        model_name="Mistral-Large-2",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] FP8 quantization: e4m3fn format selected\n"
            "[vLLM] WARNING: MI300X uses e4m3fnuz format, not e4m3fn\n"
            "[vLLM] Serving with FP8...\n"
            "[vLLM] Output quality check: FAIL (perplexity 847.3, expected <15)"
        ),
        initial_snippet=(
            "# vllm/quantization/fp8.py\n"
            "FP8_FORMAT = 'e4m3fn'  # NVIDIA default\n"
            "# AMD MI300X needs e4m3fnuz (no NaN, unsigned zero)\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("ROCm runtime is healthy.", 0.80, False),
            "dispatch": SpecialistOpinion(
                "Wrong FP8 format selected. MI300X uses e4m3fnuz, not e4m3fn. "
                "The backend selector should detect AMD and switch format.", 0.93, True
            ),
            "kernel": SpecialistOpinion(
                "The GEMM kernel runs but produces wrong results due to format mismatch.", 0.85, True
            ),
            "loader": SpecialistOpinion(
                "Weight dequantization might be wrong for AMD FP8 format.", 0.65, False
            ),
        },
        inspect_results=InspectResult(
            logs="[FP8] Using e4m3fn format\n[FP8] AMD GPU detected but format not switched\n[FP8] Numerical errors in first GEMM",
            config="fp8_format: e4m3fn\ngpu_vendor: AMD\nexpected_format: e4m3fnuz\nformat_mismatch: true",
            snippet="# e4m3fn: 1 sign, 4 exp, 3 mantissa, has NaN encoding\n# e4m3fnuz: 1 sign, 4 exp, 3 mantissa, NO NaN, unsigned zero\n# Bit patterns interpreted differently -> garbage output",
            metrics="output_perplexity: 847.3\nexpected_perplexity: 12.5\ngemm_numerical_errors: 100%",
        ),
        specialist_followups={
            "runtime": "ROCm fine. This is a numerical issue, not runtime.",
            "dispatch": "Switch the FP8 format selector to use e4m3fnuz for AMD GPUs. Clear fix.",
            "kernel": "The kernel math is correct for the format it's given — the problem is the format itself.",
            "loader": "Actually, weights are fine. The issue is at the GEMM dispatch level.",
        },
    ))

    # --- model_config scenarios ---
    scenarios.append(Scenario(
        id="model_config_01",
        root_cause="model_config",
        correct_fix="update_model_config",
        incident_ticket=(
            "INCIDENT: DeepSeek-V3 MoE routing crashes with shape mismatch. "
            "Error: 'Expected expert count 256, got 160'. "
            "Model just updated to new checkpoint, was working before."
        ),
        hardware="NVIDIA H100",
        model_name="DeepSeek-V3-671B",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] Loading DeepSeek-V3-671B...\n"
            "[SGLang] MoE config: num_experts=256 (from config.json)\n"
            "[SGLang] Actual weight shape: experts.0-159\n"
            "[SGLang] ERROR: Shape mismatch in MoE layer: expected 256 experts, found 160"
        ),
        initial_snippet=(
            "# config.json (model repo)\n"
            '{\n'
            '  "num_local_experts": 256,\n'
            '  "num_experts_per_tok": 8,\n'
            '  "intermediate_size": 2048\n'
            '}\n'
            "# But actual checkpoint has 160 experts\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime is fine. Model loading proceeds until shape error.", 0.75, False),
            "dispatch": SpecialistOpinion("Not a dispatch bug — the model config is wrong.", 0.70, False),
            "kernel": SpecialistOpinion(
                "MoE kernel expects expert count from config. Config says 256 but weights have 160. "
                "Config needs to be updated to match the new checkpoint.", 0.90, True
            ),
            "loader": SpecialistOpinion(
                "The model config doesn't match the checkpoint. num_local_experts should be 160.", 0.92, True
            ),
        },
        inspect_results=InspectResult(
            logs="[SGLang] config.json: num_local_experts=256\n[SGLang] checkpoint expert layers: 160\n[SGLang] Mismatch detected at layer 0",
            config="num_local_experts: 256 (config)\nactual_experts: 160 (checkpoint)\nnum_experts_per_tok: 8\ncheckpoint_version: v3.1",
            snippet="# New checkpoint v3.1 reduced experts from 256 to 160\n# But config.json wasn't updated\n# Fix: set num_local_experts=160 in config.json",
            metrics="model_load_progress: 15%\nlayers_loaded: 0/60\nerror_at: moe_layer_0",
        ),
        specialist_followups={
            "runtime": "No runtime issue. Pure config mismatch.",
            "dispatch": "Dispatch looks fine. The error is before dispatch even runs.",
            "kernel": "The grouped GEMM kernel allocates buffers based on config expert count. Fix the config.",
            "loader": "Config.json says 256 experts but the v3.1 checkpoint only has 160. Update the config.",
        },
    ))

    scenarios.append(Scenario(
        id="model_config_02",
        root_cause="model_config",
        correct_fix="update_model_config",
        incident_ticket=(
            "INCIDENT: Qwen3 MoE model gives wrong results after hardware migration. "
            "Output is coherent but factually wrong. "
            "Same model on old cluster was correct."
        ),
        hardware="NVIDIA B200",
        model_name="Qwen3-235B-A22B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Loading Qwen3-235B-A22B...\n"
            "[vLLM] Config: rope_theta=1000000.0\n"
            "[vLLM] WARNING: RoPE scaling config missing for extended context\n"
            "[vLLM] Serving... output quality degraded at positions > 4096"
        ),
        initial_snippet=(
            "# config.json\n"
            '{\n'
            '  "rope_theta": 1000000.0,\n'
            '  "max_position_embeddings": 32768\n'
            '  // Missing: rope_scaling config for YaRN\n'
            '}\n'
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime fine. No crashes.", 0.80, False),
            "dispatch": SpecialistOpinion("Backend selected correctly.", 0.65, False),
            "kernel": SpecialistOpinion(
                "RoPE computation looks standard. Config might be missing the scaling parameters.", 0.78, True
            ),
            "loader": SpecialistOpinion(
                "Model config is incomplete — missing rope_scaling section for YaRN. "
                "Old cluster had a patched config.", 0.91, True
            ),
        },
        inspect_results=InspectResult(
            logs="[vLLM] RoPE: theta=1e6, no scaling applied\n[vLLM] Quality degrades > 4096 tokens\n[vLLM] Old cluster config had rope_scaling: {type: yarn, factor: 4}",
            config="rope_theta: 1000000.0\nrope_scaling: null\nmax_position_embeddings: 32768\nold_config_had: {rope_scaling: {type: yarn, factor: 4}}",
            snippet="# Missing rope_scaling config:\n# rope_scaling: {type: 'yarn', factor: 4, ...}\n# Without it, positions > 4096 are garbage",
            metrics="quality_0_4k: 95%\nquality_4k_8k: 43%\nquality_8k_plus: 12%",
        ),
        specialist_followups={
            "runtime": "No runtime issues.",
            "dispatch": "Backend is correct. Not a dispatch issue.",
            "kernel": "The RoPE kernel is fine — it just doesn't have the scaling config to apply YaRN.",
            "loader": "The config.json from the model repo is missing rope_scaling. Add it back.",
        },
    ))

    # --- weight_layout scenarios ---
    scenarios.append(Scenario(
        id="weight_layout_01",
        root_cause="weight_layout",
        correct_fix="fix_weight_mapping",
        incident_ticket=(
            "INCIDENT: Model produces random output after converting weights from "
            "HuggingFace format to TensorRT-LLM format. Conversion reported success "
            "but inference output is gibberish."
        ),
        hardware="NVIDIA H100",
        model_name="Llama-3.3-70B-Instruct",
        backend="TensorRT-LLM 0.18",
        initial_log=(
            "[TRT-LLM] Loading converted weights...\n"
            "[TRT-LLM] Weight shapes match expected layout\n"
            "[TRT-LLM] Running inference...\n"
            "[TRT-LLM] Output: 'asdfjkl; the the the purple 2847...'\n"
            "[TRT-LLM] Perplexity: 2341.7 (expected < 10)"
        ),
        initial_snippet=(
            "# convert_weights.py\n"
            "# gate_proj and up_proj were swapped during conversion\n"
            "mapping = {\n"
            "    'gate_proj': 'linear_fc1_gate',\n"
            "    'up_proj': 'linear_fc1_up',\n"
            "}\n"
            "# TRT-LLM expects opposite order\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime and engine init successful. No errors.", 0.80, False),
            "dispatch": SpecialistOpinion("Backend dispatch is correct. TRT engine built fine.", 0.70, False),
            "kernel": SpecialistOpinion(
                "Kernels execute without error. This is a data issue, not compute.", 0.75, False
            ),
            "loader": SpecialistOpinion(
                "Weight mapping is wrong. gate_proj and up_proj are swapped in the conversion script. "
                "TRT-LLM expects the opposite order.", 0.94, True
            ),
        },
        inspect_results=InspectResult(
            logs="[TRT-LLM] Weight conversion: gate_proj -> linear_fc1_gate, up_proj -> linear_fc1_up\n[TRT-LLM] Expected: gate_proj -> linear_fc1_up, up_proj -> linear_fc1_gate",
            config="weight_mapping:\n  gate_proj: linear_fc1_gate  # WRONG\n  up_proj: linear_fc1_up      # WRONG\n  # Should be swapped",
            snippet="# TRT-LLM MLP layout: [up_proj; gate_proj] concatenated\n# But converter wrote [gate_proj; up_proj]\n# Result: SiLU applied to wrong half",
            metrics="output_perplexity: 2341.7\nexpected_perplexity: 8.2\nweight_shapes: correct\nweight_values: misaligned",
        ),
        specialist_followups={
            "runtime": "Engine runs fine. Not a runtime issue.",
            "dispatch": "TRT engine dispatch is correct.",
            "kernel": "Compute is correct for the data it gets. Fix the data (weights).",
            "loader": "Classic weight mapping bug. Swap gate_proj and up_proj in the conversion mapping.",
        },
    ))

    scenarios.append(Scenario(
        id="weight_layout_02",
        root_cause="weight_layout",
        correct_fix="fix_weight_mapping",
        incident_ticket=(
            "INCIDENT: QKV attention weights transposed incorrectly for GQA model. "
            "Attention scores are wrong — model generates repetitive text. "
            "Happened after switching from MHA to GQA config."
        ),
        hardware="AMD MI300X",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="FlashInfer 0.4",
        initial_log=(
            "[FlashInfer] GQA mode: 64 query heads, 8 KV heads\n"
            "[FlashInfer] WARNING: QKV projection weight shape unexpected\n"
            "[FlashInfer] Expected Q:[8192,8192] K:[8192,1024] V:[8192,1024]\n"
            "[FlashInfer] Got Q:[8192,8192] K:[8192,8192] V:[8192,1024]\n"
            "[FlashInfer] Repetitive output detected"
        ),
        initial_snippet=(
            "# weight_converter.py\n"
            "# GQA: Q has num_heads, K/V have num_kv_heads\n"
            "q_proj = weights['q_proj']  # [8192, 8192] correct\n"
            "k_proj = weights['q_proj']  # BUG: should be 'k_proj'\n"
            "v_proj = weights['v_proj']  # [8192, 1024] correct\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("ROCm runtime fine.", 0.75, False),
            "dispatch": SpecialistOpinion("FlashInfer dispatch selected GQA path correctly.", 0.70, False),
            "kernel": SpecialistOpinion(
                "GQA attention kernel is correct but K weights are wrong shape. "
                "Looks like Q weights loaded twice instead of K.", 0.88, True
            ),
            "loader": SpecialistOpinion(
                "Weight mapping bug: k_proj loaded from q_proj key. Copy-paste error in converter.", 0.95, True
            ),
        },
        inspect_results=InspectResult(
            logs="[FlashInfer] K weight shape [8192,8192] != expected [8192,1024]\n[FlashInfer] K weights appear identical to Q weights\n[FlashInfer] This causes attention to compute Q*Q^T instead of Q*K^T",
            config="num_query_heads: 64\nnum_kv_heads: 8\nhead_dim: 128\nq_shape: [8192,8192]\nk_shape: [8192,8192] # WRONG\nv_shape: [8192,1024]",
            snippet="# Bug in weight_converter.py line 47:\n# k_proj = weights['q_proj']  # should be weights['k_proj']\n# Result: K = Q, so attention = softmax(Q @ Q^T) -> repetitive",
            metrics="attention_entropy: 0.03 (expected > 2.0)\nrepetition_rate: 94%\nperplexity: 567.8",
        ),
        specialist_followups={
            "runtime": "No runtime problems.",
            "dispatch": "GQA dispatch path is correct for this model.",
            "kernel": "Attention kernel computes correctly for the data given. K weights are just wrong.",
            "loader": "Line 47 has `weights['q_proj']` instead of `weights['k_proj']`. Classic copy-paste bug.",
        },
    ))

    # --- arch_guard additional scenarios ---
    scenarios.append(Scenario(
        id="arch_guard_03",
        root_cause="arch_guard",
        correct_fix="relax_arch_check",
        incident_ticket=(
            "INCIDENT: TensorRT-LLM refuses to build engine for B200 GPU. "
            "Error: 'Unsupported compute capability 120'. "
            "Same model builds fine targeting H100."
        ),
        hardware="NVIDIA B200",
        model_name="Qwen3-235B-A22B",
        backend="TensorRT-LLM 0.18",
        initial_log=(
            "[TRT-LLM] Building engine for gpu_arch=sm_120...\n"
            "[TRT-LLM] ERROR: Compute capability 120 not in supported set\n"
            "[TRT-LLM] Supported: {70, 75, 80, 86, 89, 90}"
        ),
        initial_snippet=(
            "# tensorrt_llm/builder.py\n"
            "SUPPORTED_SM = {70, 75, 80, 86, 89, 90}\n"
            "if sm not in SUPPORTED_SM:\n"
            "    raise UnsupportedGPU(f'sm_{sm}')"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA 13 runtime loaded fine.", 0.78, False),
            "dispatch": SpecialistOpinion(
                "Architecture guard rejects sm_120. B200 uses Blackwell arch not in the allowlist.", 0.91, True
            ),
            "kernel": SpecialistOpinion(
                "Try switching to a different quantization scheme for B200.", 0.45, False
            ),
            "loader": SpecialistOpinion("No weight loading attempted yet — blocked at engine build.", 0.72, False),
        },
        inspect_results=InspectResult(
            logs="[TRT-LLM] sm_120 not in {70,75,80,86,89,90}\n[TRT-LLM] Engine build aborted before weight conversion",
            config="target_gpu: sm_120\nsupported_sm: [70,75,80,86,89,90]\nbuilder_version: 0.18.0",
            snippet="# B200 (sm_120) supports FP8 MMA, BF16 HMMA\n# Same instruction set as H100 for inference\n# Just not in the allowlist",
            metrics="engine_build_attempts: 1\nengine_build_failures: 1\nmodel_loaded: false",
        ),
        specialist_followups={
            "runtime": "Runtime is fine. Engine builder is the blocker.",
            "dispatch": "Add sm_120 (and sm_12x family) to SUPPORTED_SM. The instructions are compatible.",
            "kernel": "On reflection, quantization scheme isn't the issue. It's the arch check.",
            "loader": "Can't load weights until engine builds.",
        },
    ))

    scenarios.append(Scenario(
        id="arch_guard_04",
        root_cause="arch_guard",
        correct_fix="relax_arch_check",
        incident_ticket=(
            "INCIDENT: Flash-Attention fwd pass returns CUDA error on MI355X. "
            "Error: 'Unsupported AMD GPU architecture'. "
            "MI300X works fine with same code."
        ),
        hardware="AMD MI355X",
        model_name="Llama-3.3-70B-Instruct",
        backend="vLLM 0.8.x",
        initial_log=(
            "[Flash-Attn] Checking GPU: AMD Instinct MI355X (gfx950)\n"
            "[Flash-Attn] Supported AMD archs: [gfx90a, gfx942]\n"
            "[Flash-Attn] ERROR: gfx950 not supported"
        ),
        initial_snippet=(
            "# flash_attn/amd_check.py\n"
            "AMD_SUPPORTED = ['gfx90a', 'gfx942']\n"
            "if gpu_arch not in AMD_SUPPORTED:\n"
            "    raise RuntimeError(f'{gpu_arch} not supported')"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("ROCm 6.4 runtime operational.", 0.80, False),
            "dispatch": SpecialistOpinion(
                "gfx950 (MI355X/CDNA4) isn't in the AMD arch allowlist. Needs to be added.", 0.92, True
            ),
            "kernel": SpecialistOpinion(
                "MI355X has different MFMA tile sizes — kernel might actually be incompatible.", 0.55, False
            ),
            "loader": SpecialistOpinion("Can't assess — kernel never launched.", 0.60, False),
        },
        inspect_results=InspectResult(
            logs="[Flash-Attn] gfx950 not in [gfx90a, gfx942]\n[Flash-Attn] MI355X CDNA4 arch check failed",
            config="gpu_arch: gfx950\namd_supported: [gfx90a, gfx942]\nrocm_version: 6.4",
            snippet="# MI355X (gfx950/CDNA4) extends gfx942 instruction set\n# MFMA f32_32x32x16_fp8 available\n# Just missing from allowlist",
            metrics="kernel_launch_failures: 1\ngpu_utilization: 0%",
        ),
        specialist_followups={
            "runtime": "ROCm works. Not a runtime issue.",
            "dispatch": "Add gfx950 to AMD_SUPPORTED. CDNA4 is backwards-compatible with gfx942 kernels.",
            "kernel": "I was wrong — gfx950 does support the needed MFMA instructions. It's just the allowlist.",
            "loader": "No weight issues.",
        },
    ))

    scenarios.append(Scenario(
        id="arch_guard_05",
        root_cause="arch_guard",
        correct_fix="relax_arch_check",
        incident_ticket=(
            "INCIDENT: Triton kernel compilation fails on RTX 5090 for custom MoE layer. "
            "Error: 'target sm_120 not recognized'. Compiled fine for sm_90."
        ),
        hardware="NVIDIA SM120 (GeForce RTX 5090)",
        model_name="DeepSeek-V3-671B",
        backend="SGLang 0.5.x",
        initial_log=(
            "[Triton] Compiling MoE routing kernel for sm_120...\n"
            "[Triton] ERROR: Unknown target 'sm_120'\n"
            "[Triton] Known targets: sm_70, sm_75, sm_80, sm_86, sm_89, sm_90"
        ),
        initial_snippet=(
            "# triton/compiler/target.py\n"
            "KNOWN_TARGETS = ['sm_70','sm_75','sm_80','sm_86','sm_89','sm_90']\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA and Triton installed correctly.", 0.78, False),
            "dispatch": SpecialistOpinion(
                "Triton's target list doesn't include sm_120. Need to add Blackwell family.", 0.90, True
            ),
            "kernel": SpecialistOpinion(
                "The MoE kernel uses standard tl.dot which works on any SM >= 70.", 0.82, True
            ),
            "loader": SpecialistOpinion(
                "Weights load fine. Error is at JIT compilation stage.", 0.70, False
            ),
        },
        inspect_results=InspectResult(
            logs="[Triton] JIT target 'sm_120' not recognized\n[Triton] Compilation aborted before PTX generation",
            config="triton_target: sm_120\nknown_targets: [sm_70..sm_90]\ntriton_version: 3.2",
            snippet="# Triton target registry doesn't know sm_120\n# sm_120 can use sm_90 codegen path\n# Add sm_120 to target list or use family mapping",
            metrics="jit_compile_failures: 1\nkernel_cache_hits: 0",
        ),
        specialist_followups={
            "runtime": "No runtime issue. Triton JIT compiler is the blocker.",
            "dispatch": "Triton target registry needs sm_120. Can map to sm_90 codegen path since instruction set overlaps.",
            "kernel": "The kernel code is fine — it's the compiler target check, not the kernel logic.",
            "loader": "No weight involvement at this stage.",
        },
    ))

    # --- backend_whitelist additional scenarios ---
    scenarios.append(Scenario(
        id="backend_whitelist_03",
        root_cause="backend_whitelist",
        correct_fix="add_whitelist_entry",
        incident_ticket=(
            "INCIDENT: GPTQ quantization fails on B200 with 'GPU not whitelisted for Marlin'. "
            "Same quantized model serves fine on H100. B200 has FP16 working."
        ),
        hardware="NVIDIA B200",
        model_name="Mistral-Large-2",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Loading GPTQ model on B200...\n"
            "[vLLM] Marlin check: GPU 'NVIDIA B200' not whitelisted\n"
            "[vLLM] Available kernels for non-whitelisted: none\n"
            "[vLLM] ERROR: Cannot serve quantized model"
        ),
        initial_snippet=(
            "# vllm/quantization/marlin.py\n"
            "WHITELIST = {'A100','H100','A10G','L40S','RTX 4090'}\n"
            "if gpu_name not in WHITELIST:\n"
            "    raise RuntimeError('GPU not whitelisted')\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA runtime healthy on B200.", 0.80, False),
            "dispatch": SpecialistOpinion(
                "Whitelist check is string-based. 'B200' not in the set. Add it.", 0.93, True
            ),
            "kernel": SpecialistOpinion(
                "B200 FP8 is different from H100. Might need a different quantization kernel.", 0.50, False
            ),
            "loader": SpecialistOpinion("Quantized weights loaded correctly.", 0.75, False),
        },
        inspect_results=InspectResult(
            logs="[Marlin] GPU 'NVIDIA B200' not in whitelist\n[Marlin] Whitelist: {A100,H100,A10G,L40S,RTX 4090}",
            config="gpu_name: NVIDIA B200\nmarlin_whitelist: [A100,H100,A10G,L40S,RTX 4090]\nquant_method: gptq",
            snippet="# B200 supports all Marlin GEMM ops (INT4 deq + FP16 MMA)\n# Name-based whitelist just doesn't include it\n# Fix: add 'B200' or switch to arch-based check",
            metrics="quant_init_failures: 1\nfp16_serving: available\nquant_serving: blocked",
        ),
        specialist_followups={
            "runtime": "Runtime fine.",
            "dispatch": "Simple whitelist gap. Add 'B200' to WHITELIST set.",
            "kernel": "I was wrong — B200 Marlin kernels use same INT4 deq + MMA path as H100. Whitelist issue only.",
            "loader": "Weights are fine.",
        },
    ))

    scenarios.append(Scenario(
        id="backend_whitelist_04",
        root_cause="backend_whitelist",
        correct_fix="add_whitelist_entry",
        incident_ticket=(
            "INCIDENT: FlashInfer FP8 GEMM blocked on DGX Spark. "
            "Error: 'FP8 dispatch not available for this GPU'. "
            "SM121 should support FP8 natively."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="DeepSeek-R1-Distill-70B",
        backend="FlashInfer 0.4",
        initial_log=(
            "[FlashInfer] FP8 GEMM dispatch...\n"
            "[FlashInfer] GPU family check: sm_121\n"
            "[FlashInfer] FP8 whitelist: [sm_89, sm_90]\n"
            "[FlashInfer] ERROR: FP8 not available for sm_121"
        ),
        initial_snippet=(
            "# flashinfer/gemm/fp8_dispatch.py\n"
            "FP8_ENABLED_SM = {89, 90}  # Ada, Hopper\n"
            "# Missing SM12x which has FP8 MMA\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA 13 runtime fine.", 0.78, False),
            "dispatch": SpecialistOpinion(
                "FP8 dispatch whitelist only has Ada/Hopper. SM121 supports FP8 MMA natively but isn't listed.", 0.94, True
            ),
            "kernel": SpecialistOpinion(
                "SM121 FP8 might use different MMA instruction encoding.", 0.48, False
            ),
            "loader": SpecialistOpinion("FP8 weights loaded. Dispatch is the blocker.", 0.82, True),
        },
        inspect_results=InspectResult(
            logs="[FlashInfer] sm_121 not in FP8_ENABLED_SM {89, 90}\n[FlashInfer] FP8 GEMM dispatch blocked",
            config="gpu_sm: 121\nfp8_whitelist: [89, 90]\nfp8_hw_support: true",
            snippet="# SM121 uses m16n8k32 FP8 MMA (same encoding as SM90)\n# Just not in FP8_ENABLED_SM set\n# Add 120, 121 to enable FP8 dispatch",
            metrics="fp8_dispatch_blocked: true\nfp8_hw_capable: true\nfallback_to_bf16: not_attempted",
        ),
        specialist_followups={
            "runtime": "Runtime is fine.",
            "dispatch": "Add SM12x to FP8_ENABLED_SM. SM121 uses identical FP8 MMA to SM90.",
            "kernel": "I checked — SM121 uses the same m16n8k32 encoding as SM90. My concern was unfounded.",
            "loader": "FP8 weights are ready. Just need dispatch to be unblocked.",
        },
    ))

    scenarios.append(Scenario(
        id="backend_whitelist_05",
        root_cause="backend_whitelist",
        correct_fix="add_whitelist_entry",
        incident_ticket=(
            "INCIDENT: SGLang refuses to enable speculative decoding on RTX 5090. "
            "Error: 'Speculative decoding not supported for consumer GPUs'. "
            "Feature works on A100."
        ),
        hardware="NVIDIA SM120 (GeForce RTX 5090)",
        model_name="Llama-3.3-70B-Instruct",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] Speculative decoding requested...\n"
            "[SGLang] GPU: GeForce RTX 5090\n"
            "[SGLang] Spec decode whitelist: [A100, H100, A10G]\n"
            "[SGLang] ERROR: Consumer GPU not in spec-decode whitelist"
        ),
        initial_snippet=(
            "# sglang/server/spec_decode.py\n"
            "SPEC_DECODE_GPUS = ['A100', 'H100', 'A10G']\n"
            "# Only data center GPUs whitelisted\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime fine. GPU has 24GB VRAM.", 0.78, False),
            "dispatch": SpecialistOpinion(
                "RTX 5090 not in spec-decode whitelist. Datacenter-only check is too restrictive.", 0.91, True
            ),
            "kernel": SpecialistOpinion(
                "RTX 5090 might not have enough VRAM for speculative decoding with 70B.", 0.60, False
            ),
            "loader": SpecialistOpinion("Model weights fine.", 0.72, False),
        },
        inspect_results=InspectResult(
            logs="[SGLang] GPU 'GeForce RTX 5090' not in SPEC_DECODE_GPUS\n[SGLang] Whitelist is datacenter-only",
            config="gpu_name: GeForce RTX 5090\nspec_decode_whitelist: [A100,H100,A10G]\nvram: 32GB",
            snippet="# RTX 5090 has 32GB VRAM, sufficient for spec decode\n# Whitelist artificially restricts to datacenter GPUs\n# Add RTX 5090 or use VRAM-based check",
            metrics="spec_decode_attempts: 1\nspec_decode_blocked: true\nvram_available: 32GB",
        ),
        specialist_followups={
            "runtime": "No runtime issue.",
            "dispatch": "Add RTX 5090 to whitelist. 32GB VRAM is plenty for spec decode.",
            "kernel": "32GB is sufficient for speculative decoding with 70B quantized. VRAM isn't the issue.",
            "loader": "Weights loaded. Dispatch blocker only.",
        },
    ))

    # --- runtime_loader additional scenarios ---
    scenarios.append(Scenario(
        id="runtime_loader_03",
        root_cause="runtime_loader",
        correct_fix="fix_runtime_path",
        incident_ticket=(
            "INCIDENT: vLLM fails with 'libcublas.so.13 not found' on freshly provisioned node. "
            "nvidia-smi shows GPU. CUDA toolkit installed. Other CUDA apps work."
        ),
        hardware="NVIDIA H100",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Initializing CUDA...\n"
            "[vLLM] ERROR: libcublas.so.13: cannot open shared object file\n"
            "[vLLM] LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu\n"
            "[vLLM] Note: /usr/local/cuda-13/lib64 not in path"
        ),
        initial_snippet=(
            "# /etc/environment\n"
            "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu\n"
            "# Missing: /usr/local/cuda-13/lib64\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA 13 is installed but its lib64 directory isn't in LD_LIBRARY_PATH. Path fix needed.", 0.95, True
            ),
            "dispatch": SpecialistOpinion("Server crashes before any dispatch.", 0.65, False),
            "kernel": SpecialistOpinion("Not a kernel issue — can't load CUDA libraries.", 0.70, False),
            "loader": SpecialistOpinion(
                "Dynamic linker can't find libcublas.so.13. Add CUDA 13 lib path.", 0.90, True
            ),
        },
        inspect_results=InspectResult(
            logs="[ldconfig] libcublas.so.13 not in cache\n[System] /usr/local/cuda-13/lib64/libcublas.so.13 EXISTS but not in path",
            config="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu\ncuda_13_libs=/usr/local/cuda-13/lib64\nldconfig_cache: stale",
            snippet="# libcublas.so.13 exists at /usr/local/cuda-13/lib64/\n# But LD_LIBRARY_PATH doesn't include it\n# Fix: add /usr/local/cuda-13/lib64 to LD_LIBRARY_PATH",
            metrics="import_failures: 1\ncuda_available: false (library missing)",
        ),
        specialist_followups={
            "runtime": "Classic provisioning issue. CUDA installed but path not configured. Add to LD_LIBRARY_PATH.",
            "dispatch": "Nothing to dispatch — server won't start.",
            "kernel": "No kernel involvement.",
            "loader": "Add /usr/local/cuda-13/lib64 to LD_LIBRARY_PATH or run ldconfig.",
        },
    ))

    scenarios.append(Scenario(
        id="runtime_loader_04",
        root_cause="runtime_loader",
        correct_fix="fix_runtime_path",
        incident_ticket=(
            "INCIDENT: FlashInfer JIT compilation fails with 'nvcc not found'. "
            "GPU inference should work but JIT kernels can't compile. "
            "nvidia-smi works fine."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="Qwen3-235B-A22B",
        backend="FlashInfer 0.4",
        initial_log=(
            "[FlashInfer] JIT compiling attention kernel for sm_121...\n"
            "[FlashInfer] Searching for nvcc...\n"
            "[FlashInfer] ERROR: nvcc not found in PATH\n"
            "[FlashInfer] CUDA_HOME not set"
        ),
        initial_snippet=(
            "# Container environment\n"
            "PATH=/usr/local/bin:/usr/bin:/bin\n"
            "# Missing: /usr/local/cuda-13/bin (where nvcc lives)\n"
            "CUDA_HOME=  # not set\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA toolkit is installed but nvcc isn't in PATH and CUDA_HOME isn't set.", 0.93, True
            ),
            "dispatch": SpecialistOpinion("Dispatch can't run without JIT-compiled kernels.", 0.60, False),
            "kernel": SpecialistOpinion(
                "SM121 needs JIT compilation for attention kernels. Without nvcc, it can't compile.", 0.80, True
            ),
            "loader": SpecialistOpinion("Try using pre-compiled AOT kernels instead.", 0.45, False),
        },
        inspect_results=InspectResult(
            logs="[System] which nvcc -> not found\n[System] ls /usr/local/cuda-13/bin/nvcc -> EXISTS\n[System] CUDA_HOME unset",
            config="PATH=/usr/local/bin:/usr/bin:/bin\nCUDA_HOME=(unset)\nnvcc_location=/usr/local/cuda-13/bin/nvcc",
            snippet="# nvcc exists at /usr/local/cuda-13/bin/ but not in PATH\n# Fix: export CUDA_HOME=/usr/local/cuda-13\n# Fix: export PATH=$CUDA_HOME/bin:$PATH",
            metrics="jit_compile_attempts: 3\njit_compile_failures: 3\naot_kernels_available: false",
        ),
        specialist_followups={
            "runtime": "Set CUDA_HOME=/usr/local/cuda-13 and add its bin/ to PATH.",
            "dispatch": "Once nvcc is found, JIT compilation will work and dispatch proceeds normally.",
            "kernel": "The kernel code is ready to compile. Just need the compiler to be findable.",
            "loader": "AOT kernels aren't available for SM121 yet. JIT path is needed.",
        },
    ))

    scenarios.append(Scenario(
        id="runtime_loader_05",
        root_cause="runtime_loader",
        correct_fix="fix_runtime_path",
        incident_ticket=(
            "INCIDENT: Python can't import torch on MI300X node. "
            "Error: 'libtorch_hip.so: cannot open shared object'. "
            "PyTorch ROCm wheel installed but missing HIP libs."
        ),
        hardware="AMD MI300X",
        model_name="Mistral-Large-2",
        backend="vLLM 0.8.x",
        initial_log=(
            "[Python] import torch\n"
            "[Python] ERROR: libtorch_hip.so: cannot open shared object file\n"
            "[System] ROCm installed at /opt/rocm-6.3\n"
            "[System] LD_LIBRARY_PATH does not include /opt/rocm-6.3/lib"
        ),
        initial_snippet=(
            "# Container env\n"
            "LD_LIBRARY_PATH=/usr/local/lib\n"
            "# Needs: /opt/rocm-6.3/lib:/opt/rocm-6.3/hip/lib\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm 6.3 installed but libs not in LD_LIBRARY_PATH. Classic path issue.", 0.94, True
            ),
            "dispatch": SpecialistOpinion("Can't assess — Python crashes on import.", 0.50, False),
            "kernel": SpecialistOpinion("Maybe PyTorch ROCm wheel is for wrong ROCm version.", 0.55, False),
            "loader": SpecialistOpinion(
                "Dynamic linker needs /opt/rocm-6.3/lib in LD_LIBRARY_PATH.", 0.90, True
            ),
        },
        inspect_results=InspectResult(
            logs="[System] /opt/rocm-6.3/lib/libtorch_hip.so EXISTS\n[System] ldd: libtorch_hip.so => not found\n[System] LD_LIBRARY_PATH=/usr/local/lib only",
            config="LD_LIBRARY_PATH=/usr/local/lib\nrocm_path=/opt/rocm-6.3\nrocm_lib=/opt/rocm-6.3/lib",
            snippet="# ROCm libs at /opt/rocm-6.3/lib/ and /opt/rocm-6.3/hip/lib/\n# Not in LD_LIBRARY_PATH\n# Fix: export LD_LIBRARY_PATH=/opt/rocm-6.3/lib:/opt/rocm-6.3/hip/lib:$LD_LIBRARY_PATH",
            metrics="import_failures: 1\ntorch_available: false",
        ),
        specialist_followups={
            "runtime": "Add ROCm lib paths to LD_LIBRARY_PATH. Standard post-install issue.",
            "dispatch": "Can't run without PyTorch importing.",
            "kernel": "The ROCm version matches the wheel. It's just a path issue.",
            "loader": "Add /opt/rocm-6.3/lib to LD_LIBRARY_PATH.",
        },
    ))

    # --- backend_selector additional scenarios ---
    scenarios.append(Scenario(
        id="backend_selector_03",
        root_cause="backend_selector",
        correct_fix="switch_backend",
        incident_ticket=(
            "INCIDENT: SGLang MoE expert parallelism selecting wrong GEMM backend. "
            "Using generic GEMM instead of grouped GEMM for MoE layers. "
            "Throughput is 5x lower than expected."
        ),
        hardware="NVIDIA H100",
        model_name="DeepSeek-V3-671B",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] MoE layer: 256 experts, top-8 routing\n"
            "[SGLang] GEMM backend: generic (cublas)\n"
            "[SGLang] WARNING: Grouped GEMM backend not selected\n"
            "[SGLang] Throughput: 15 tok/s (expected: 80 tok/s)"
        ),
        initial_snippet=(
            "# sglang/moe/dispatch.py\n"
            "def select_moe_backend(num_experts, gpu):\n"
            "    if num_experts <= 64:\n"
            "        return 'grouped_gemm'\n"
            "    return 'generic'  # Wrong fallback for large expert count\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA runtime fine. No errors.", 0.75, False),
            "dispatch": SpecialistOpinion(
                "MoE backend selector falls back to generic GEMM when experts > 64. "
                "Should use grouped GEMM for any expert count on H100.", 0.95, True
            ),
            "kernel": SpecialistOpinion(
                "Generic cuBLAS GEMM launches one kernel per expert. Grouped GEMM batches them. "
                "Switch to grouped GEMM backend.", 0.88, True
            ),
            "loader": SpecialistOpinion("Weights loaded. Not a loading issue.", 0.72, False),
        },
        inspect_results=InspectResult(
            logs="[SGLang] 256 experts > 64 threshold -> generic backend\n[SGLang] Each expert: separate cuBLAS call\n[SGLang] Kernel launch overhead: 256 launches/layer",
            config="num_experts: 256\nmoe_backend: generic\nthreshold: 64\ngpu: H100",
            snippet="# Backend selector has wrong threshold logic\n# Should use grouped_gemm for ALL expert counts on H100\n# Current: only grouped_gemm when experts <= 64",
            metrics="throughput_tok_s: 15\nexpected_throughput: 80\nkernel_launches_per_step: 256\ngpu_utilization: 18%",
        ),
        specialist_followups={
            "runtime": "No runtime issues.",
            "dispatch": "Switch to grouped_gemm backend. The 64-expert threshold is a bug.",
            "kernel": "Grouped GEMM would batch all 256 experts into one kernel launch. 10-15x fewer launches.",
            "loader": "Not a weight issue.",
        },
    ))

    scenarios.append(Scenario(
        id="backend_selector_04",
        root_cause="backend_selector",
        correct_fix="switch_backend",
        incident_ticket=(
            "INCIDENT: Attention on B200 using FlashAttention v1 path instead of v2. "
            "Memory usage 3x higher than expected. OOM on large batch sizes. "
            "Same model fits in memory on H100."
        ),
        hardware="NVIDIA B200",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Attention backend: flash_attn_v1\n"
            "[vLLM] WARNING: v2 backend not selected (GPU not in v2 list)\n"
            "[vLLM] Memory: attention uses O(n^2) instead of O(n)\n"
            "[vLLM] OOM at batch_size=32 (expected to fit at batch_size=128)"
        ),
        initial_snippet=(
            "# vllm/attention/selector.py\n"
            "def select_flash_version(gpu_sm):\n"
            "    if gpu_sm in {80, 86, 89, 90}:\n"
            "        return 'v2'\n"
            "    return 'v1'  # B200 (sm_120) falls here\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA runtime OK. Memory allocation works.", 0.75, False),
            "dispatch": SpecialistOpinion(
                "Backend selector picks FA v1 for sm_120. B200 supports v2 — selector needs updating.", 0.93, True
            ),
            "kernel": SpecialistOpinion(
                "FA v1 uses O(n^2) memory. v2 uses O(n). That explains the OOM.", 0.85, True
            ),
            "loader": SpecialistOpinion(
                "Maybe model weights are larger than expected for this architecture.", 0.45, False
            ),
        },
        inspect_results=InspectResult(
            logs="[vLLM] sm_120 not in {80,86,89,90} -> flash_attn_v1\n[vLLM] FA v1 attention memory: O(seq_len^2)\n[vLLM] OOM threshold hit at 32 batch",
            config="gpu_sm: 120\nflash_attn_version: v1\nv2_supported_sm: [80,86,89,90]\nmemory_profile: quadratic",
            snippet="# B200 (sm_120) supports FlashAttention v2\n# Selector only checks old SM list\n# Fix: add sm_120 to v2 supported set or switch to v2 backend",
            metrics="attention_memory_gb: 24.5\nexpected_attention_memory_gb: 2.1\nbatch_size_limit: 32\nexpected_batch_limit: 128",
        ),
        specialist_followups={
            "runtime": "Memory system works. Problem is FA v1's quadratic memory.",
            "dispatch": "Add sm_120 to v2 supported set. B200 has full v2 support.",
            "kernel": "FA v1 materializes full attention matrix. v2 uses tiling. Fix the selector.",
            "loader": "Weight size is correct. It's the attention memory that's excessive.",
        },
    ))

    scenarios.append(Scenario(
        id="backend_selector_05",
        root_cause="backend_selector",
        correct_fix="switch_backend",
        incident_ticket=(
            "INCIDENT: MI300X inference using CK (Composable Kernel) attention but should use Triton. "
            "CK path has a known bug with GQA + variable-length sequences. "
            "Random crashes during batched inference."
        ),
        hardware="AMD MI300X",
        model_name="Qwen3-235B-A22B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] AMD GPU detected -> Composable Kernel attention\n"
            "[vLLM] GQA + varlen: CK backend selected\n"
            "[vLLM] CRASH: segfault in ck_attention_varlen_gqa\n"
            "[vLLM] This is a known CK bug. Use Triton backend instead."
        ),
        initial_snippet=(
            "# vllm/attention/backends/rocm.py\n"
            "def get_rocm_backend(config):\n"
            "    return 'composable_kernel'  # Always uses CK\n"
            "    # Should check for known CK bugs and use Triton\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("ROCm runtime fine before the segfault.", 0.72, False),
            "dispatch": SpecialistOpinion(
                "Backend selector always picks CK on AMD. Should use Triton for GQA+varlen due to known CK bug.", 0.94, True
            ),
            "kernel": SpecialistOpinion(
                "Known CK bug with GQA + varlen sequences. Triton attention works correctly.", 0.90, True
            ),
            "loader": SpecialistOpinion("Might be a weight alignment issue for AMD.", 0.40, False),
        },
        inspect_results=InspectResult(
            logs="[CK] ck_attention_varlen_gqa: SIGSEGV\n[CK] Known issue: GQA + variable-length triggers OOB access\n[Triton] Triton attention works for this config",
            config="rocm_attention: composable_kernel\ngqa_enabled: true\nvarlen: true\nknown_ck_bugs: [gqa_varlen]",
            snippet="# CK has a bug in GQA + varlen attention (OOB memory access)\n# Triton backend handles this correctly\n# Fix: route GQA+varlen to Triton on AMD",
            metrics="crashes: 3/10 requests\nsegfaults: 3\ntriton_fallback: not_configured",
        ),
        specialist_followups={
            "runtime": "The segfault is in CK library code, not a runtime issue.",
            "dispatch": "Switch to Triton attention for GQA+varlen on AMD. CK bug is known and not yet fixed upstream.",
            "kernel": "CK varlen GQA kernel has off-by-one in tile boundary. Triton implementation doesn't have this bug.",
            "loader": "Not a weight issue. The crash is in the attention computation.",
        },
    ))

    # --- model_config additional scenarios ---
    scenarios.append(Scenario(
        id="model_config_03",
        root_cause="model_config",
        correct_fix="update_model_config",
        incident_ticket=(
            "INCIDENT: DeepSeek MLA attention produces wrong KV cache size. "
            "OOM on sequences that should fit. Config shows standard MHA dimensions "
            "but model uses MLA with compressed KV."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="DeepSeek-V3-671B",
        backend="FlashInfer 0.4",
        initial_log=(
            "[FlashInfer] KV cache: allocating for 64 KV heads x 128 dim = 8192 per token\n"
            "[FlashInfer] Expected MLA: kv_lora_rank=512, much smaller KV cache\n"
            "[FlashInfer] OOM: KV cache exceeds 80GB at seq_len=4096"
        ),
        initial_snippet=(
            "# config.json\n"
            '{\n'
            '  "num_key_value_heads": 64,\n'
            '  "head_dim": 128\n'
            '  // Missing: kv_lora_rank, qk_rope_head_dim for MLA\n'
            '}\n'
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Memory allocation works. Just allocating too much.", 0.72, False),
            "dispatch": SpecialistOpinion("FlashInfer correctly reading config. Config is the problem.", 0.68, False),
            "kernel": SpecialistOpinion(
                "MLA attention needs kv_lora_rank in config to use compressed KV. "
                "Without it, falls back to full MHA KV cache sizing.", 0.92, True
            ),
            "loader": SpecialistOpinion(
                "Config.json doesn't have MLA parameters. Need kv_lora_rank=512 and qk_rope_head_dim=64.", 0.93, True
            ),
        },
        inspect_results=InspectResult(
            logs="[FlashInfer] No kv_lora_rank in config -> full MHA KV\n[FlashInfer] KV per token: 64*128*2=16384 (should be 512*2=1024 with MLA)\n[FlashInfer] 16x memory overhead",
            config="num_kv_heads: 64\nhead_dim: 128\nkv_lora_rank: (missing)\nqk_rope_head_dim: (missing)\nattention_type: inferred as MHA",
            snippet="# DeepSeek MLA config needs:\n# kv_lora_rank: 512\n# qk_rope_head_dim: 64\n# Without these, system allocates full MHA KV cache",
            metrics="kv_cache_per_token_bytes: 16384\nexpected_bytes: 1024\nmemory_overhead: 16x\noom_at_seq_len: 4096",
        ),
        specialist_followups={
            "runtime": "No runtime issue. Memory allocation succeeds until OOM.",
            "dispatch": "Config drives the dispatch. Fix the config.",
            "kernel": "MLA kernel exists but won't activate without kv_lora_rank in config.",
            "loader": "Add kv_lora_rank=512 and qk_rope_head_dim=64 to config.json.",
        },
    ))

    scenarios.append(Scenario(
        id="model_config_04",
        root_cause="model_config",
        correct_fix="update_model_config",
        incident_ticket=(
            "INCIDENT: Llama-4 Maverick MoE model failing with 'Expected 128 experts'. "
            "Config lists num_local_experts=128 but actual checkpoint uses sparse layout "
            "with 16 active experts per token from 128 total, stored differently."
        ),
        hardware="NVIDIA H100",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] MoE init: 128 experts, 2 active per token\n"
            "[vLLM] Loading expert weights...\n"
            "[vLLM] WARNING: Expert weight tensor shape doesn't match config\n"
            "[vLLM] Expected: [128, hidden, ffn] Got: [128, ffn//4, hidden]"
        ),
        initial_snippet=(
            "# config.json\n"
            '{\n'
            '  "num_local_experts": 128,\n'
            '  "num_experts_per_tok": 2,\n'
            '  "expert_layout": "dense"\n'
            '  // Should be "interleaved" for Maverick architecture\n'
            '}\n'
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime OK.", 0.75, False),
            "dispatch": SpecialistOpinion("MoE dispatch looks correct for the config.", 0.60, False),
            "kernel": SpecialistOpinion(
                "Expert weight tensor shape is transposed vs config expectation. "
                "Config says dense layout but weights are in interleaved format.", 0.85, True
            ),
            "loader": SpecialistOpinion(
                "Config expert_layout should be 'interleaved' not 'dense'. "
                "Maverick uses interleaved expert storage.", 0.93, True
            ),
        },
        inspect_results=InspectResult(
            logs="[vLLM] Config: expert_layout=dense\n[vLLM] Actual weights: interleaved layout\n[vLLM] Shape mismatch in MoE layer 0",
            config="expert_layout: dense (wrong)\nactual_layout: interleaved\nnum_experts: 128\nexperts_per_token: 2",
            snippet="# Maverick checkpoint uses interleaved expert layout:\n# experts stored as [expert_idx, ffn_chunk, hidden]\n# Config says 'dense' which expects [expert_idx, hidden, ffn]\n# Fix: set expert_layout='interleaved'",
            metrics="model_load_progress: 5%\nshape_mismatches: 128\nerror_at: expert_layer_0",
        ),
        specialist_followups={
            "runtime": "Not a runtime issue.",
            "dispatch": "Dispatch follows config. Fix the config first.",
            "kernel": "Weight shapes don't match the layout assumption. Config needs updating.",
            "loader": "Set expert_layout to 'interleaved' in config.json. Maverick stores experts interleaved.",
        },
    ))

    scenarios.append(Scenario(
        id="model_config_05",
        root_cause="model_config",
        correct_fix="update_model_config",
        incident_ticket=(
            "INCIDENT: Sliding window attention not activating for Mistral model. "
            "Memory usage growing linearly with sequence length. "
            "Should plateau after window size."
        ),
        hardware="NVIDIA B200",
        model_name="Mistral-Large-2",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] Attention config: full attention (no sliding window)\n"
            "[SGLang] KV cache growing linearly with seq_len\n"
            "[SGLang] Memory at 32k tokens: 40GB (expected: 12GB with sliding window)\n"
            "[SGLang] sliding_window not found in config.json"
        ),
        initial_snippet=(
            "# config.json\n"
            '{\n'
            '  "max_position_embeddings": 32768,\n'
            '  "num_attention_heads": 96\n'
            '  // Missing: "sliding_window": 4096\n'
            '}\n'
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime fine. Memory growing as expected for full attention.", 0.78, False),
            "dispatch": SpecialistOpinion(
                "Backend correctly doing full attention because config doesn't specify sliding window.", 0.70, True
            ),
            "kernel": SpecialistOpinion(
                "Kernel supports sliding window. Config just needs the parameter.", 0.82, True
            ),
            "loader": SpecialistOpinion(
                "Config.json missing sliding_window=4096. Mistral models use 4096-token sliding window.", 0.92, True
            ),
        },
        inspect_results=InspectResult(
            logs="[SGLang] No sliding_window in config -> full attention\n[SGLang] KV cache: 32k * 96 heads * 128 dim * 2 = 40GB",
            config="sliding_window: null\nmax_position_embeddings: 32768\nexpected_sliding_window: 4096",
            snippet="# Mistral-Large-2 uses 4096-token sliding window\n# Config missing: sliding_window: 4096\n# Without it, full O(n) KV cache used",
            metrics="kv_cache_32k_gb: 40\nexpected_kv_cache_gb: 12\nmemory_overhead: 3.3x",
        ),
        specialist_followups={
            "runtime": "Memory growth is correct for the config given. Fix the config.",
            "dispatch": "Backend reads config. Add sliding_window=4096.",
            "kernel": "Sliding window attention kernel exists. Just needs the config parameter to activate.",
            "loader": "Add sliding_window: 4096 to config.json.",
        },
    ))

    # --- weight_layout additional scenarios ---
    scenarios.append(Scenario(
        id="weight_layout_03",
        root_cause="weight_layout",
        correct_fix="fix_weight_mapping",
        incident_ticket=(
            "INCIDENT: Model outputs garbage after quantization with GPTQ. "
            "Original FP16 model is fine. GPTQ quantization reports success "
            "but group indices are misaligned."
        ),
        hardware="NVIDIA H100",
        model_name="Qwen3-235B-A22B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Loading GPTQ-quantized Qwen3...\n"
            "[vLLM] Quantization: 4-bit, group_size=128\n"
            "[vLLM] WARNING: g_idx tensor shape mismatch in layer 0\n"
            "[vLLM] Output: incoherent (perplexity 1247)"
        ),
        initial_snippet=(
            "# GPTQ packing\n"
            "# g_idx maps each weight column to its quantization group\n"
            "# Expected shape: [in_features]\n"
            "# Got shape: [in_features // group_size] (wrong!)\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA fine. Kernels launch.", 0.78, False),
            "dispatch": SpecialistOpinion("GPTQ backend selected correctly.", 0.65, False),
            "kernel": SpecialistOpinion(
                "Dequantization kernel gets wrong group assignments because g_idx is wrong shape.", 0.82, True
            ),
            "loader": SpecialistOpinion(
                "GPTQ group index (g_idx) tensor has wrong shape. The quantization script packed it incorrectly. "
                "Needs regeneration with correct per-column group mapping.", 0.94, True
            ),
        },
        inspect_results=InspectResult(
            logs="[GPTQ] g_idx shape: [128] (wrong) vs expected [16384]\n[GPTQ] Each column needs its own group index\n[GPTQ] Wrong g_idx causes random dequant scale selection",
            config="group_size: 128\nin_features: 16384\ng_idx_shape: [128]\nexpected_g_idx_shape: [16384]",
            snippet="# g_idx should be per-column: shape [in_features]\n# But quantizer produced per-group: shape [in_features//group_size]\n# This assigns wrong scales during dequantization",
            metrics="perplexity: 1247\nexpected_perplexity: 10.2\nlayers_affected: all\ng_idx_misaligned: true",
        ),
        specialist_followups={
            "runtime": "No runtime issues.",
            "dispatch": "Backend selection is fine.",
            "kernel": "Kernel dequantizes correctly when given right g_idx. Fix the mapping.",
            "loader": "Regenerate g_idx with per-column mapping (shape [in_features], not [in_features//group_size]).",
        },
    ))

    scenarios.append(Scenario(
        id="weight_layout_04",
        root_cause="weight_layout",
        correct_fix="fix_weight_mapping",
        incident_ticket=(
            "INCIDENT: FP8 model on MI300X gives NaN after first layer. "
            "Dequantization scales appear transposed. "
            "Same checkpoint works on NVIDIA with e4m3fn format."
        ),
        hardware="AMD MI300X",
        model_name="DeepSeek-R1-Distill-70B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] FP8 dequant: loading scales...\n"
            "[vLLM] Scale tensor shape: [out_features, 1] — expected [1, out_features] for AMD\n"
            "[vLLM] Layer 0 output: NaN (scale applied to wrong dimension)\n"
            "[vLLM] All subsequent layers: NaN"
        ),
        initial_snippet=(
            "# fp8_weights.py\n"
            "# NVIDIA: scales are per-output-channel [out, 1]\n"
            "# AMD: scales are per-input-channel [1, in]\n"
            "# Converter didn't transpose for AMD\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("ROCm runtime fine.", 0.78, False),
            "dispatch": SpecialistOpinion("FP8 backend selected. Format mismatch possible.", 0.65, False),
            "kernel": SpecialistOpinion(
                "FP8 GEMM applies scale in wrong dimension due to transposed scale tensor.", 0.85, True
            ),
            "loader": SpecialistOpinion(
                "FP8 scale tensors need transposing for AMD. NVIDIA uses [out,1], AMD uses [1,in]. "
                "Weight converter didn't handle this.", 0.95, True
            ),
        },
        inspect_results=InspectResult(
            logs="[FP8] Scale shape [4096,1] but AMD MFMA expects [1,4096]\n[FP8] Dequant: scale broadcast on wrong axis -> NaN\n[FP8] First non-NaN result never produced",
            config="fp8_scale_shape: [out_features, 1]\namd_expected: [1, in_features]\nscale_transpose_needed: true",
            snippet="# NVIDIA layout: W_fp8 * scale[out,1] -> per-output-channel\n# AMD layout: W_fp8 * scale[1,in] -> per-input-channel\n# Converter assumed NVIDIA layout\n# Fix: transpose scales for AMD",
            metrics="nan_outputs: 100%\nlayers_producing_nan: all\nfirst_nan_at: layer_0",
        ),
        specialist_followups={
            "runtime": "Not a runtime issue.",
            "dispatch": "FP8 selected correctly. Scale orientation is the issue.",
            "kernel": "GEMM kernel applies scale along wrong dimension. Transpose the scales.",
            "loader": "Transpose FP8 scale tensors from [out,1] to [1,in] for AMD.",
        },
    ))

    scenarios.append(Scenario(
        id="weight_layout_05",
        root_cause="weight_layout",
        correct_fix="fix_weight_mapping",
        incident_ticket=(
            "INCIDENT: Embedding layer produces identical vectors for all tokens. "
            "After checkpoint conversion, embedding weights appear row-shuffled. "
            "Tokenizer maps to wrong rows."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] Embedding layer: 128256 tokens x 4096 dim\n"
            "[SGLang] Token 'Hello' -> embedding row 85432 (expected: row 9906)\n"
            "[SGLang] All outputs identical — embeddings mapped to wrong rows\n"
            "[SGLang] Suspect: tokenizer vocab offset not applied during conversion"
        ),
        initial_snippet=(
            "# convert_checkpoint.py\n"
            "embed = original_weights['embed_tokens.weight']  # [128256, 4096]\n"
            "# BUG: added_tokens offset not applied\n"
            "# Tokenizer expects base_vocab at rows 0-127999\n"
            "# Converter put added_tokens at rows 0-255\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime fine. Model loads.", 0.75, False),
            "dispatch": SpecialistOpinion("Backend dispatch correct.", 0.68, False),
            "kernel": SpecialistOpinion(
                "Embedding lookup works mechanically but returns wrong vectors. Data issue.", 0.78, True
            ),
            "loader": SpecialistOpinion(
                "Embedding weight rows are misaligned after conversion. Tokenizer indices map to wrong rows. "
                "Converter needs to preserve original row ordering.", 0.94, True
            ),
        },
        inspect_results=InspectResult(
            logs="[SGLang] Token 'Hello' (id=9906) -> embedding from original row 85432\n[SGLang] Row mapping offset: 75526\n[SGLang] Converter applied wrong row permutation",
            config="vocab_size: 128256\nembed_dim: 4096\nrow_offset_error: 75526",
            snippet="# Converter reordered rows: put added_tokens (256) first, then base vocab\n# Tokenizer expects base vocab at row 0\n# Fix: preserve original row order in embedding conversion",
            metrics="embedding_cosine_sim_to_expected: 0.02\nall_outputs_identical: true\nperplexity: infinity",
        ),
        specialist_followups={
            "runtime": "No runtime issue.",
            "dispatch": "Dispatch is correct.",
            "kernel": "Embedding lookup returns whatever is at the indexed row. The rows are just wrong.",
            "loader": "Converter put added_tokens at index 0. Fix: keep original row order.",
        },
    ))

    # --- Additional eval scenarios (_06 suffix) ---
    scenarios.append(Scenario(
        id="arch_guard_06",
        root_cause="arch_guard",
        correct_fix="relax_arch_check",
        incident_ticket=(
            "INCIDENT: CUTLASS GEMM kernel rejects SM121 with 'unsupported architecture'. "
            "is_family_of() check fails because SM121 not in family table. "
            "FP8 inference completely blocked."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="Mistral-Large-2",
        backend="TensorRT-LLM 0.18",
        initial_log=(
            "[CUTLASS] is_family_of(sm_121, sm_90) = false\n"
            "[CUTLASS] SM121 not registered in family hierarchy\n"
            "[CUTLASS] FP8 GEMM dispatch: BLOCKED"
        ),
        initial_snippet=(
            "# cutlass/arch/family.py\n"
            "FAMILY_MAP = {90: [90], 89: [89], 86: [86], 80: [80]}\n"
            "# SM121 not in any family\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA 13 fine.", 0.78, False),
            "dispatch": SpecialistOpinion(
                "CUTLASS family map doesn't include SM12x. Need to register SM120/121 family.", 0.93, True
            ),
            "kernel": SpecialistOpinion(
                "The kernel weight format might be wrong for SM121.", 0.40, False
            ),
            "loader": SpecialistOpinion("Engine built. Weights loaded. GEMM dispatch blocked.", 0.70, False),
        },
        inspect_results=InspectResult(
            logs="[CUTLASS] FAMILY_MAP has no entry for 121\n[CUTLASS] is_family_of(121, 90) -> False\n[CUTLASS] FP8 GEMM requires family >= 90",
            config="gpu_sm: 121\nfamily_map: {90:[90],89:[89],...}\nsm121_family: undefined",
            snippet="# SM12x is its own family but shares FP8 MMA with SM90\n# Fix: add 120: [120, 121] and 121: [120, 121] to FAMILY_MAP\n# Or: register SM12x as SM90-compatible for GEMM",
            metrics="fp8_gemm_blocked: true\nbf16_gemm: functional",
        ),
        specialist_followups={
            "runtime": "Runtime fine.",
            "dispatch": "Register SM12x family in CUTLASS. SM121 FP8 MMA is SM90-compatible.",
            "kernel": "Weight format is fine. It's the arch family check blocking dispatch.",
            "loader": "Weights loaded correctly. GEMM dispatch is the issue.",
        },
    ))

    scenarios.append(Scenario(
        id="backend_selector_06",
        root_cause="backend_selector",
        correct_fix="switch_backend",
        incident_ticket=(
            "INCIDENT: DGX Spark running PagedAttention v1 instead of v2. "
            "Prefix caching not working. Cache hit rate near 0%. "
            "Same prompts re-computed every request."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="DeepSeek-V3-671B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] PagedAttention version: v1\n"
            "[vLLM] Prefix caching: disabled (requires PA v2)\n"
            "[vLLM] Cache hit rate: 0.1% (expected: 60%+ with repeated prefixes)\n"
            "[vLLM] TTFT p99: 2100ms (expected: 400ms with caching)"
        ),
        initial_snippet=(
            "# vllm/core/scheduler.py\n"
            "def select_paged_attention(gpu_sm):\n"
            "    if gpu_sm >= 80 and gpu_sm <= 90:\n"
            "        return 'v2'  # with prefix caching\n"
            "    return 'v1'  # SM121 > 90, falls here\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("CUDA runtime fine. Server runs.", 0.75, False),
            "dispatch": SpecialistOpinion(
                "PagedAttention version selector has range bug. SM121 > 90 so gets v1 without prefix caching.", 0.94, True
            ),
            "kernel": SpecialistOpinion(
                "PA v2 kernel works on SM121. It's the selector that's wrong.", 0.85, True
            ),
            "loader": SpecialistOpinion("Model loaded fine. Not a weight issue.", 0.72, False),
        },
        inspect_results=InspectResult(
            logs="[vLLM] sm_121 not in range [80,90] -> PA v1\n[vLLM] PA v1 doesn't support prefix caching\n[vLLM] Every prefix re-computed from scratch",
            config="paged_attention: v1\nprefix_caching: disabled\ngpu_sm: 121\nv2_range: [80, 90]",
            snippet="# PA v2 supports prefix caching, reducing TTFT 3-5x\n# Selector range [80,90] excludes SM121\n# Fix: include SM12x in v2-eligible set",
            metrics="cache_hit_rate: 0.1%\nexpected_cache_hit_rate: 62%\nttft_p99_ms: 2100\nexpected_ttft_ms: 400",
        ),
        specialist_followups={
            "runtime": "Server runs fine. Performance issue only.",
            "dispatch": "Fix the range check to include SM12x. PA v2 works on SM121.",
            "kernel": "PA v2 kernel is compatible. Just need the selector to pick it.",
            "loader": "Not a loading issue.",
        },
    ))

    scenarios.append(Scenario(
        id="runtime_loader_06",
        root_cause="runtime_loader",
        correct_fix="fix_runtime_path",
        incident_ticket=(
            "INCIDENT: Container on B200 node fails with 'CUDA driver version insufficient'. "
            "Host has driver 565 but container sees driver 535. "
            "nvidia-smi inside container shows old driver."
        ),
        hardware="NVIDIA B200",
        model_name="Llama-3.3-70B-Instruct",
        backend="vLLM 0.8.x",
        initial_log=(
            "[Container] nvidia-smi: Driver Version: 535.183.01\n"
            "[Host] nvidia-smi: Driver Version: 565.57.01\n"
            "[vLLM] CUDA 13 requires driver >= 560\n"
            "[vLLM] ERROR: CUDA driver version insufficient for CUDA runtime"
        ),
        initial_snippet=(
            "# Docker run command\n"
            "docker run --gpus all \\\n"
            "  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\\n"
            "  -e NVIDIA_VISIBLE_DEVICES=all \\\n"
            "  # Missing: --runtime=nvidia or proper CDI config\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "Container seeing old driver. Docker GPU passthrough not configured correctly. "
                "Need proper nvidia-container-runtime setup.", 0.94, True
            ),
            "dispatch": SpecialistOpinion("Server never starts. Can't assess dispatch.", 0.50, False),
            "kernel": SpecialistOpinion(
                "Maybe the B200 needs a newer CUDA toolkit version.", 0.45, False
            ),
            "loader": SpecialistOpinion(
                "Container's nvidia driver libs are stale. Bind mount is pointing to wrong driver version.", 0.88, True
            ),
        },
        inspect_results=InspectResult(
            logs="[Container] /usr/lib/x86_64-linux-gnu/libnvidia-ml.so -> driver 535\n[Host] /usr/lib/x86_64-linux-gnu/libnvidia-ml.so -> driver 565\n[Docker] nvidia-container-runtime not in daemon.json",
            config="host_driver: 565.57.01\ncontainer_driver: 535.183.01\nnvidia_runtime: not_configured",
            snippet="# Docker daemon.json missing nvidia runtime\n# Container bundles old driver libs instead of using host driver\n# Fix: configure nvidia-container-runtime or CDI",
            metrics="container_start_failures: 1\ndriver_mismatch: true\ncuda_init: failed",
        ),
        specialist_followups={
            "runtime": "nvidia-container-toolkit needs to be configured to pass host driver into container.",
            "dispatch": "Can't run without CUDA init.",
            "kernel": "The toolkit version is fine. It's the driver passthrough that's broken.",
            "loader": "Container needs host's driver libs mounted. Fix Docker runtime config.",
        },
    ))

    scenarios.append(Scenario(
        id="model_config_06",
        root_cause="model_config",
        correct_fix="update_model_config",
        incident_ticket=(
            "INCIDENT: BF16 model serving on MI300X has 2x expected memory usage. "
            "Config says float16 dtype but model should use bfloat16. "
            "Unnecessary fp16->bf16 conversion happening at runtime."
        ),
        hardware="AMD MI300X",
        model_name="DeepSeek-R1-Distill-70B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Config dtype: float16\n"
            "[vLLM] Actual weights: bfloat16\n"
            "[vLLM] Runtime conversion float16 config -> bfloat16 weights\n"
            "[vLLM] Extra memory for conversion buffers: 35GB"
        ),
        initial_snippet=(
            "# config.json\n"
            '{\n'
            '  "torch_dtype": "float16"\n'
            '  // Actual checkpoint is bfloat16\n'
            '  // Mismatch causes runtime conversion overhead\n'
            '}\n'
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("ROCm runtime healthy. Memory available.", 0.78, False),
            "dispatch": SpecialistOpinion("Backend dispatch fine.", 0.65, False),
            "kernel": SpecialistOpinion(
                "Kernels running with dtype conversion overhead. "
                "Config says fp16 but weights are bf16, so vLLM converts at load time.", 0.82, True
            ),
            "loader": SpecialistOpinion(
                "Config torch_dtype=float16 doesn't match checkpoint dtype=bfloat16. "
                "Fix config to say bfloat16 to avoid conversion overhead.", 0.93, True
            ),
        },
        inspect_results=InspectResult(
            logs="[vLLM] Config: float16, Checkpoint: bfloat16\n[vLLM] Allocating conversion buffers: 35GB\n[vLLM] Total memory: model(35GB) + conversion(35GB) = 70GB",
            config="torch_dtype: float16\ncheckpoint_dtype: bfloat16\nmismatch: true",
            snippet="# Config says float16 but checkpoint is bfloat16\n# vLLM allocates both versions during conversion\n# Fix: set torch_dtype='bfloat16' in config.json",
            metrics="memory_used_gb: 70\nexpected_memory_gb: 35\nconversion_overhead_gb: 35",
        ),
        specialist_followups={
            "runtime": "Memory subsystem fine. Just using too much.",
            "dispatch": "Dispatch fine after conversion.",
            "kernel": "Conversion overhead is the issue. Fix config to match checkpoint dtype.",
            "loader": "Set torch_dtype to bfloat16 in config.json.",
        },
    ))

    scenarios.append(Scenario(
        id="weight_layout_06",
        root_cause="weight_layout",
        correct_fix="fix_weight_mapping",
        incident_ticket=(
            "INCIDENT: Rotary position encoding giving wrong angles after checkpoint merge. "
            "Two LoRA adapters merged into base model, but RoPE inv_freq tensor "
            "accidentally overwritten with adapter values. Outputs degrade past position 128."
        ),
        hardware="NVIDIA H100",
        model_name="Mistral-Large-2",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Loading merged checkpoint...\n"
            "[vLLM] RoPE inv_freq shape: [64] (correct)\n"
            "[vLLM] RoPE inv_freq values: [0.001, 0.001, ...] (all same — WRONG)\n"
            "[vLLM] Expected: geometric sequence 1/10000^(2i/d)"
        ),
        initial_snippet=(
            "# merge_lora.py\n"
            "# BUG: LoRA merge accidentally overwrote inv_freq\n"
            "merged['inv_freq'] = adapter_state['inv_freq']  # adapter had dummy values\n"
            "# Should have kept base model's inv_freq\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion("Runtime fine.", 0.78, False),
            "dispatch": SpecialistOpinion("Backend dispatch correct.", 0.65, False),
            "kernel": SpecialistOpinion(
                "RoPE kernel computes correct rotations for the freq values given. But freq values are wrong.", 0.80, True
            ),
            "loader": SpecialistOpinion(
                "LoRA merge script overwrote inv_freq with adapter's dummy values. "
                "Need to restore base model's inv_freq or regenerate from formula.", 0.95, True
            ),
        },
        inspect_results=InspectResult(
            logs="[RoPE] inv_freq: all values = 0.001 (constant)\n[RoPE] Expected: geometric decay from 1.0 to 1e-4\n[RoPE] Position encoding essentially constant -> no position info after ~128 tokens",
            config="inv_freq_values: [0.001]*64\nexpected: geometric_series(1/10000, dim=128)\nrope_theta: 10000",
            snippet="# inv_freq should be: 1 / (theta ** (torch.arange(0, dim, 2) / dim))\n# Instead: all 0.001 from LoRA adapter dummy init\n# Fix: regenerate inv_freq from formula or restore from base model",
            metrics="quality_0_128: 90%\nquality_128_1k: 25%\nquality_1k_plus: 5%",
        ),
        specialist_followups={
            "runtime": "No runtime issue.",
            "dispatch": "Dispatch correct.",
            "kernel": "RoPE kernel works. Just getting wrong frequencies.",
            "loader": "Restore inv_freq from base model. LoRA merge script has a bug that overwrites non-LoRA tensors.",
        },
    ))

    # --- memory_oom scenarios ---
    scenarios.append(Scenario(
        id="memory_oom_01",
        root_cause="memory_oom",
        correct_fix="tune_memory_config",
        incident_ticket=(
            "INCIDENT: vLLM OOM crash serving DeepSeek-V3-671B on 8xH100. "
            "Model loads successfully but crashes after ~50 concurrent requests. "
            "GPU memory fragmentation suspected. KV cache allocation fails."
        ),
        hardware="NVIDIA H100",
        model_name="DeepSeek-V3-671B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Model loaded: 671B params across 8 GPUs (tensor parallel=8)\n"
            "[vLLM] KV cache: allocated 45GB per GPU (gpu_memory_utilization=0.95)\n"
            "[vLLM] Serving... 48 concurrent requests OK\n"
            "[vLLM] Request 51: torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 2.1 GiB. GPU 3 has 0.8 GiB free."
        ),
        initial_snippet=(
            "# vllm serve config\n"
            "gpu_memory_utilization: 0.95\n"
            "max_num_seqs: 256\n"
            "max_model_len: 32768\n"
            "# No swap space configured\n"
            "# No memory headroom for activation spikes\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime is healthy. OOM is a memory planning issue, not runtime.", 0.80, False
            ),
            "dispatch": SpecialistOpinion(
                "Backend dispatch is fine. This is a memory capacity issue.", 0.65, False
            ),
            "kernel": SpecialistOpinion(
                "MoE expert activation creates memory spikes. With 256 experts and "
                "dynamic routing, peak memory is unpredictable. Need to lower "
                "gpu_memory_utilization to leave headroom.", 0.91, True
            ),
            "loader": SpecialistOpinion(
                "gpu_memory_utilization=0.95 leaves no headroom. With MoE models, "
                "activation memory varies per-request depending on expert routing. "
                "Reduce to 0.85 and set max_num_seqs to 128.", 0.93, True
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[vLLM] Per-GPU memory: 80GB total, 76GB allocated (95%)\n"
                "[vLLM] Model weights: 28GB per GPU\n"
                "[vLLM] KV cache: 45GB per GPU\n"
                "[vLLM] Remaining: 3GB (insufficient for MoE activation spikes)\n"
                "[vLLM] Peak activation for DeepSeek MoE: ~5GB per GPU"
            ),
            config=(
                "gpu_memory_utilization: 0.95\n"
                "max_num_seqs: 256\n"
                "max_model_len: 32768\n"
                "swap_space_gb: 0\n"
                "tensor_parallel_size: 8\n"
                "model_weights_per_gpu_gb: 28\n"
                "kv_cache_per_gpu_gb: 45"
            ),
            snippet=(
                "# Memory budget too tight for MoE model\n"
                "# Model weights: 28GB + KV cache: 45GB = 73GB\n"
                "# Remaining: 3GB for activations\n"
                "# MoE expert routing can spike to 5GB+ activation\n"
                "# Fix: gpu_memory_utilization=0.85, max_num_seqs=128"
            ),
            metrics=(
                "oom_crashes: 14 in 1 hour\n"
                "avg_concurrent_at_crash: 52\n"
                "peak_activation_gb: 5.2\n"
                "available_at_crash_gb: 0.8\n"
                "gpu_memory_utilization: 0.95"
            ),
        ),
        specialist_followups={
            "runtime": "CUDA runtime is fine. The OOM is caused by overcommitted memory planning.",
            "dispatch": "Backend is dispatching correctly. Memory budget is the issue.",
            "kernel": "MoE activation spikes are the trigger. Reduce gpu_memory_utilization to 0.85.",
            "loader": "Lower gpu_memory_utilization to 0.85 and cap max_num_seqs at 128 for MoE headroom.",
        },
    ))

    scenarios.append(Scenario(
        id="memory_oom_02",
        root_cause="memory_oom",
        correct_fix="tune_memory_config",
        incident_ticket=(
            "INCIDENT: SGLang server on B200 crashes with OOM during long-context requests. "
            "Short prompts (<4K tokens) work fine. Any request over 16K tokens causes crash. "
            "Serving Llama-4-Maverick with 128 experts."
        ),
        hardware="NVIDIA B200",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] Model loaded: Llama-4-Maverick-17Bx128E\n"
            "[SGLang] KV cache pre-allocated for max_total_tokens=65536\n"
            "[SGLang] Request (len=18432): allocating KV blocks...\n"
            "[SGLang] FATAL: torch.cuda.OutOfMemoryError during KV cache expansion\n"
            "[SGLang] GPU memory: 189GB used / 192GB total"
        ),
        initial_snippet=(
            "# sglang_config.yaml\n"
            "max_total_tokens: 65536\n"
            "chunked_prefill_size: 8192\n"
            "mem_fraction_static: 0.90\n"
            "# KV cache over-allocated for 128-expert MoE model\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime healthy. Memory fragmentation from large contiguous allocations.", 0.72, False
            ),
            "dispatch": SpecialistOpinion(
                "Chunked prefill is working but chunk size may be too large for this context length.", 0.55, False
            ),
            "kernel": SpecialistOpinion(
                "KV cache expansion tries to allocate contiguous memory. "
                "With 128 experts, the KV cache per-token is much larger than dense models. "
                "max_total_tokens needs to be reduced for MoE.", 0.90, True
            ),
            "loader": SpecialistOpinion(
                "mem_fraction_static=0.90 is too high for MoE. With 128 experts, "
                "the shared expert and routing tensors need dynamic memory. "
                "Lower to 0.80 and reduce max_total_tokens to 32768.", 0.94, True
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[SGLang] Per-token KV size: 2.4MB (128 experts x shared KV)\n"
                "[SGLang] 65536 tokens * 2.4MB = 157GB KV cache requested\n"
                "[SGLang] Model weights: 35GB\n"
                "[SGLang] Total needed: 192GB (exceeds 192GB GPU memory)\n"
                "[SGLang] OOM at KV block allocation for long sequences"
            ),
            config=(
                "max_total_tokens: 65536\n"
                "mem_fraction_static: 0.90\n"
                "kv_per_token_mb: 2.4\n"
                "model_weights_gb: 35\n"
                "gpu_memory_gb: 192"
            ),
            snippet=(
                "# MoE KV cache is much larger than dense model estimate\n"
                "# Dense model: ~0.5MB per token KV\n"
                "# 128-expert MoE: ~2.4MB per token KV (shared + expert KV)\n"
                "# Fix: max_total_tokens=32768, mem_fraction_static=0.80"
            ),
            metrics=(
                "oom_count: 23\n"
                "avg_sequence_len_at_oom: 17500\n"
                "max_successful_len: 15200\n"
                "kv_cache_gb_at_crash: 157\n"
                "available_gb: 192"
            ),
        ),
        specialist_followups={
            "runtime": "Not a runtime issue. Memory planning doesn't account for MoE KV overhead.",
            "dispatch": "Chunked prefill helps but doesn't solve the KV cache over-allocation.",
            "kernel": "KV cache per-token is 4.8x larger than estimated. Reduce max_total_tokens.",
            "loader": "Set max_total_tokens=32768 and mem_fraction_static=0.80 for MoE headroom.",
        },
    ))

    scenarios.append(Scenario(
        id="memory_oom_03",
        root_cause="memory_oom",
        correct_fix="tune_memory_config",
        incident_ticket=(
            "INCIDENT: TensorRT-LLM engine build OOM on RTX 5090 for Qwen3-235B. "
            "Engine build phase exhausts 32GB VRAM during weight conversion. "
            "Same model builds fine on 80GB A100."
        ),
        hardware="NVIDIA SM120 (GeForce RTX 5090)",
        model_name="Qwen3-235B-A22B",
        backend="TensorRT-LLM 0.18",
        initial_log=(
            "[TensorRT-LLM] Building engine for Qwen3-235B-A22B...\n"
            "[TensorRT-LLM] Weight conversion phase: loading FP16 weights into GPU memory\n"
            "[TensorRT-LLM] ERROR: torch.cuda.OutOfMemoryError during weight quantization\n"
            "[TensorRT-LLM] Attempted to allocate 28GB for layer conversion buffer\n"
            "[TensorRT-LLM] GPU memory: 31.5GB used / 32GB total"
        ),
        initial_snippet=(
            "# trtllm_build_config.py\n"
            "build_config = {\n"
            "    'max_batch_size': 64,\n"
            "    'max_input_len': 8192,\n"
            "    'weight_streaming': False,  # loads all weights to GPU\n"
            "    'use_paged_context_fmha': True,\n"
            "}\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA 13 runtime is fine. GPU memory is just 32GB — too small for full model load.", 0.82, True
            ),
            "dispatch": SpecialistOpinion(
                "TensorRT-LLM engine build shouldn't need all weights on GPU at once. "
                "Enable weight_streaming to convert layer-by-layer.", 0.91, True
            ),
            "kernel": SpecialistOpinion(
                "Maybe try a different quantization algorithm that uses less memory.", 0.45, False
            ),
            "loader": SpecialistOpinion(
                "The model is too large for this GPU. Need a bigger GPU.", 0.40, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[TensorRT-LLM] Total FP16 weight size: 470GB\n"
                "[TensorRT-LLM] Per-layer max weight: 3.2GB\n"
                "[TensorRT-LLM] Conversion buffer: 28GB (loading 9 layers at once)\n"
                "[TensorRT-LLM] weight_streaming: disabled -> all layers loaded to GPU\n"
                "[TensorRT-LLM] GPU VRAM: 32GB"
            ),
            config=(
                "weight_streaming: false\n"
                "layers_loaded_simultaneously: 9\n"
                "per_layer_weight_gb: 3.2\n"
                "conversion_buffer_gb: 28\n"
                "gpu_vram_gb: 32"
            ),
            snippet=(
                "# weight_streaming=False loads multiple layers to GPU simultaneously\n"
                "# 9 layers * 3.2GB = 28GB conversion buffer\n"
                "# + engine workspace + CUDA context = >32GB\n"
                "# Fix: enable weight_streaming for layer-by-layer conversion"
            ),
            metrics=(
                "build_oom_count: 1\n"
                "conversion_buffer_gb: 28\n"
                "gpu_vram_gb: 32\n"
                "layers_loaded: 9\n"
                "max_layers_for_32gb: 3"
            ),
        ),
        specialist_followups={
            "runtime": "32GB VRAM is sufficient if weight streaming is enabled for layer-by-layer conversion.",
            "dispatch": "Enable weight_streaming=True so the builder converts one layer at a time instead of loading 9.",
            "kernel": "Quantization algorithm is fine. The issue is the build-time memory strategy.",
            "loader": "The model can serve on 32GB with streaming. It's the build phase that's misconfigured.",
        },
    ))

    scenarios.append(Scenario(
        id="memory_oom_04",
        root_cause="memory_oom",
        correct_fix="tune_memory_config",
        incident_ticket=(
            "INCIDENT: MI355X serving Llama-3.3-70B runs out of memory when beam search "
            "is enabled (num_beams=8). Greedy decoding works fine. "
            "OOM occurs during the first beam expansion step."
        ),
        hardware="AMD MI355X",
        model_name="Llama-3.3-70B-Instruct",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Serving Llama-3.3-70B-Instruct on MI355X\n"
            "[vLLM] Greedy decoding: OK (memory stable at 88%)\n"
            "[vLLM] Beam search (num_beams=8): allocating beam KV caches...\n"
            "[vLLM] ERROR: OOM during beam expansion. Each beam duplicates KV cache.\n"
            "[vLLM] Attempted: 8 * 14GB = 112GB KV for single request"
        ),
        initial_snippet=(
            "# vllm serve params\n"
            "--gpu-memory-utilization 0.92\n"
            "--max-num-seqs 64\n"
            "--max-model-len 16384\n"
            "# Beam search multiplies KV cache by num_beams\n"
            "# No per-request memory limit configured\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm runtime healthy. OOM is a memory allocation issue.", 0.75, False
            ),
            "dispatch": SpecialistOpinion(
                "Beam search KV duplication is the root cause. With 8 beams, each beam "
                "needs its own KV cache copy. Need to limit beam count or reserve memory.", 0.92, True
            ),
            "kernel": SpecialistOpinion(
                "The beam search kernel should share KV prefix across beams using copy-on-write. "
                "But vLLM's beam search implementation does full copies. This is a known issue.", 0.88, True
            ),
            "loader": SpecialistOpinion(
                "Maybe the model weights are in the wrong dtype causing extra memory.", 0.40, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[vLLM] Greedy: 1 KV cache per request = 14GB\n"
                "[vLLM] Beam search: 8 KV caches per request = 112GB\n"
                "[vLLM] GPU memory: 128GB (MI355X)\n"
                "[vLLM] Model weights: 35GB\n"
                "[vLLM] Available for KV: 83GB (insufficient for 112GB beam KV)"
            ),
            config=(
                "gpu_memory_utilization: 0.92\n"
                "max_num_seqs: 64\n"
                "num_beams: 8\n"
                "kv_per_request_gb: 14\n"
                "gpu_total_gb: 128"
            ),
            snippet=(
                "# Beam search duplicates full KV cache per beam\n"
                "# 8 beams * 14GB = 112GB for single request\n"
                "# Available after model weights: 83GB\n"
                "# Fix: reduce num_beams to 4, lower max_model_len, or use\n"
                "# gpu_memory_utilization=0.85 with max_num_seqs=8 for beam mode"
            ),
            metrics=(
                "oom_on_beam_search: true\n"
                "kv_per_beam_gb: 14\n"
                "total_beam_kv_gb: 112\n"
                "available_gb: 83\n"
                "greedy_oom: false"
            ),
        ),
        specialist_followups={
            "runtime": "ROCm runtime is fine. Memory budget doesn't account for beam duplication.",
            "dispatch": "Limit beams to 4 and reduce max_model_len to 8192 for beam search mode.",
            "kernel": "vLLM beam search copies full KV per beam. Use fewer beams or enable prefix sharing.",
            "loader": "Model weights are correct dtype. The issue is beam search KV duplication.",
        },
    ))

    scenarios.append(Scenario(
        id="memory_oom_05",
        root_cause="memory_oom",
        correct_fix="tune_memory_config",
        incident_ticket=(
            "INCIDENT: Triton Inference Server on DGX Spark OOMs when loading second model. "
            "First model (DeepSeek-R1-Distill-70B) loads fine. Loading Mistral-Large-2 "
            "concurrently causes OOM. Both models fit individually but not together."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="DeepSeek-R1-Distill-70B",
        backend="Triton Inference Server",
        initial_log=(
            "[Triton] Model 1 (DeepSeek-R1-Distill-70B): loaded, 65GB VRAM\n"
            "[Triton] Model 2 (Mistral-Large-2): loading...\n"
            "[Triton] ERROR: CUDA OOM allocating 48GB for model 2\n"
            "[Triton] GPU memory: 91GB used / 96GB total\n"
            "[Triton] Model instance groups both set to GPU 0"
        ),
        initial_snippet=(
            "# model_repository/config.pbtxt (both models)\n"
            "instance_group [\n"
            "  { count: 1, kind: KIND_GPU, gpus: [0] }\n"
            "]\n"
            "# Both models pinned to GPU 0 (96GB)\n"
            "# No model loading policy configured\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime fine. Both models are pinned to GPU 0 instead of being spread "
                "across available GPUs.", 0.87, True
            ),
            "dispatch": SpecialistOpinion(
                "Triton model placement is wrong. Both models on GPU 0 but 96GB isn't enough "
                "for both. Need to set model_loading_policy or spread across GPUs.", 0.92, True
            ),
            "kernel": SpecialistOpinion(
                "Maybe one of the models has a memory leak in its kernels.", 0.35, False
            ),
            "loader": SpecialistOpinion(
                "Both models are loaded eagerly at startup. Try lazy loading with "
                "model_loading_policy=on_demand.", 0.60, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[Triton] GPU 0: 65GB (model 1) + 5GB (CUDA ctx) = 70GB used\n"
                "[Triton] GPU 0: 96GB total, 26GB free\n"
                "[Triton] Model 2 needs 48GB — doesn't fit on GPU 0\n"
                "[Triton] GPU 1: 96GB free (unused!)\n"
                "[Triton] Both models pinned to gpus: [0]"
            ),
            config=(
                "model_1_gpu: [0]\n"
                "model_2_gpu: [0]\n"
                "available_gpus: [0, 1, 2, 3]\n"
                "model_1_vram_gb: 65\n"
                "model_2_vram_gb: 48\n"
                "gpu_0_total_gb: 96"
            ),
            snippet=(
                "# Both models pinned to GPU 0 via instance_group config\n"
                "# GPU 0: 96GB (insufficient for 65+48=113GB)\n"
                "# GPUs 1-3: completely idle\n"
                "# Fix: assign model 2 to GPU 1, or use auto-placement"
            ),
            metrics=(
                "gpu_0_used_gb: 70\n"
                "gpu_0_free_gb: 26\n"
                "gpu_1_used_gb: 0\n"
                "model_2_needed_gb: 48\n"
                "total_gpus: 4"
            ),
        ),
        specialist_followups={
            "runtime": "GPU 1-3 are idle. Assign model 2 to GPU 1 in instance_group config.",
            "dispatch": "Fix the instance_group config to spread models across GPUs. Or use auto-placement.",
            "kernel": "No memory leak. Both models just don't fit on one 96GB GPU.",
            "loader": "Lazy loading doesn't help — both models are needed concurrently. Spread across GPUs.",
        },
    ))

    scenarios.append(Scenario(
        id="memory_oom_06",
        root_cause="memory_oom",
        correct_fix="tune_memory_config",
        incident_ticket=(
            "INCIDENT: vLLM on MI300X crashes with OOM when prefix caching is enabled. "
            "Without prefix caching, Qwen3-235B serves fine. With prefix caching, "
            "memory grows unbounded until OOM after ~200 requests."
        ),
        hardware="AMD MI300X",
        model_name="Qwen3-235B-A22B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Prefix caching: enabled\n"
            "[vLLM] Initial GPU memory usage: 142GB / 192GB\n"
            "[vLLM] After 50 requests: 165GB / 192GB (prefix cache growing)\n"
            "[vLLM] After 150 requests: 188GB / 192GB\n"
            "[vLLM] Request 203: OOM — prefix cache consumed all free memory"
        ),
        initial_snippet=(
            "# vllm serve\n"
            "--enable-prefix-caching\n"
            "--gpu-memory-utilization 0.90\n"
            "--max-num-seqs 32\n"
            "# No prefix cache eviction policy\n"
            "# No max prefix cache size configured\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm runtime is fine. The prefix cache is never evicted and grows forever.", 0.83, True
            ),
            "dispatch": SpecialistOpinion(
                "Try disabling prefix caching entirely. It doesn't work with MoE models.", 0.40, False
            ),
            "kernel": SpecialistOpinion(
                "Prefix cache blocks are never freed. vLLM's evictor isn't respecting the "
                "memory budget. Need to set max_prefix_cache_tokens or lower gpu_memory_utilization.", 0.92, True
            ),
            "loader": SpecialistOpinion(
                "Model loaded correctly. Memory issue is in the KV cache subsystem.", 0.70, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[vLLM] Prefix cache entries: 847 (never evicted)\n"
                "[vLLM] Prefix cache size: 46GB\n"
                "[vLLM] Cache eviction policy: NONE (default)\n"
                "[vLLM] gpu_memory_utilization=0.90 doesn't cap prefix cache\n"
                "[vLLM] OOM when active KV + prefix cache > available memory"
            ),
            config=(
                "enable_prefix_caching: true\n"
                "prefix_cache_eviction: none\n"
                "max_prefix_cache_tokens: unlimited\n"
                "gpu_memory_utilization: 0.90\n"
                "prefix_cache_gb: 46"
            ),
            snippet=(
                "# Prefix cache grows without bound\n"
                "# gpu_memory_utilization only limits initial KV cache allocation\n"
                "# Prefix cache is separate and has no eviction\n"
                "# Fix: set gpu_memory_utilization=0.80 and enable LRU eviction"
            ),
            metrics=(
                "prefix_cache_entries: 847\n"
                "prefix_cache_gb: 46\n"
                "time_to_oom_minutes: 22\n"
                "requests_before_oom: 203\n"
                "eviction_events: 0"
            ),
        ),
        specialist_followups={
            "runtime": "Prefix cache grows without eviction. Set a size limit or enable LRU eviction.",
            "dispatch": "Prefix caching works fine with MoE. The issue is unbounded cache growth.",
            "kernel": "Set gpu_memory_utilization=0.80 and configure prefix cache eviction policy to LRU.",
            "loader": "Weights loaded fine. Tune the prefix cache memory budget.",
        },
    ))

    # --- quantization_error scenarios ---
    scenarios.append(Scenario(
        id="quantization_error_01",
        root_cause="quantization_error",
        correct_fix="fix_quantization",
        incident_ticket=(
            "INCIDENT: GPTQ 4-bit quantized Llama-3.3-70B produces garbled output on H100. "
            "Perplexity is 450+ (expected <10). The quantization was done with an older "
            "AutoGPTQ version. Suspected calibration data mismatch."
        ),
        hardware="NVIDIA H100",
        model_name="Llama-3.3-70B-Instruct",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Loading GPTQ model: 4-bit, group_size=128\n"
            "[vLLM] Quantization config: bits=4, sym=True, desc_act=True\n"
            "[vLLM] WARNING: qweight shape unexpected for layer 0: [4096, 1024]\n"
            "[vLLM] Output sample: 'the the the the the the...'\n"
            "[vLLM] Perplexity check: 458.7 (FAIL, threshold: 15)"
        ),
        initial_snippet=(
            "# quantize.py (AutoGPTQ v0.4)\n"
            "quantize_config = BaseQuantizeConfig(\n"
            "    bits=4, group_size=128, desc_act=True,\n"
            "    sym=True, true_sequential=True\n"
            ")\n"
            "# Calibration dataset: 128 samples of random tokens\n"
            "# Should use representative text from target domain\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime is fine. Model loads and runs. Output quality is the issue.", 0.78, False
            ),
            "dispatch": SpecialistOpinion(
                "GPTQ dequantization kernel dispatched correctly for H100.", 0.65, False
            ),
            "kernel": SpecialistOpinion(
                "The dequantization kernel is correct. qweight tensor packing format changed "
                "between AutoGPTQ v0.4 and v0.7. Old format is being read with new unpacker.", 0.93, True
            ),
            "loader": SpecialistOpinion(
                "GPTQ checkpoint was quantized with AutoGPTQ v0.4 which uses a different "
                "packing order (column-major) than vLLM expects (row-major). "
                "Re-quantize with compatible version or convert packing format.", 0.91, True
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[GPTQ] qweight packing: column-major (AutoGPTQ v0.4)\n"
                "[GPTQ] Expected packing: row-major (vLLM/AutoGPTQ v0.7+)\n"
                "[GPTQ] Bit unpacking yields wrong int4 values\n"
                "[GPTQ] Layer 0 dequantized weights: mean=47.3, std=812 (expected mean~0, std~0.02)"
            ),
            config=(
                "autogptq_version: 0.4.2\n"
                "vllm_expected_version: 0.7+\n"
                "packing_format: column_major\n"
                "expected_packing: row_major\n"
                "bits: 4\n"
                "group_size: 128"
            ),
            snippet=(
                "# AutoGPTQ v0.4: packs 8 int4 values column-major into int32\n"
                "# AutoGPTQ v0.7+: packs row-major (matches Marlin/Exllama kernels)\n"
                "# vLLM's Marlin kernel expects row-major packing\n"
                "# Fix: re-quantize with AutoGPTQ>=0.7 or use repack script"
            ),
            metrics=(
                "perplexity: 458.7\n"
                "expected_perplexity: 8.2\n"
                "dequant_mean_error: 47.3\n"
                "dequant_std_error: 812\n"
                "affected_layers: all"
            ),
        ),
        specialist_followups={
            "runtime": "Runtime is fine. Quantized weights are just unpacked incorrectly.",
            "dispatch": "Marlin kernel dispatched. But input data is in wrong packing format.",
            "kernel": "Marlin kernel unpacks row-major. Checkpoint is column-major. Re-quantize or repack.",
            "loader": "Re-quantize with AutoGPTQ >= 0.7 to get row-major packing compatible with vLLM.",
        },
    ))

    scenarios.append(Scenario(
        id="quantization_error_02",
        root_cause="quantization_error",
        correct_fix="fix_quantization",
        incident_ticket=(
            "INCIDENT: AWQ 4-bit Mistral-Large-2 on RTX 5090 has quality degradation "
            "only in the last 20 layers. First 40 layers produce correct activations. "
            "Full-precision model works perfectly. Suspected quantization calibration issue."
        ),
        hardware="NVIDIA SM120 (GeForce RTX 5090)",
        model_name="Mistral-Large-2",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] AWQ model loaded: 4-bit, group_size=128\n"
            "[vLLM] Layer 0-39: activations normal (cosine sim > 0.99 vs FP16)\n"
            "[vLLM] Layer 40+: activations diverging (cosine sim < 0.3 vs FP16)\n"
            "[vLLM] Output: partially coherent, degrades mid-sentence\n"
            "[vLLM] Perplexity: 87.3 (expected: 9.1)"
        ),
        initial_snippet=(
            "# awq_calibration.py\n"
            "calib_data = load_dataset('wikitext', split='train[:128]')\n"
            "# Calibration only runs 128 samples with max_length=512\n"
            "# Mistral-Large-2 has 60 layers\n"
            "# Short calibration doesn't capture activation ranges in deep layers\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA 13 runtime is fine. No errors during inference.", 0.75, False
            ),
            "dispatch": SpecialistOpinion(
                "AWQ backend selected and dispatching correctly.", 0.60, False
            ),
            "kernel": SpecialistOpinion(
                "The AWQ dequantization kernels are mathematically correct. "
                "The issue is the quantization scales themselves — they were computed from "
                "insufficient calibration data. Deep layers got poor scale estimates.", 0.90, True
            ),
            "loader": SpecialistOpinion(
                "AWQ calibration used only 128 samples with max_length=512. "
                "Deep layers (40-59) never saw diverse activation ranges. "
                "Re-quantize with 512+ samples at max_length=2048.", 0.94, True
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[AWQ] Calibration stats: 128 samples, max_len=512\n"
                "[AWQ] Layer 0-39 scale variance: 0.012 (good)\n"
                "[AWQ] Layer 40-59 scale variance: 0.0001 (flat — under-calibrated)\n"
                "[AWQ] Deep layer scales almost identical — poor quantization"
            ),
            config=(
                "awq_bits: 4\n"
                "group_size: 128\n"
                "calib_samples: 128\n"
                "calib_max_length: 512\n"
                "model_layers: 60\n"
                "under_calibrated_layers: 40-59"
            ),
            snippet=(
                "# AWQ scales for layers 40-59 are nearly flat\n"
                "# Calibration data too short to activate deep layers meaningfully\n"
                "# Fix: re-quantize with 512+ samples, max_length=2048\n"
                "# This ensures all layers see representative activations"
            ),
            metrics=(
                "perplexity: 87.3\n"
                "expected_perplexity: 9.1\n"
                "cosine_sim_layer0_39: 0.993\n"
                "cosine_sim_layer40_59: 0.27\n"
                "under_calibrated_layers: 20"
            ),
        ),
        specialist_followups={
            "runtime": "No runtime issue. Quantization quality is the problem.",
            "dispatch": "AWQ backend is correct. Scale values are the issue.",
            "kernel": "Dequant kernel is correct. The scales for deep layers are flat. Re-calibrate.",
            "loader": "Re-quantize with more calibration data (512 samples, max_length=2048) to fix deep layers.",
        },
    ))

    scenarios.append(Scenario(
        id="quantization_error_03",
        root_cause="quantization_error",
        correct_fix="fix_quantization",
        incident_ticket=(
            "INCIDENT: FP8 quantized DeepSeek-V3-671B on B200 produces NaN in MoE expert "
            "layers. Dense layers work fine. Suspected FP8 scale overflow in expert weights "
            "due to outlier channels in MoE FFN."
        ),
        hardware="NVIDIA B200",
        model_name="DeepSeek-V3-671B",
        backend="TensorRT-LLM 0.18",
        initial_log=(
            "[TensorRT-LLM] FP8 inference: DeepSeek-V3-671B\n"
            "[TensorRT-LLM] Dense layers 0-3: output OK\n"
            "[TensorRT-LLM] MoE layer 4, expert 37: NaN detected\n"
            "[TensorRT-LLM] FP8 dequant scale for expert 37: 1847.5 (extreme)\n"
            "[TensorRT-LLM] Overflow during dequantization: scale * FP8_max > FP16_max"
        ),
        initial_snippet=(
            "# fp8_quantize.py\n"
            "# Per-tensor FP8 quantization\n"
            "scale = weight.abs().max() / FP8_E4M3_MAX\n"
            "# MoE expert FFN layers have outlier channels with |w| > 500\n"
            "# Per-tensor scale too coarse for these layers\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime fine. NaN propagation from dequantization.", 0.78, False
            ),
            "dispatch": SpecialistOpinion(
                "FP8 backend dispatched correctly. The quantization scheme is the issue.", 0.70, False
            ),
            "kernel": SpecialistOpinion(
                "FP8 GEMM kernel produces NaN because dequant scale overflows FP16 range. "
                "Expert 37 has outlier weight channels that make per-tensor scaling too coarse. "
                "Need per-channel FP8 scaling for MoE experts.", 0.94, True
            ),
            "loader": SpecialistOpinion(
                "MoE expert weights have extreme outliers in a few channels. "
                "Per-tensor FP8 quantization can't handle this — scale is dominated by outliers "
                "while other channels get crushed to zero. Switch to per-channel quantization.", 0.91, True
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[FP8] Expert 37 weight stats: max=547.2, mean=0.03, outlier_ratio=0.001\n"
                "[FP8] Per-tensor scale: 547.2 / 448.0 = 1.22\n"
                "[FP8] Dequant: 448.0 * 1.22 = 546.6 (close to FP16 overflow with accumulation)\n"
                "[FP8] Non-outlier channels quantized to 0 (lost information)"
            ),
            config=(
                "fp8_format: e4m3fn\n"
                "quantization_scheme: per_tensor\n"
                "expert_37_weight_max: 547.2\n"
                "fp8_e4m3_max: 448.0\n"
                "fp16_max: 65504.0"
            ),
            snippet=(
                "# Per-tensor scale for expert 37: max(|w|)=547.2\n"
                "# FP8 max representable: 448.0\n"
                "# Scale = 547.2/448 = 1.22\n"
                "# Dequant of max values: 448*1.22 = 546.6\n"
                "# Accumulated in FP16 GEMM -> overflow with large sequences\n"
                "# Fix: use per-channel FP8 scaling for MoE expert layers"
            ),
            metrics=(
                "nan_experts: [37, 91, 203]\n"
                "total_experts: 256\n"
                "outlier_ratio: 0.001\n"
                "per_tensor_scale_max: 1847.5\n"
                "non_outlier_precision_loss: 95%"
            ),
        ),
        specialist_followups={
            "runtime": "NaN is from dequant overflow, not runtime issue.",
            "dispatch": "FP8 dispatch is correct. Quantization granularity needs to change.",
            "kernel": "Switch to per-channel FP8 scaling for MoE expert FFN layers to handle outliers.",
            "loader": "Re-quantize MoE experts with per-channel scaling. Dense layers can stay per-tensor.",
        },
    ))

    scenarios.append(Scenario(
        id="quantization_error_04",
        root_cause="quantization_error",
        correct_fix="fix_quantization",
        incident_ticket=(
            "INCIDENT: SmoothQuant INT8 on H100 for Qwen3-235B has severe accuracy loss. "
            "Perplexity 130+ vs expected 8.5. SmoothQuant migration factor alpha=0.5 "
            "was tuned for a different model architecture."
        ),
        hardware="NVIDIA H100",
        model_name="Qwen3-235B-A22B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] SmoothQuant INT8: alpha=0.5\n"
            "[vLLM] Activation smoothing applied...\n"
            "[vLLM] WARNING: post-smooth activation range still large: [-127, 127] maps to [-847, 923]\n"
            "[vLLM] INT8 GEMM output quality: FAIL\n"
            "[vLLM] Perplexity: 134.2 (expected: 8.5)"
        ),
        initial_snippet=(
            "# smooth_quant_config.yaml\n"
            "quantization: smoothquant\n"
            "smoothquant_alpha: 0.5\n"
            "# alpha=0.5 was tuned for Llama-2-70B\n"
            "# Qwen3 MoE has different activation distribution\n"
            "# Needs per-layer alpha tuning for MoE gating layers\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "Runtime is fine. INT8 GEMM kernels launch correctly.", 0.75, False
            ),
            "dispatch": SpecialistOpinion(
                "INT8 backend dispatched. But the quantization configuration is wrong for this model.", 0.68, True
            ),
            "kernel": SpecialistOpinion(
                "The INT8 GEMM kernel is correct. SmoothQuant alpha=0.5 doesn't properly balance "
                "the activation/weight difficulty for Qwen3's MoE architecture. "
                "MoE gating layers need alpha=0.75+.", 0.93, True
            ),
            "loader": SpecialistOpinion(
                "Weights loaded fine. Try increasing batch size to improve throughput.", 0.35, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[SmoothQuant] alpha=0.5 (global)\n"
                "[SmoothQuant] Dense layer activation range after smooth: [-23, 31] (OK)\n"
                "[SmoothQuant] MoE gating layer activation range after smooth: [-847, 923] (BAD)\n"
                "[SmoothQuant] MoE gating layers need higher alpha to push more to weights"
            ),
            config=(
                "smoothquant_alpha: 0.5\n"
                "optimal_alpha_dense: 0.5\n"
                "optimal_alpha_moe_gate: 0.80\n"
                "optimal_alpha_moe_ffn: 0.70\n"
                "per_layer_alpha: false"
            ),
            snippet=(
                "# SmoothQuant: X_smooth = X / diag(s^alpha), W_smooth = W * diag(s^alpha)\n"
                "# alpha=0.5 works for dense Llama but not for Qwen3 MoE\n"
                "# MoE gating/FFN layers have spikier activations needing alpha=0.7-0.8\n"
                "# Fix: enable per-layer alpha tuning or use alpha=0.75 globally"
            ),
            metrics=(
                "perplexity: 134.2\n"
                "expected_perplexity: 8.5\n"
                "dense_layer_accuracy: 97%\n"
                "moe_layer_accuracy: 34%\n"
                "activation_clipping_rate: 42%"
            ),
        ),
        specialist_followups={
            "runtime": "No runtime issue. Quantization accuracy is the problem.",
            "dispatch": "INT8 dispatch is fine. The quantization parameters need adjustment.",
            "kernel": "Re-run SmoothQuant calibration with per-layer alpha or alpha=0.75 for Qwen3 MoE.",
            "loader": "Batch size is irrelevant. The issue is SmoothQuant alpha mismatch.",
        },
    ))

    scenarios.append(Scenario(
        id="quantization_error_05",
        root_cause="quantization_error",
        correct_fix="fix_quantization",
        incident_ticket=(
            "INCIDENT: GGUF Q4_K_M quantized DeepSeek-R1-Distill-70B fails to load in "
            "SGLang on MI355X. Error: 'unsupported quant type Q4_K_M for ROCm'. "
            "Same GGUF file works on NVIDIA hardware."
        ),
        hardware="AMD MI355X",
        model_name="DeepSeek-R1-Distill-70B",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] Loading GGUF: DeepSeek-R1-Distill-70B-Q4_K_M.gguf\n"
            "[SGLang] Detected quant type: Q4_K_M (k-quant mixed)\n"
            "[SGLang] ROCm backend: checking dequant kernel support...\n"
            "[SGLang] ERROR: Q4_K_M dequantization kernel not available for ROCm/HIP\n"
            "[SGLang] Supported ROCm quant types: Q4_0, Q8_0, FP16"
        ),
        initial_snippet=(
            "# sglang/quantization/gguf_loader.py\n"
            "ROCM_SUPPORTED_QUANTS = {'Q4_0', 'Q8_0', 'FP16'}\n"
            "# K-quant types (Q4_K_M, Q5_K_M, etc.) have CUDA kernels only\n"
            "# ROCm/HIP dequant kernels not yet implemented for k-quants\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm runtime loads fine. The dequantization kernel is missing for this quant type.", 0.82, True
            ),
            "dispatch": SpecialistOpinion(
                "Dispatch correctly identifies the quant type but has no ROCm kernel to call.", 0.80, True
            ),
            "kernel": SpecialistOpinion(
                "K-quant dequantization uses CUDA-specific warp shuffle instructions. "
                "Need to re-quantize to Q4_0 or Q8_0 for ROCm, or convert to a supported format.", 0.92, True
            ),
            "loader": SpecialistOpinion(
                "The GGUF file is corrupted. Try re-downloading.", 0.30, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[SGLang] GGUF header: Q4_K_M, 70B params, 80 layers\n"
                "[SGLang] ROCm dequant kernel registry: Q4_0, Q8_0, FP16\n"
                "[SGLang] Q4_K_M not in registry\n"
                "[SGLang] K-quant kernels use __shfl_xor_sync (CUDA only)"
            ),
            config=(
                "gguf_quant_type: Q4_K_M\n"
                "rocm_supported: [Q4_0, Q8_0, FP16]\n"
                "k_quant_rocm_support: false\n"
                "gpu_vendor: AMD"
            ),
            snippet=(
                "# Q4_K_M uses k-quant mixed precision (4-bit with 6-bit super-blocks)\n"
                "# CUDA kernel uses __shfl_xor_sync for dequant\n"
                "# No HIP equivalent implemented in SGLang\n"
                "# Fix: re-quantize model to Q8_0 or Q4_0 for ROCm\n"
                "# Or: convert GGUF to safetensors with AWQ/GPTQ quantization"
            ),
            metrics=(
                "load_failures: 1\n"
                "quant_type: Q4_K_M\n"
                "rocm_kernel_available: false\n"
                "nvidia_kernel_available: true"
            ),
        ),
        specialist_followups={
            "runtime": "ROCm works. K-quant dequant kernels are CUDA-only in SGLang.",
            "dispatch": "Dispatch is correct in rejecting unsupported quant type. Need compatible quant format.",
            "kernel": "Re-quantize to Q4_0 or Q8_0 which have HIP/ROCm kernels. Or use AWQ instead of GGUF.",
            "loader": "File is fine. The quant format just isn't supported on ROCm.",
        },
    ))

    scenarios.append(Scenario(
        id="quantization_error_06",
        root_cause="quantization_error",
        correct_fix="fix_quantization",
        incident_ticket=(
            "INCIDENT: Marlin INT4 kernel on DGX Spark gives wrong results for "
            "Llama-4-Maverick MoE model. Dense layers quantize fine but expert FFN "
            "weights have incorrect permutation after Marlin repacking."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Marlin INT4 kernel loading...\n"
            "[vLLM] Repacking GPTQ weights to Marlin format...\n"
            "[vLLM] Dense layers: repack OK (verified against reference)\n"
            "[vLLM] Expert 0 FFN: repack MISMATCH (output differs from GPTQ reference)\n"
            "[vLLM] 73 of 128 experts have repack errors"
        ),
        initial_snippet=(
            "# vllm/quantization/marlin_repack.py\n"
            "def repack_gptq_to_marlin(qweight, scales, g_idx, num_experts=None):\n"
            "    # BUG: expert FFN weight shape is [E, N, K//8]\n"
            "    # Repacker assumes [N, K//8] (no expert dim)\n"
            "    perm = get_marlin_permutation(qweight.shape[-2], qweight.shape[-1])\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime fine. Marlin kernel launches. Output is wrong.", 0.75, False
            ),
            "dispatch": SpecialistOpinion(
                "Marlin kernel dispatched. But the weight repacking has a shape handling bug.", 0.72, True
            ),
            "kernel": SpecialistOpinion(
                "Marlin repacker doesn't handle the 3D expert weight tensor [E, N, K//8]. "
                "It treats the expert dimension as part of N, creating wrong permutation. "
                "Need to repack each expert slice independently.", 0.95, True
            ),
            "loader": SpecialistOpinion(
                "The model has too many experts. Try reducing the number of active experts.", 0.30, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[Marlin] Expert weight shape: [128, 4096, 512]\n"
                "[Marlin] Repacker expected: [4096, 512] per expert\n"
                "[Marlin] Repacker got: [128*4096, 512] (flattened expert dim into N)\n"
                "[Marlin] Permutation computed for 524288 rows instead of 4096 rows"
            ),
            config=(
                "num_experts: 128\n"
                "expert_ffn_shape: [128, 4096, 512]\n"
                "expected_per_expert: [4096, 512]\n"
                "marlin_repack_dim: 2D_only\n"
                "repack_bug: expert_dim_flattened"
            ),
            snippet=(
                "# Marlin repacker assumes 2D weight matrix [N, K//8]\n"
                "# MoE expert weights are 3D [E, N, K//8]\n"
                "# Repacker flattens to [E*N, K//8] -> wrong permutation\n"
                "# Fix: iterate over expert dim and repack each [N, K//8] slice"
            ),
            metrics=(
                "experts_with_repack_error: 73\n"
                "total_experts: 128\n"
                "dense_layers_correct: true\n"
                "output_cosine_sim: 0.12\n"
                "expected_cosine_sim: 0.99"
            ),
        ),
        specialist_followups={
            "runtime": "Runtime fine. Weight repacking is wrong for 3D expert tensors.",
            "dispatch": "Marlin dispatch is OK. Fix the repack function to handle expert dimension.",
            "kernel": "Repack each expert slice [N, K//8] independently. Don't flatten the expert dim.",
            "loader": "Number of experts is fine. The Marlin repacker has a shape bug for MoE weights.",
        },
    ))

    # --- distributed_comm scenarios ---
    scenarios.append(Scenario(
        id="distributed_comm_01",
        root_cause="distributed_comm",
        correct_fix="fix_comm_config",
        incident_ticket=(
            "INCIDENT: 8-GPU tensor parallel DeepSeek-V3-671B hangs during all-reduce on H100. "
            "Model loads, first forward pass completes, second forward pass hangs indefinitely. "
            "NCCL timeout after 300 seconds. Single-GPU works fine."
        ),
        hardware="NVIDIA H100",
        model_name="DeepSeek-V3-671B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Tensor parallel size: 8\n"
            "[vLLM] NCCL version: 2.21.5\n"
            "[vLLM] Forward pass 1: OK (12.3s)\n"
            "[vLLM] Forward pass 2: all-reduce HANG at layer 3\n"
            "[NCCL] WARN Timeout after 300000ms on rank 5"
        ),
        initial_snippet=(
            "# Environment\n"
            "NCCL_P2P_DISABLE=0\n"
            "NCCL_IB_DISABLE=0\n"
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
            "# GPUs span two NVSwitch domains (0-3 and 4-7)\n"
            "# NCCL topology detection may be incorrect\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime is fine on all 8 GPUs individually. The hang is in NCCL collective.", 0.82, True
            ),
            "dispatch": SpecialistOpinion(
                "Backend dispatches correctly. Tensor parallel scatter/gather is the issue.", 0.70, False
            ),
            "kernel": SpecialistOpinion(
                "NCCL all-reduce hangs because rank 5 (GPU 5) is using a different communicator "
                "than ranks 0-4. Two NVSwitch domains need NCCL_CROSS_NIC=1 and proper "
                "NCCL_SOCKET_IFNAME configuration.", 0.93, True
            ),
            "loader": SpecialistOpinion(
                "Weights sharded correctly across 8 GPUs. Not a loading issue.", 0.65, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[NCCL] Topology: 2 NVSwitch domains [0-3] [4-7]\n"
                "[NCCL] Ring 0: 0->1->2->3->4 (crosses domain boundary)\n"
                "[NCCL] Rank 5 timeout: peer 4 not responding on NVLink\n"
                "[NCCL] NVLink 4->5: bandwidth 0 (link not established across domains)\n"
                "[NCCL] NCCL_CROSS_NIC not set, defaults to 0"
            ),
            config=(
                "tensor_parallel_size: 8\n"
                "nccl_version: 2.21.5\n"
                "nvswitch_domains: 2\n"
                "domain_0_gpus: [0,1,2,3]\n"
                "domain_1_gpus: [4,5,6,7]\n"
                "nccl_cross_nic: 0"
            ),
            snippet=(
                "# 8 GPUs span 2 NVSwitch domains\n"
                "# NCCL ring tries to route 4->5 via NVLink but no direct link\n"
                "# Cross-domain traffic needs to go via PCIe/InfiniBand\n"
                "# Fix: set NCCL_CROSS_NIC=1, NCCL_NET_GDR_LEVEL=5"
            ),
            metrics=(
                "successful_allreduce: 1\n"
                "hung_allreduce: 1\n"
                "timeout_rank: 5\n"
                "nvlink_4_5_bandwidth: 0\n"
                "nccl_timeout_ms: 300000"
            ),
        ),
        specialist_followups={
            "runtime": "CUDA fine on each GPU. NCCL cross-domain communication is the issue.",
            "dispatch": "Tensor parallel dispatch is correct. NCCL topology config is wrong.",
            "kernel": "Set NCCL_CROSS_NIC=1 and NCCL_NET_GDR_LEVEL=5 for cross-NVSwitch-domain communication.",
            "loader": "Weights sharded correctly. Communication config is the problem.",
        },
    ))

    scenarios.append(Scenario(
        id="distributed_comm_02",
        root_cause="distributed_comm",
        correct_fix="fix_comm_config",
        incident_ticket=(
            "INCIDENT: Pipeline parallel Qwen3-235B on 4xMI300X has 50% lower throughput "
            "than expected. Bubble overhead is 60% instead of expected 15%. "
            "Pipeline stages are severely imbalanced."
        ),
        hardware="AMD MI300X",
        model_name="Qwen3-235B-A22B",
        backend="SGLang 0.5.x",
        initial_log=(
            "[SGLang] Pipeline parallel: 4 stages\n"
            "[SGLang] Stage 0 (layers 0-14): avg 45ms\n"
            "[SGLang] Stage 1 (layers 15-29): avg 42ms\n"
            "[SGLang] Stage 2 (layers 30-44): avg 180ms (MoE layers!)\n"
            "[SGLang] Stage 3 (layers 45-59): avg 175ms\n"
            "[SGLang] Pipeline bubble: 60% (stages 0-1 idle waiting for 2-3)"
        ),
        initial_snippet=(
            "# sglang_config.yaml\n"
            "pipeline_parallel_size: 4\n"
            "# Default: equal layer split (15 layers per stage)\n"
            "# Qwen3 MoE: layers 30-59 have MoE blocks (4x slower)\n"
            "# Uniform split creates severe load imbalance\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm runtime healthy on all 4 GPUs. No communication errors.", 0.75, False
            ),
            "dispatch": SpecialistOpinion(
                "Pipeline parallel stages have severe load imbalance. MoE layers in stages 2-3 "
                "are 4x slower than dense layers in stages 0-1. Need non-uniform stage partitioning.", 0.94, True
            ),
            "kernel": SpecialistOpinion(
                "MoE expert dispatch takes longer due to all-to-all communication within each "
                "pipeline stage. Consider tensor parallel for MoE layers instead.", 0.55, False
            ),
            "loader": SpecialistOpinion(
                "Pipeline split should put fewer MoE layers per stage. "
                "Recommended: 20 layers in stages 0-1, 10 layers in stages 2-3.", 0.90, True
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[SGLang] Stage timing (ms): [45, 42, 180, 175]\n"
                "[SGLang] Slowest stage: 180ms (4x faster stages)\n"
                "[SGLang] Pipeline bubble: 135ms per micro-batch\n"
                "[SGLang] MoE layers: 30-59 (all in stages 2-3)\n"
                "[SGLang] Dense layers: 0-29 (all in stages 0-1)"
            ),
            config=(
                "pipeline_parallel_size: 4\n"
                "layers_per_stage: [15, 15, 15, 15]\n"
                "moe_layers: [30, 31, ..., 59]\n"
                "stage_0_1_type: dense\n"
                "stage_2_3_type: moe"
            ),
            snippet=(
                "# Uniform split: [0-14, 15-29, 30-44, 45-59]\n"
                "# Stages 0-1: dense only (fast)\n"
                "# Stages 2-3: MoE only (slow)\n"
                "# Fix: rebalance to [0-19, 20-39, 40-49, 50-59]\n"
                "# Or: [0-24, 25-44, 45-54, 55-59] for more even timing"
            ),
            metrics=(
                "pipeline_bubble_pct: 60\n"
                "expected_bubble_pct: 15\n"
                "throughput_tokens_per_sec: 850\n"
                "expected_throughput: 1700\n"
                "stage_imbalance_ratio: 4.3x"
            ),
        ),
        specialist_followups={
            "runtime": "No communication errors. Throughput issue from pipeline imbalance.",
            "dispatch": "Rebalance pipeline stages: put more dense layers per stage, fewer MoE layers per stage.",
            "kernel": "MoE layers are just slower. The fix is pipeline partitioning, not kernel changes.",
            "loader": "Re-partition: stages 0-1 get 20 layers each, stages 2-3 get 10 layers each.",
        },
    ))

    scenarios.append(Scenario(
        id="distributed_comm_03",
        root_cause="distributed_comm",
        correct_fix="fix_comm_config",
        incident_ticket=(
            "INCIDENT: vLLM tensor parallel on 2xRTX 5090 via PCIe has 90% communication "
            "overhead. All-reduce takes 85ms but compute only takes 9ms per layer. "
            "Expected NVLink but system has PCIe only."
        ),
        hardware="NVIDIA SM120 (GeForce RTX 5090)",
        model_name="Llama-3.3-70B-Instruct",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Tensor parallel size: 2\n"
            "[vLLM] NCCL init: using PCIe transport (no NVLink detected)\n"
            "[vLLM] All-reduce per layer: 85ms (PCIe 5.0 x16)\n"
            "[vLLM] Compute per layer: 9ms\n"
            "[vLLM] Communication overhead: 90.4%\n"
            "[vLLM] Total latency per token: 7520ms (expected: ~800ms)"
        ),
        initial_snippet=(
            "# system config\n"
            "# 2x RTX 5090 in PCIe slots (no NVLink bridge)\n"
            "# PCIe 5.0 x16: ~64 GB/s bidirectional\n"
            "# NVLink 5: ~900 GB/s (not available)\n"
            "tensor_parallel_size: 2\n"
            "# All-reduce over PCIe is bottleneck\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime fine. PCIe link healthy but slow for TP all-reduce.", 0.80, True
            ),
            "dispatch": SpecialistOpinion(
                "Tensor parallel is wrong strategy for PCIe-only multi-GPU. "
                "Should use pipeline parallel instead — it only sends activations between stages "
                "(much less data than TP all-reduce). Or use single-GPU with quantization.", 0.94, True
            ),
            "kernel": SpecialistOpinion(
                "Maybe the NCCL kernel is using a suboptimal algorithm for PCIe.", 0.50, False
            ),
            "loader": SpecialistOpinion(
                "Model weights split correctly across 2 GPUs. Loading is fine.", 0.65, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[NCCL] Transport: PCIe 5.0 x16\n"
                "[NCCL] Bandwidth: 32 GB/s per direction\n"
                "[NCCL] All-reduce data per layer: 2.7 GB\n"
                "[NCCL] All-reduce time: 2.7GB / 32GB/s = 84ms\n"
                "[NCCL] 80 layers * 84ms = 6720ms communication per token"
            ),
            config=(
                "transport: pcie_5_x16\n"
                "bandwidth_per_dir_gbs: 32\n"
                "allreduce_per_layer_gb: 2.7\n"
                "nvlink_available: false\n"
                "tensor_parallel_size: 2"
            ),
            snippet=(
                "# TP all-reduce sends hidden_size * dtype_bytes per layer\n"
                "# 8192 * 2 * batch * seq_len = ~2.7GB per layer\n"
                "# PCIe: 2.7GB / 32GB/s = 84ms per layer\n"
                "# Fix: switch to pipeline parallel (sends only activations)\n"
                "# Or: use single-GPU with INT4 quantization (fits in 32GB)"
            ),
            metrics=(
                "allreduce_ms_per_layer: 85\n"
                "compute_ms_per_layer: 9\n"
                "comm_overhead_pct: 90.4\n"
                "total_latency_ms: 7520\n"
                "expected_with_nvlink_ms: 800"
            ),
        ),
        specialist_followups={
            "runtime": "PCIe link is healthy but bandwidth-limited for TP. Switch parallel strategy.",
            "dispatch": "Switch from tensor parallel to pipeline parallel, or use single-GPU with INT4 quantization.",
            "kernel": "NCCL algorithm is optimal for PCIe. The interconnect is just too slow for TP.",
            "loader": "Weights loaded fine. The parallel strategy needs to change.",
        },
    ))

    scenarios.append(Scenario(
        id="distributed_comm_04",
        root_cause="distributed_comm",
        correct_fix="fix_comm_config",
        incident_ticket=(
            "INCIDENT: 4-node distributed vLLM serving DeepSeek-V3-671B has intermittent "
            "NCCL errors. Every ~100 requests, one node reports 'unhandled system error' "
            "in all-gather. Network MTU mismatch between nodes suspected."
        ),
        hardware="NVIDIA B200",
        model_name="DeepSeek-V3-671B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Distributed: 4 nodes, 8 GPUs each (32 total)\n"
            "[NCCL] Init: IB transport, 4 HCAs per node\n"
            "[vLLM] Request 87: NCCL error on node 2\n"
            "[NCCL] node2: unhandled system error (NCCL_SYSTEM_ERROR)\n"
            "[NCCL] Suspected: IB message size exceeds MTU on switch"
        ),
        initial_snippet=(
            "# Network config\n"
            "# Nodes 0,1,3: MTU=4096 (IB)\n"
            "# Node 2: MTU=2048 (different switch port config)\n"
            "# NCCL_IB_GID_INDEX=3\n"
            "# No explicit NCCL_IB_TC or NCCL_IB_TIMEOUT set\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime healthy on all nodes. NCCL error is network-level.", 0.82, True
            ),
            "dispatch": SpecialistOpinion(
                "The error seems random but always involves node 2. Check node 2's IB config.", 0.75, False
            ),
            "kernel": SpecialistOpinion(
                "NCCL uses IB RDMA with MTU-sized messages. Node 2 has MTU=2048 while others "
                "have 4096. When NCCL sends a 4096-byte message to node 2, the IB switch drops it. "
                "Fix: align MTU to 4096 on node 2's switch port.", 0.94, True
            ),
            "loader": SpecialistOpinion(
                "Model weights distributed correctly across 32 GPUs. Not a loading issue.", 0.65, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[NCCL] Node 0 IB MTU: 4096\n"
                "[NCCL] Node 1 IB MTU: 4096\n"
                "[NCCL] Node 2 IB MTU: 2048 (MISMATCH)\n"
                "[NCCL] Node 3 IB MTU: 4096\n"
                "[NCCL] Messages >2048 to node 2: DROPPED\n"
                "[NCCL] Error frequency: ~1 per 100 requests (when large all-gather hits node 2)"
            ),
            config=(
                "nodes: 4\n"
                "gpus_per_node: 8\n"
                "ib_mtu_node0: 4096\n"
                "ib_mtu_node1: 4096\n"
                "ib_mtu_node2: 2048\n"
                "ib_mtu_node3: 4096\n"
                "nccl_ib_timeout: default"
            ),
            snippet=(
                "# Node 2 IB switch port configured with MTU=2048\n"
                "# Other nodes: MTU=4096\n"
                "# NCCL sends 4096-byte IB messages -> dropped at node 2's switch\n"
                "# Fix: reconfigure node 2's switch port to MTU=4096\n"
                "# Workaround: NCCL_IB_TC=128, NCCL_IB_TIMEOUT=22"
            ),
            metrics=(
                "nccl_errors: 47 in 1 hour\n"
                "errors_on_node2: 47\n"
                "errors_on_other_nodes: 0\n"
                "avg_requests_between_errors: 103\n"
                "ib_mtu_mismatch: true"
            ),
        ),
        specialist_followups={
            "runtime": "CUDA runtime is fine. NCCL errors from IB MTU mismatch on node 2.",
            "dispatch": "Node 2 is the problem. IB config needs to match other nodes.",
            "kernel": "Set node 2 IB MTU to 4096 to match the cluster. Or set NCCL_IB_TC=128 as workaround.",
            "loader": "Weight distribution is correct. Network MTU mismatch is the root cause.",
        },
    ))

    scenarios.append(Scenario(
        id="distributed_comm_05",
        root_cause="distributed_comm",
        correct_fix="fix_comm_config",
        incident_ticket=(
            "INCIDENT: Expert parallel DeepSeek-V3-671B on 8xMI355X has all-to-all "
            "communication deadlock. Token routing to remote experts hangs when "
            "more than 4 GPUs are involved. 4-GPU EP works fine."
        ),
        hardware="AMD MI355X",
        model_name="DeepSeek-V3-671B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] Expert parallel: 8 GPUs, 32 experts per GPU\n"
            "[RCCL] All-to-all init: 8 ranks\n"
            "[vLLM] MoE token routing: sending tokens to remote experts...\n"
            "[RCCL] HANG: all-to-all stuck at rank 6 (waiting for rank 2)\n"
            "[RCCL] Timeout after 600s. 4-GPU EP (ranks 0-3) works fine."
        ),
        initial_snippet=(
            "# RCCL config\n"
            "RCCL_ALLTOALL_KERNEL=1  # basic kernel\n"
            "# 8 MI355X connected via xGMI + Infinity Fabric\n"
            "# xGMI: 4-GPU pod (GPUs 0-3) and (GPUs 4-7)\n"
            "# Cross-pod: Infinity Fabric bridge\n"
            "# RCCL kernel 1 doesn't handle cross-pod routing\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm runtime healthy. RCCL all-to-all kernel selection is wrong for 8-GPU topology.", 0.85, True
            ),
            "dispatch": SpecialistOpinion(
                "Expert parallel dispatch routes tokens correctly. The communication primitive is the issue.", 0.72, False
            ),
            "kernel": SpecialistOpinion(
                "RCCL all-to-all kernel 1 assumes fully connected topology. "
                "8 MI355X has 2 xGMI pods bridged by Infinity Fabric. "
                "Need kernel 2 (hierarchical) for cross-pod communication.", 0.93, True
            ),
            "loader": SpecialistOpinion(
                "Expert weights distributed correctly. All-to-all comm is the bottleneck.", 0.68, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[RCCL] Topology: 2 xGMI pods [0-3] [4-7]\n"
                "[RCCL] Intra-pod bandwidth: 896 GB/s\n"
                "[RCCL] Cross-pod bandwidth: 128 GB/s (Infinity Fabric)\n"
                "[RCCL] All-to-all kernel 1: flat ring (assumes uniform links)\n"
                "[RCCL] Ring 0->1->2->3->4 blocked at 3->4 (cross-pod)\n"
                "[RCCL] Deadlock: rank 6 waits for rank 2 via pod bridge"
            ),
            config=(
                "expert_parallel_size: 8\n"
                "rccl_alltoall_kernel: 1\n"
                "topology: 2_xgmi_pods\n"
                "intra_pod_bw_gbs: 896\n"
                "cross_pod_bw_gbs: 128"
            ),
            snippet=(
                "# RCCL kernel 1: flat ring all-to-all\n"
                "# Assumes all links have equal bandwidth\n"
                "# Cross-pod link (128 GB/s) vs intra-pod (896 GB/s)\n"
                "# Flat ring deadlocks when cross-pod link saturates\n"
                "# Fix: RCCL_ALLTOALL_KERNEL=2 (hierarchical)\n"
                "# Or: set RCCL_NCHANNELS=16 for cross-pod"
            ),
            metrics=(
                "deadlock_count: 12 in 1 hour\n"
                "4gpu_ep_deadlocks: 0\n"
                "8gpu_ep_deadlocks: 12\n"
                "cross_pod_saturation_pct: 100\n"
                "intra_pod_utilization_pct: 15"
            ),
        ),
        specialist_followups={
            "runtime": "ROCm fine. RCCL kernel selection needs to account for 2-pod topology.",
            "dispatch": "Expert routing is correct. RCCL transport is the issue.",
            "kernel": "Set RCCL_ALLTOALL_KERNEL=2 for hierarchical all-to-all across xGMI pods.",
            "loader": "Expert weight distribution is correct. Fix RCCL kernel config.",
        },
    ))

    scenarios.append(Scenario(
        id="distributed_comm_06",
        root_cause="distributed_comm",
        correct_fix="fix_comm_config",
        incident_ticket=(
            "INCIDENT: TensorRT-LLM multi-node inference on 2x DGX Spark (16 GPUs total) "
            "has 40% throughput drop compared to single-node 8-GPU. "
            "Inter-node all-reduce over RoCE is bottlenecked. TCP fallback detected."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="Mistral-Large-2",
        backend="TensorRT-LLM 0.18",
        initial_log=(
            "[TensorRT-LLM] 2 nodes, 8 GPUs each, TP=16\n"
            "[NCCL] Transport: NET/Socket (TCP fallback!)\n"
            "[NCCL] WARNING: RDMA not available, using TCP\n"
            "[NCCL] Inter-node all-reduce: 340ms (expected: 12ms with RDMA)\n"
            "[TensorRT-LLM] Throughput: 420 tok/s (expected: 700 tok/s)"
        ),
        initial_snippet=(
            "# Network config\n"
            "# RoCE v2 NICs installed but not configured\n"
            "# NCCL_SOCKET_IFNAME=eth0 (TCP fallback)\n"
            "# RoCE NIC: mlx5_0 (not specified in NCCL config)\n"
            "# GDR (GPU Direct RDMA) not enabled\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA runtime fine. Network transport is the bottleneck.", 0.78, False
            ),
            "dispatch": SpecialistOpinion(
                "NCCL fell back to TCP sockets instead of RDMA over RoCE. "
                "Need to configure NCCL_SOCKET_IFNAME to point to the RoCE NIC "
                "and enable GPU Direct RDMA.", 0.94, True
            ),
            "kernel": SpecialistOpinion(
                "TensorRT-LLM kernels are fine. Inter-node communication is bottlenecked by TCP.", 0.82, True
            ),
            "loader": SpecialistOpinion(
                "Maybe the model is too large for 16-way tensor parallel. Try pipeline parallel.", 0.40, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[NCCL] NCCL_SOCKET_IFNAME=eth0 (1Gbps TCP)\n"
                "[NCCL] mlx5_0: RoCE v2, 400Gbps (UNCONFIGURED)\n"
                "[NCCL] GPU Direct RDMA: disabled\n"
                "[NCCL] TCP all-reduce: 340ms per layer\n"
                "[NCCL] Expected RDMA all-reduce: 12ms per layer"
            ),
            config=(
                "nccl_socket_ifname: eth0\n"
                "eth0_speed: 1Gbps\n"
                "mlx5_0_speed: 400Gbps\n"
                "rdma_enabled: false\n"
                "gdr_enabled: false"
            ),
            snippet=(
                "# NCCL using TCP over 1Gbps eth0 instead of 400Gbps RoCE\n"
                "# RoCE NIC (mlx5_0) available but not configured\n"
                "# Fix: NCCL_SOCKET_IFNAME=mlx5_0\n"
                "# Fix: NCCL_NET_GDR_LEVEL=5 (enable GPU Direct RDMA)\n"
                "# Fix: NCCL_IB_DISABLE=0"
            ),
            metrics=(
                "throughput_tok_s: 420\n"
                "expected_throughput: 700\n"
                "tcp_allreduce_ms: 340\n"
                "rdma_allreduce_ms: 12\n"
                "inter_node_bandwidth_gbps: 1"
            ),
        ),
        specialist_followups={
            "runtime": "Network stack works. Just using wrong NIC for NCCL.",
            "dispatch": "Set NCCL_SOCKET_IFNAME=mlx5_0 and enable GDR for 400Gbps RDMA transport.",
            "kernel": "Compute kernels are fine. Inter-node latency will drop 28x with RDMA.",
            "loader": "Model size is fine for TP=16. The network config is the problem.",
        },
    ))

    # --- driver_compat scenarios ---
    scenarios.append(Scenario(
        id="driver_compat_01",
        root_cause="driver_compat",
        correct_fix="update_driver_config",
        incident_ticket=(
            "INCIDENT: DGX Spark (SM121) with CUDA 12.4 driver fails to run FlashInfer 0.4. "
            "Error: 'CUDA driver version is insufficient for CUDA runtime version'. "
            "FlashInfer compiled against CUDA 13.0 but driver only supports up to 12.4."
        ),
        hardware="NVIDIA SM121 (DGX Spark)",
        model_name="DeepSeek-V3-671B",
        backend="FlashInfer 0.4",
        initial_log=(
            "[FlashInfer] Loading CUDA kernels...\n"
            "[CUDA] Driver version: 12040 (12.4)\n"
            "[CUDA] Runtime version: 13000 (13.0)\n"
            "[FlashInfer] ERROR: CUDA driver version is insufficient for CUDA runtime version\n"
            "[FlashInfer] Required: CUDA driver >= 13.0, found: 12.4"
        ),
        initial_snippet=(
            "# System info\n"
            "nvidia-smi: Driver Version: 550.54.15 (CUDA 12.4)\n"
            "# FlashInfer 0.4 was compiled with CUDA 13.0 toolkit\n"
            "# SM121 GPUs require CUDA 13.0+ driver\n"
            "# Driver 550.x only supports up to CUDA 12.4\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA driver 550.x (12.4) cannot load CUDA 13.0 compiled kernels. "
                "Need to upgrade to driver 570+ which supports CUDA 13.0.", 0.95, True
            ),
            "dispatch": SpecialistOpinion(
                "FlashInfer dispatch cannot even start. This is a driver-level incompatibility.", 0.80, True
            ),
            "kernel": SpecialistOpinion(
                "The FlashInfer kernel PTX targets SM121 which needs CUDA 13.0 JIT compilation. "
                "Driver 550.x cannot JIT compile for SM121.", 0.85, True
            ),
            "loader": SpecialistOpinion(
                "Try downgrading FlashInfer to a version compiled with CUDA 12.4.", 0.40, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[CUDA] Driver: 550.54.15 (supports up to CUDA 12.4)\n"
                "[CUDA] Runtime: 13.0 (required by FlashInfer 0.4)\n"
                "[CUDA] SM121 JIT compilation requires driver with CUDA 13.0 support\n"
                "[CUDA] cuModuleLoadData failed: CUDA_ERROR_NO_BINARY_FOR_GPU"
            ),
            config=(
                "driver_version: 550.54.15\n"
                "max_cuda_version: 12.4\n"
                "required_cuda_version: 13.0\n"
                "gpu_arch: sm_121\n"
                "flashinfer_cuda_version: 13.0"
            ),
            snippet=(
                "# Driver 550.x: CUDA 12.4 max\n"
                "# FlashInfer 0.4: compiled with CUDA 13.0\n"
                "# SM121 PTX needs CUDA 13.0 JIT\n"
                "# Fix: upgrade driver to 570+ (CUDA 13.0 support)"
            ),
            metrics=(
                "driver_cuda_max: 12.4\n"
                "required_cuda: 13.0\n"
                "kernel_load_failures: 1\n"
                "jit_compilation: failed"
            ),
        ),
        specialist_followups={
            "runtime": "Upgrade NVIDIA driver to 570+ for CUDA 13.0 support.",
            "dispatch": "Can't dispatch until driver supports CUDA 13.0.",
            "kernel": "SM121 kernels need CUDA 13.0 JIT. Driver 550.x can't do it.",
            "loader": "Downgrading FlashInfer won't help — SM121 requires CUDA 13.0 regardless.",
        },
    ))

    scenarios.append(Scenario(
        id="driver_compat_02",
        root_cause="driver_compat",
        correct_fix="update_driver_config",
        incident_ticket=(
            "INCIDENT: ROCm 6.3 driver on MI300X silently produces wrong FP8 results. "
            "Model outputs have subtle accuracy degradation (perplexity 14 vs expected 9). "
            "ROCm 6.3.0 has known bug in FP8 MFMA instruction for certain matrix sizes."
        ),
        hardware="AMD MI300X",
        model_name="DeepSeek-R1-Distill-70B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] ROCm version: 6.3.0\n"
            "[vLLM] FP8 inference: e4m3fnuz format\n"
            "[vLLM] Output quality: marginal (perplexity 14.2, threshold 10)\n"
            "[vLLM] BF16 reference: perplexity 8.9 (correct)\n"
            "[vLLM] FP8 accuracy gap: 5.3 perplexity points (abnormal)"
        ),
        initial_snippet=(
            "# ROCm 6.3.0 release notes (known issues)\n"
            "# BUG: FP8 MFMA instruction produces incorrect results\n"
            "#   when M=16, N=16, K=32 and matrix A is transposed\n"
            "#   Affects: MI300X, MI355X\n"
            "#   Fixed in: ROCm 6.3.1\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm 6.3.0 has a known FP8 MFMA bug. The driver produces silently wrong results "
                "for specific matrix dimensions. Need ROCm 6.3.1 hotfix.", 0.95, True
            ),
            "dispatch": SpecialistOpinion(
                "FP8 backend selected correctly. The hardware instruction has a bug.", 0.70, False
            ),
            "kernel": SpecialistOpinion(
                "The FP8 MFMA kernel triggers a known ROCm 6.3.0 driver bug for M16N16K32 "
                "with transposed A. This isn't a software kernel issue — it's the hardware "
                "microcode. Update to ROCm 6.3.1.", 0.92, True
            ),
            "loader": SpecialistOpinion(
                "Weights loaded correctly. BF16 works fine, only FP8 is affected.", 0.68, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[ROCm] Version: 6.3.0 (known FP8 MFMA bug)\n"
                "[FP8] MFMA M16N16K32 with transA: incorrect accumulation\n"
                "[FP8] Affected layers: 23 of 80 (those using transposed A)\n"
                "[FP8] BF16 MFMA: correct (no bug in BF16 path)\n"
                "[ROCm] Fix available: ROCm 6.3.1"
            ),
            config=(
                "rocm_version: 6.3.0\n"
                "fp8_mfma_bug: true\n"
                "affected_config: M16N16K32_transA\n"
                "fix_version: 6.3.1\n"
                "bf16_affected: false"
            ),
            snippet=(
                "# ROCm 6.3.0 FP8 MFMA bug:\n"
                "# mfma_f32_16x16x32_fp8 with transposed A matrix\n"
                "# Accumulator gets wrong partial sums\n"
                "# Silent data corruption (no error, just wrong results)\n"
                "# Fix: upgrade to ROCm 6.3.1 or use BF16 as workaround"
            ),
            metrics=(
                "fp8_perplexity: 14.2\n"
                "bf16_perplexity: 8.9\n"
                "accuracy_gap: 5.3\n"
                "affected_layers: 23\n"
                "total_layers: 80"
            ),
        ),
        specialist_followups={
            "runtime": "Upgrade to ROCm 6.3.1 which fixes the FP8 MFMA accumulator bug.",
            "dispatch": "Dispatch is correct. The hardware instruction is buggy.",
            "kernel": "Not a kernel bug — it's ROCm 6.3.0 microcode. Update to 6.3.1.",
            "loader": "Weights are fine. Only FP8 MFMA is affected.",
        },
    ))

    scenarios.append(Scenario(
        id="driver_compat_03",
        root_cause="driver_compat",
        correct_fix="update_driver_config",
        incident_ticket=(
            "INCIDENT: RTX 5090 (SM120) fails TensorRT-LLM engine build. Error: "
            "'No registered converter for SM120'. TensorRT-LLM 0.18 was built against "
            "CUDA 12.8 but SM120 needs CUDA 13.0 compiler support."
        ),
        hardware="NVIDIA SM120 (GeForce RTX 5090)",
        model_name="Llama-3.3-70B-Instruct",
        backend="TensorRT-LLM 0.18",
        initial_log=(
            "[TensorRT-LLM] Building engine for SM120...\n"
            "[TensorRT] CUDA toolkit: 12.8\n"
            "[TensorRT] ERROR: No code generator for sm_120\n"
            "[TensorRT] sm_120 requires CUDA toolkit >= 13.0\n"
            "[TensorRT] Available targets: sm_70, sm_75, sm_80, sm_86, sm_89, sm_90"
        ),
        initial_snippet=(
            "# TensorRT-LLM build environment\n"
            "cuda_toolkit: 12.8\n"
            "tensorrt_version: 10.4\n"
            "# SM120 code generation requires CUDA 13.0+\n"
            "# TensorRT-LLM 0.18 was compiled with CUDA 12.8\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "CUDA toolkit 12.8 cannot generate code for SM120. Need CUDA 13.0+ toolkit "
                "and matching TensorRT version.", 0.94, True
            ),
            "dispatch": SpecialistOpinion(
                "TensorRT engine can't be built. Need newer CUDA toolkit.", 0.82, True
            ),
            "kernel": SpecialistOpinion(
                "Try using compute_90 as a fallback target for SM120.", 0.45, False
            ),
            "loader": SpecialistOpinion(
                "The model format is fine. It's the build toolchain that's outdated.", 0.70, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[TensorRT] CUDA toolkit 12.8: sm_120 not in target list\n"
                "[TensorRT] sm_120 added in CUDA 13.0\n"
                "[TensorRT] TensorRT 10.4 built with CUDA 12.8\n"
                "[TensorRT] Need TensorRT 10.6+ with CUDA 13.0"
            ),
            config=(
                "cuda_toolkit: 12.8\n"
                "tensorrt: 10.4\n"
                "sm120_support_min_cuda: 13.0\n"
                "sm120_support_min_trt: 10.6\n"
                "available_sm_targets: [70,75,80,86,89,90]"
            ),
            snippet=(
                "# CUDA 12.8 code generator has no SM120 target\n"
                "# SM120 introduced in CUDA 13.0 toolkit\n"
                "# TensorRT 10.6+ bundles CUDA 13.0 codegen\n"
                "# Fix: upgrade to TensorRT-LLM with TRT 10.6 / CUDA 13.0"
            ),
            metrics=(
                "engine_build_failures: 1\n"
                "sm120_codegen: unavailable\n"
                "cuda_toolkit: 12.8\n"
                "required_cuda_toolkit: 13.0"
            ),
        ),
        specialist_followups={
            "runtime": "Upgrade CUDA toolkit to 13.0 and TensorRT to 10.6+ for SM120 support.",
            "dispatch": "Engine can't be built without SM120 codegen. Upgrade toolchain.",
            "kernel": "compute_90 fallback won't work — SM120 has different register file layout.",
            "loader": "Model is fine. Toolchain needs updating for SM120.",
        },
    ))

    scenarios.append(Scenario(
        id="driver_compat_04",
        root_cause="driver_compat",
        correct_fix="update_driver_config",
        incident_ticket=(
            "INCIDENT: NVIDIA driver 570.86 on B200 causes random GPU resets during "
            "long inference sequences (>8K tokens). System log shows 'Xid 79: GPU has "
            "fallen off the bus'. Known driver bug in 570.86 for B200."
        ),
        hardware="NVIDIA B200",
        model_name="Llama-4-Maverick-17Bx128E",
        backend="vLLM 0.8.x",
        initial_log=(
            "[System] Xid 79: GPU 3 has fallen off the bus\n"
            "[vLLM] ERROR: CUDA error on GPU 3: device-side assert triggered\n"
            "[vLLM] Inference failed at token position 8247\n"
            "[System] nvidia-smi: GPU 3: ERR! (needs reset)\n"
            "[System] Driver: 570.86.01"
        ),
        initial_snippet=(
            "# NVIDIA driver 570.86 release notes:\n"
            "# Known issue: B200 GPUs may experience Xid 79 errors\n"
            "#   during sustained high-power workloads\n"
            "#   Affected: long-running inference >8K tokens\n"
            "#   Fixed in: 570.100+\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "Driver 570.86 has a known bug for B200 that causes GPU bus resets "
                "during sustained high-power inference. Upgrade to 570.100+.", 0.95, True
            ),
            "dispatch": SpecialistOpinion(
                "Cannot fix at the application level. This is a driver bug.", 0.75, False
            ),
            "kernel": SpecialistOpinion(
                "Xid 79 is a hardware-level bus error. The GPU physically disconnects from PCIe. "
                "This is a driver power management bug, not a kernel issue.", 0.88, True
            ),
            "loader": SpecialistOpinion(
                "Try reducing GPU power limit to prevent thermal issues.", 0.35, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[System] dmesg: NVRM: Xid 79 at PCI:0000:81:00.0\n"
                "[System] Driver: 570.86.01\n"
                "[System] GPU 3 power at crash: 680W (TDP: 700W)\n"
                "[System] Crash occurs at token position: 8000-10000\n"
                "[NVIDIA] Known bug: 570.86 B200 power sequencing error"
            ),
            config=(
                "driver_version: 570.86.01\n"
                "gpu: B200\n"
                "known_bug: xid_79_b200\n"
                "fix_version: 570.100\n"
                "crash_power_watts: 680\n"
                "tdp_watts: 700"
            ),
            snippet=(
                "# Driver 570.86 B200 power management bug:\n"
                "# Sustained near-TDP workloads trigger power sequencing error\n"
                "# GPU drops off PCIe bus (Xid 79)\n"
                "# Fix: upgrade to driver 570.100+\n"
                "# Workaround: nvidia-smi -pl 600 (reduce power limit)"
            ),
            metrics=(
                "xid_79_events: 7 in 24 hours\n"
                "avg_tokens_at_crash: 8743\n"
                "gpu_power_at_crash_w: 680\n"
                "driver_version: 570.86\n"
                "fix_version: 570.100"
            ),
        ),
        specialist_followups={
            "runtime": "Upgrade driver to 570.100+ to fix the B200 power sequencing bug.",
            "dispatch": "Application-level fix won't help. Driver upgrade needed.",
            "kernel": "Xid 79 is driver-level. Upgrade to 570.100 or use power limit workaround.",
            "loader": "Reducing power limit is a workaround but the real fix is driver 570.100+.",
        },
    ))

    scenarios.append(Scenario(
        id="driver_compat_05",
        root_cause="driver_compat",
        correct_fix="update_driver_config",
        incident_ticket=(
            "INCIDENT: MI355X with ROCm 6.2 driver cannot use FlashAttention-2 for MI355X. "
            "Error: 'Unsupported AMDGPU target: gfx950'. MI355X (gfx950) was added in "
            "ROCm 6.4. System running old ROCm from base OS install."
        ),
        hardware="AMD MI355X",
        model_name="Qwen3-235B-A22B",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] GPU: AMD MI355X (gfx950)\n"
            "[ROCm] Version: 6.2.0\n"
            "[FlashAttention] Compiling for gfx950...\n"
            "[FlashAttention] ERROR: gfx950 not supported by ROCm 6.2 compiler\n"
            "[FlashAttention] Available targets: gfx900, gfx906, gfx908, gfx90a, gfx942"
        ),
        initial_snippet=(
            "# ROCm version check\n"
            "rocm_version: 6.2.0  # from base OS install\n"
            "# MI355X (gfx950) support added in ROCm 6.4\n"
            "# ROCm 6.2 compiler cannot generate gfx950 ISA\n"
            "# FlashAttention-2 kernel compilation fails\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "ROCm 6.2 does not support gfx950 (MI355X). Need ROCm 6.4+ for this GPU.", 0.95, True
            ),
            "dispatch": SpecialistOpinion(
                "FlashAttention cannot compile kernels for gfx950 on ROCm 6.2. "
                "This blocks all optimized attention paths.", 0.85, True
            ),
            "kernel": SpecialistOpinion(
                "Try compiling FlashAttention targeting gfx942 as a fallback.", 0.40, False
            ),
            "loader": SpecialistOpinion(
                "Model weights load fine. The kernel compilation is the issue.", 0.65, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[ROCm] Installed: 6.2.0\n"
                "[ROCm] gfx950 support: NOT AVAILABLE (added in 6.4)\n"
                "[hipcc] --offload-arch=gfx950: error: unknown target\n"
                "[FlashAttention] Cannot JIT compile attention kernels\n"
                "[vLLM] Falling back to naive attention (10x slower)"
            ),
            config=(
                "rocm_version: 6.2.0\n"
                "gpu_target: gfx950\n"
                "gfx950_min_rocm: 6.4\n"
                "available_targets: [gfx900,gfx906,gfx908,gfx90a,gfx942]\n"
                "flashattention_status: compilation_failed"
            ),
            snippet=(
                "# ROCm 6.2 compiler targets: gfx900-gfx942\n"
                "# gfx950 (MI355X) added in ROCm 6.4\n"
                "# No backward compatibility — gfx950 ISA differs from gfx942\n"
                "# Fix: upgrade ROCm to 6.4+"
            ),
            metrics=(
                "kernel_compilation_failures: 1\n"
                "attention_backend: naive_fallback\n"
                "attention_slowdown: 10x\n"
                "rocm_version: 6.2.0\n"
                "required_rocm: 6.4"
            ),
        ),
        specialist_followups={
            "runtime": "Upgrade ROCm to 6.4+ to get gfx950 compiler support.",
            "dispatch": "All optimized attention paths blocked. ROCm upgrade is required.",
            "kernel": "gfx942 fallback won't work — ISA is incompatible. Must use native gfx950 from ROCm 6.4.",
            "loader": "Model loads but runs on naive attention at 10x penalty. Fix is ROCm upgrade.",
        },
    ))

    scenarios.append(Scenario(
        id="driver_compat_06",
        root_cause="driver_compat",
        correct_fix="update_driver_config",
        incident_ticket=(
            "INCIDENT: H100 with driver 535 LTS cannot use FP8 tensor cores. "
            "FP8 GEMM falls back to FP16 emulation causing 3x slowdown. "
            "Driver 535 technically supports H100 but has incomplete FP8 support."
        ),
        hardware="NVIDIA H100",
        model_name="Mistral-Large-2",
        backend="vLLM 0.8.x",
        initial_log=(
            "[vLLM] GPU: H100 SXM (SM90)\n"
            "[CUDA] Driver: 535.183.01 (LTS)\n"
            "[vLLM] FP8 GEMM: attempting to use tensor cores...\n"
            "[cuBLAS] WARNING: FP8 GEMM not supported by driver, falling back to FP16 emulation\n"
            "[vLLM] Token generation: 45 tok/s (expected: 135 tok/s with native FP8)"
        ),
        initial_snippet=(
            "# Driver 535 (LTS) limitations:\n"
            "# - H100 SM90 basic support: YES\n"
            "# - FP8 tensor core (e4m3fn GEMM): INCOMPLETE\n"
            "# - cuBLAS FP8 API: stub only (returns CUBLAS_STATUS_NOT_SUPPORTED)\n"
            "# - Full FP8 requires driver 545+\n"
        ),
        specialist_opinions={
            "runtime": SpecialistOpinion(
                "Driver 535 LTS has incomplete FP8 cuBLAS support. Upgrade to 545+ for "
                "full FP8 tensor core acceleration on H100.", 0.94, True
            ),
            "dispatch": SpecialistOpinion(
                "vLLM correctly attempts FP8 but cuBLAS returns NOT_SUPPORTED. "
                "Fallback to FP16 emulation is working but slow.", 0.80, True
            ),
            "kernel": SpecialistOpinion(
                "Maybe the FP8 weights are in the wrong format for this cuBLAS version.", 0.45, False
            ),
            "loader": SpecialistOpinion(
                "FP8 model loaded fine. The slowdown is from driver-level FP8 emulation.", 0.72, False
            ),
        },
        inspect_results=InspectResult(
            logs=(
                "[cuBLAS] Version: 12.3 (driver 535)\n"
                "[cuBLAS] cublasLtMatmul with FP8: CUBLAS_STATUS_NOT_SUPPORTED\n"
                "[cuBLAS] Fallback: FP8 -> FP16 upcast -> FP16 GEMM -> FP8 downcast\n"
                "[cuBLAS] FP8 emulation overhead: 3.1x\n"
                "[cuBLAS] Native FP8 requires cuBLAS 12.5+ (driver 545+)"
            ),
            config=(
                "driver_version: 535.183.01\n"
                "cublas_version: 12.3\n"
                "fp8_native_support: false\n"
                "fp8_emulation: true\n"
                "min_driver_for_fp8: 545"
            ),
            snippet=(
                "# Driver 535 LTS: cuBLAS 12.3 stubs FP8 API\n"
                "# Returns NOT_SUPPORTED, triggers FP16 emulation path\n"
                "# FP8 -> FP16 upcast + FP16 GEMM + FP8 downcast = 3x slower\n"
                "# Fix: upgrade to driver 545+ for native FP8 cuBLAS"
            ),
            metrics=(
                "throughput_tok_s: 45\n"
                "expected_throughput: 135\n"
                "fp8_emulation_overhead: 3.1x\n"
                "cublas_fp8_status: NOT_SUPPORTED\n"
                "driver: 535.183.01"
            ),
        ),
        specialist_followups={
            "runtime": "Upgrade from driver 535 LTS to 545+ for native FP8 cuBLAS support.",
            "dispatch": "FP8 dispatch falls back to emulation. Driver upgrade will enable native path.",
            "kernel": "FP8 weight format is correct. cuBLAS just can't use FP8 tensor cores with driver 535.",
            "loader": "Weights are fine. Driver needs upgrading for FP8 acceleration.",
        },
    ))

    return scenarios


# Build the full scenario pool
_HANDCRAFTED_SCENARIOS = _make_scenarios()


# ---------------------------------------------------------------------------
# Load scraped scenarios from generated_scenarios_full.json
# ---------------------------------------------------------------------------

def load_scraped_scenarios(path: str) -> list[Scenario]:
    """Load scraped scenarios from a JSON file and convert to Scenario objects.

    Missing fields (inspect_results, specialist_followups) are synthesized
    from the available data so every Scenario is fully populated.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    scenarios: list[Scenario] = []
    for entry in raw:
        # --- specialist_opinions: dict-of-dicts -> dict-of-SpecialistOpinion ---
        specialist_opinions: dict[str, SpecialistOpinion] = {}
        for name, op_dict in entry.get("specialist_opinions", {}).items():
            specialist_opinions[name] = SpecialistOpinion(
                opinion=op_dict["opinion"],
                confidence=float(op_dict["confidence"]),
                is_correct=bool(op_dict["is_correct"]),
            )

        # --- inspect_results: synthesize from available data ---
        hardware = entry.get("hardware", "Unknown")
        backend = entry.get("backend", "Unknown")
        model_name = entry.get("model_name", "Unknown")
        initial_log = entry.get("initial_log", "")
        initial_snippet = entry.get("initial_snippet", "")

        config_str = (
            f"hardware: {hardware}\n"
            f"backend: {backend}\n"
            f"model: {model_name}\n"
            f"root_cause_family: {entry.get('root_cause', 'unknown')}"
        )
        metrics_str = (
            "error_count: 1\n"
            "restart_attempts: 0\n"
            "gpu_utilization: N/A\n"
            "inference_latency: N/A"
        )
        inspect_results = InspectResult(
            logs=initial_log if initial_log else "No detailed logs available.",
            config=config_str,
            snippet=initial_snippet if initial_snippet else "No code snippet available.",
            metrics=metrics_str,
        )

        # --- specialist_followups: generate from opinion text ---
        specialist_followups: dict[str, str] = {}
        for name, op in specialist_opinions.items():
            if op.is_correct:
                specialist_followups[name] = (
                    f"Confirmed: {op.opinion} "
                    f"I stand by my earlier assessment with {op.confidence:.0%} confidence."
                )
            else:
                specialist_followups[name] = (
                    f"On further review, my initial assessment may not be the primary issue. "
                    f"Original observation: {op.opinion}"
                )

        scenarios.append(Scenario(
            id=entry["id"],
            root_cause=entry["root_cause"],
            correct_fix=entry["correct_fix"],
            incident_ticket=entry.get("incident_ticket", ""),
            hardware=hardware,
            model_name=model_name,
            backend=backend,
            initial_log=initial_log,
            initial_snippet=initial_snippet,
            specialist_opinions=specialist_opinions,
            inspect_results=inspect_results,
            specialist_followups=specialist_followups,
        ))

    return scenarios


# Load scraped scenarios (if the data file exists)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_SCRAPED_PATH = os.path.join(_DATA_DIR, "generated_scenarios_full.json")

if os.path.exists(_SCRAPED_PATH):
    _SCRAPED_SCENARIOS = load_scraped_scenarios(_SCRAPED_PATH)
else:
    _SCRAPED_SCENARIOS = []

# 80/20 train/eval split for scraped scenarios
_scraped_split = int(len(_SCRAPED_SCENARIOS) * 0.8)
SCRAPED_TRAIN_SCENARIOS = _SCRAPED_SCENARIOS[:_scraped_split]
SCRAPED_EVAL_SCENARIOS = _SCRAPED_SCENARIOS[_scraped_split:]

# Combine hand-crafted and scraped
SCENARIOS = _HANDCRAFTED_SCENARIOS + _SCRAPED_SCENARIOS

# _01, _03, _04, _05 = train; _02, _06 = eval  (hand-crafted)
HANDCRAFTED_TRAIN = [s for s in _HANDCRAFTED_SCENARIOS if s.id.endswith(("_01", "_03", "_04", "_05"))]
HANDCRAFTED_EVAL = [s for s in _HANDCRAFTED_SCENARIOS if s.id.endswith(("_02", "_06"))]

TRAIN_SCENARIOS = HANDCRAFTED_TRAIN + SCRAPED_TRAIN_SCENARIOS
EVAL_SCENARIOS = HANDCRAFTED_EVAL + SCRAPED_EVAL_SCENARIOS


def get_scenario(scenario_id: str | None = None, split: str = "train") -> Scenario:
    """Get a scenario by ID, or random from the given split."""
    if scenario_id:
        for s in SCENARIOS:
            if s.id == scenario_id:
                return s
        raise ValueError(f"Unknown scenario: {scenario_id}")
    pool = TRAIN_SCENARIOS if split == "train" else EVAL_SCENARIOS
    return random.choice(pool)


def randomize_specialist_opinions(
    scenario: Scenario,
) -> dict[str, SpecialistOpinion]:
    """Return a new dict of specialist opinions with per-episode randomization.

    - Randomly pick 1-2 specialists to have their correctness swapped
      (a correct one becomes wrong, a wrong one becomes correct).
    - Add noise to confidence scores (multiply by uniform(0.85, 1.15),
      clamped to [0.3, 0.99]).
    - The original scenario is NOT mutated.
    """
    names = list(scenario.specialist_opinions.keys())
    correct_names = [n for n in names if scenario.specialist_opinions[n].is_correct]
    incorrect_names = [n for n in names if not scenario.specialist_opinions[n].is_correct]

    # Determine how many to swap (1 or 2), limited by pool sizes
    max_swaps = min(len(correct_names), len(incorrect_names))
    if max_swaps == 0:
        num_swaps = 0
    else:
        num_swaps = random.randint(1, min(2, max_swaps))

    # Pick which specialists get swapped
    swap_correct = random.sample(correct_names, num_swaps) if num_swaps > 0 else []
    swap_incorrect = random.sample(incorrect_names, num_swaps) if num_swaps > 0 else []

    # Build swap mapping: correct[i] gets incorrect[i]'s opinion text, and vice versa
    swap_pairs: dict[str, str] = {}
    for c, ic in zip(swap_correct, swap_incorrect):
        swap_pairs[c] = ic
        swap_pairs[ic] = c

    new_opinions: dict[str, SpecialistOpinion] = {}
    for name in names:
        orig = scenario.specialist_opinions[name]

        if name in swap_pairs:
            # Swap: take the opinion text from the partner, flip is_correct
            partner = scenario.specialist_opinions[swap_pairs[name]]
            opinion_text = partner.opinion
            is_correct = not orig.is_correct
        else:
            opinion_text = orig.opinion
            is_correct = orig.is_correct

        # Add noise to confidence
        noisy_confidence = orig.confidence * random.uniform(0.85, 1.15)
        noisy_confidence = max(0.3, min(0.99, noisy_confidence))

        new_opinions[name] = SpecialistOpinion(
            opinion=opinion_text,
            confidence=noisy_confidence,
            is_correct=is_correct,
        )

    return new_opinions
