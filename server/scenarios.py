"""
Scenario data for the Launch-Day War Room.

Each scenario encodes a hidden root cause, the correct fix, an incident ticket,
hardware/model/backend context, log and code snippets, and specialist opinions
(some of which may be wrong).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


ROOT_CAUSES = [
    "arch_guard",
    "backend_whitelist",
    "runtime_loader",
    "backend_selector",
    "model_config",
    "weight_layout",
]

FIXES = [
    "relax_arch_check",
    "add_whitelist_entry",
    "fix_runtime_path",
    "switch_backend",
    "update_model_config",
    "fix_weight_mapping",
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

    return scenarios


# Build the full scenario pool
SCENARIOS = _make_scenarios()
# _01, _03, _04, _05 = train; _02, _06 = eval
TRAIN_SCENARIOS = [s for s in SCENARIOS if s.id.endswith(("_01", "_03", "_04", "_05"))]
EVAL_SCENARIOS = [s for s in SCENARIOS if s.id.endswith(("_02", "_06"))]


def get_scenario(scenario_id: str | None = None, split: str = "train") -> Scenario:
    """Get a scenario by ID, or random from the given split."""
    if scenario_id:
        for s in SCENARIOS:
            if s.id == scenario_id:
                return s
        raise ValueError(f"Unknown scenario: {scenario_id}")
    pool = TRAIN_SCENARIOS if split == "train" else EVAL_SCENARIOS
    return random.choice(pool)
