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

    return scenarios


# Build the full scenario pool
SCENARIOS = _make_scenarios()
TRAIN_SCENARIOS = [s for s in SCENARIOS if not s.id.endswith("_02")]
EVAL_SCENARIOS = [s for s in SCENARIOS if s.id.endswith("_02")]


def get_scenario(scenario_id: str | None = None, split: str = "train") -> Scenario:
    """Get a scenario by ID, or random from the given split."""
    if scenario_id:
        for s in SCENARIOS:
            if s.id == scenario_id:
                return s
        raise ValueError(f"Unknown scenario: {scenario_id}")
    pool = TRAIN_SCENARIOS if split == "train" else EVAL_SCENARIOS
    return random.choice(pool)
