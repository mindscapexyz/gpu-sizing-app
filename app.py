import math
import gradio as gr

# =========================
# --- Tab 1: GPU Sizing ---
# =========================
SECONDS_PER_MONTH_30D = 30 * 24 * 3600  # 2,592,000 seconds

def derive_per_gpu_tps(cluster_tps, n_gpus, fallback_tps_per_gpu):
    if cluster_tps and n_gpus and n_gpus > 0:
        try:
            return float(cluster_tps) / float(n_gpus)
        except Exception:
            pass
    return float(fallback_tps_per_gpu)

def monthly_to_rps(calls_per_month, avg_call_minutes):
    calls_per_month = max(0.0, float(calls_per_month))
    avg_call_minutes = max(0.0, float(avg_call_minutes))
    avg_rps = calls_per_month / SECONDS_PER_MONTH_30D
    avg_handle_time_s = avg_call_minutes * 60.0
    return avg_rps, avg_handle_time_s

def size_gpus(
    calls_per_month,
    avg_call_minutes,
    max_concurrency,
    target_stream_tps,
    tps_per_gpu_input,
    util_cap,
    cluster_tps,
    n_gpus_measured,
):
    # Derived stats
    avg_rps, avg_handle_time_s = monthly_to_rps(calls_per_month, avg_call_minutes)
    per_gpu_tps = derive_per_gpu_tps(cluster_tps, n_gpus_measured, max(1.0, float(tps_per_gpu_input)))
    util_cap = min(0.95, max(0.10, float(util_cap)))
    target_stream_tps = max(0.0, float(target_stream_tps))
    max_concurrency = max(0.0, float(max_concurrency))

    # Core math
    required_fleet_tps = max_concurrency * target_stream_tps
    usable_gpu_tps = per_gpu_tps * util_cap
    required_gpus = 0 if usable_gpu_tps <= 0 else math.ceil(required_fleet_tps / usable_gpu_tps)
    n_plus_1 = 0 if required_gpus == 0 else required_gpus + 1

    # Sensitivity table
    caps = [0.50, 0.60, 0.70, 0.80]
    table = "Util Cap | GPUs Needed\n---|---\n"
    for c in caps:
        eff = max(1.0, per_gpu_tps * c)
        table += f"{int(c*100)}% | {max(0, math.ceil(required_fleet_tps / eff))}\n"

    # Summary text
    lines = []
    lines.append("📊 **Workload Summary**")
    lines.append(f"- Calls per month: **{calls_per_month:,.0f}**")
    lines.append(f"- Average call duration: **{avg_call_minutes:.2f} min** (= {avg_handle_time_s:.0f}s)")
    lines.append(f"- Average RPS (for reference): **{avg_rps:.4f} req/s**")
    lines.append(f"- Max concurrent users: **{max_concurrency:,.0f}**\n")
    lines.append("⚙️ **Sizing Inputs**")
    lines.append(f"- Target stream rate per call: **{target_stream_tps:.2f} tokens/s**")
    if cluster_tps and n_gpus_measured:
        lines.append(f"- Per-GPU TPS (derived): **{per_gpu_tps:,.0f} tok/s** from {float(cluster_tps):,.0f} TPS / {int(n_gpus_measured)} GPU(s)")
    else:
        lines.append(f"- Per-GPU TPS (input): **{per_gpu_tps:,.0f} tok/s**")
    lines.append(f"- Utilization cap: **{util_cap:.0%}**")
    lines.append(f"- Fleet required TPS: **{required_fleet_tps:,.0f} tok/s** (= max_concurrency × target_stream_tps)\n")
    lines.append(f"✅ **Required GPUs (ceil)**: **{required_gpus}**")
    if required_gpus > 0:
        lines.append(f"🧩 **N+1 suggestion**: **{n_plus_1} GPUs**")
    return "\n".join(lines), table


# =========================
# --- Tab 2: Latency Predictor (Blocks, no duplicate) ---
# =========================
COEFF_TOTAL = {
    "const": 1.4171,
    "context": 0.0001,
    "concurrency": 0.0092,
    "TP": -0.0201,
    "DP": 0.2126,
    "gpu_type_encoded": -0.1040,
}
COEFF_TTFT = {
    "const": 1.2604,
    "context": 9.302e-05,
    "concurrency": 0.0041,
    "TP": -0.0749,
    "DP": -0.1258,
    "gpu_type_encoded": 0.1627,
}

def _predict_generic(context, TP, DP, concurrency, gpu_type, coeffs):
    gpu_encoded = 1 if gpu_type == "8× H200" else 0
    latency = (
        coeffs["const"]
        + coeffs["context"] * context
        + coeffs["concurrency"] * concurrency
        + coeffs["TP"] * TP
        + coeffs["DP"] * DP
        + coeffs["gpu_type_encoded"] * gpu_encoded
    )
    return round(latency, 3)

def predict_latency(model, context, TP, DP, concurrency, gpu_type):
    if TP * DP > 8:
        msg = "⚠️ Invalid configuration: TP × DP must not exceed 8 GPUs."
        return (msg, msg)
    if gpu_type not in ["8× H100", "8× H200"]:
        msg = "⚠️ Invalid GPU selection. Only 8× H100 or 8× H200 supported."
        return (msg, msg)

    total = _predict_generic(context, TP, DP, concurrency, gpu_type, COEFF_TOTAL)
    ttft = _predict_generic(context, TP, DP, concurrency, gpu_type, COEFF_TTFT)
    return (
        f"Model: {model}\nGPU: {gpu_type}\n\nPredicted Total Response Latency: {total:.3f} s",
        f"Model: {model}\nGPU: {gpu_type}\n\nPredicted First Token Latency (TTFT): {ttft:.3f} s",
    )

def clear_latency():
    return ("", "")

# =========================
# --- Unified App (Soft Theme) ---
# =========================
with gr.Blocks(title="LLM GPU Planner", theme="soft") as app:
    gr.Markdown(
        "# 🧮 LLM GPU Planner\n"
        "Two tools to help you **predict latency** and **estimate GPU requirements**. Use the tabs below 👇"
    )

    with gr.Tabs():
        # ---- GPU Sizing Tab ----
        with gr.Tab("📈 GPU Sizing (Concurrency)"):
            gr.Markdown(
                "Estimate GPUs needed from **max concurrency**.\n\n"
                "Formula: `GPUs = ceil(max_concurrency × target_stream_tps / (per_gpu_tps × util_cap))`"
            )
            with gr.Row():
                with gr.Column():
                    calls_month = gr.Number(label="Calls per month", value=66200, precision=0)
                    avg_minutes = gr.Number(label="Average call duration (minutes)", value=10.0)
                    max_conc = gr.Number(label="Max concurrent active users", value=90)
                with gr.Column():
                    target_tps = gr.Number(label="Target stream tokens/s per call", value=10.0)
                    util = gr.Slider(label="Utilization cap (headroom)", value=0.60, minimum=0.10, maximum=0.95, step=0.01)
                    tps_per_gpu = gr.Number(label="Per-GPU tokens/s (measured or expected)", value=8300)
                    gr.Markdown("**Optional:** derive Per-GPU TPS from a measured cluster")
                    cluster_tps = gr.Number(label="Measured cluster tokens/s", value=None)
                    n_gpus_measured = gr.Number(label="#GPUs in that measurement", value=None, precision=0)
            btn = gr.Button("Calculate", variant="primary")
            summary_md = gr.Markdown()
            table_md = gr.Markdown()
            btn.click(
                size_gpus,
                inputs=[calls_month, avg_minutes, max_conc, target_tps, tps_per_gpu, util, cluster_tps, n_gpus_measured],
                outputs=[summary_md, table_md],
            )

        # ---- Latency Predictor Tab (Blocks only; no Interface duplication) ----
        with gr.Tab("⚡ LLM Latency Predictor"):
            gr.Markdown(
                "Estimate **total response** and **first-token (TTFT)** latency using your fitted model.\n\n"
                "⚙️ Validation: TP × DP ≤ 8; only 8× GPU setups are valid."
            )
            with gr.Row():
                with gr.Column():
                    model_dd = gr.Dropdown(["Qwen3-32B"], value="Qwen3-32B", label="Model")
                    ctx = gr.Slider(1024, 32768, value=8192, step=1024, label="Context Length (tokens)")
                    tp = gr.Dropdown([1, 2, 4, 8], value=4, label="Tensor Parallel (TP)")
                    dp = gr.Dropdown([1, 2, 4, 8], value=2, label="Data Parallel (DP)")
                    conc = gr.Slider(1, 200, value=100, step=5, label="Concurrency (requests)")
                    gpu_dd = gr.Dropdown(["8× H100", "8× H200"], value="8× H100", label="GPU Type")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant="primary")
                with gr.Column():
                    total_out = gr.Textbox(label="Predicted Total Response Latency")
                    ttft_out = gr.Textbox(label="Predicted First Token Latency (TTFT)")

            submit_btn.click(
                predict_latency,
                inputs=[model_dd, ctx, tp, dp, conc, gpu_dd],
                outputs=[total_out, ttft_out],
            )
            clear_btn.click(
                clear_latency,
                inputs=None,
                outputs=[total_out, ttft_out],
            )

if __name__ == "__main__":
    app.launch()
