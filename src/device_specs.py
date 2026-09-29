"""Real device classes for the *serving infrastructure* (Tier 2/3 nodes).

These describe the machines services are placed on - NOT the requesting
devices. A Tier-1 request stays pure natural-language text; nothing in
this table ever rides inside a request payload.

CPU / RAM / power figures are transcribed from the official vendor
datasheets cited per entry (no other source; see ARCHITECTURE.md
"Where real numbers were approximated"). Where a datasheet offers a range
of configurations, the representative choice made here is stated in
`config_choice`.

`deploy_ms_median` / `deploy_ms_sigma` (container start-up time, a
log-normal) are NOT datasheet numbers - they are an explicit modelling
assumption, flagged `deploy_is_assumption=True`, used only until
scripts/generate_device_calibration.py --measure-container-start has
written a measured value into data_foundation/validated/device_calibration.json
(see load_device_calibration()). They only affect the `deployment`
latency component, which is identical across all compared systems for
the same placement, so they cannot change a system ranking.
"""
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class DeviceClass:
    key: str
    display_name: str
    cpu_desc: str
    cores: int
    ram_gb: float
    power_w: float
    power_note: str
    accelerator: Optional[str]         # None | "gpu" | "edge_tpu"
    accel_note: str
    config_choice: str
    source_url: str
    source_note: str
    deploy_ms_median: float
    deploy_ms_sigma: float
    deploy_is_assumption: bool = True


DEVICE_CLASSES = {
    # Raspberry Pi 4 Model B official specifications:
    # https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/
    # product brief: https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-product-brief.pdf
    "raspberry_pi_4": DeviceClass(
        key="raspberry_pi_4",
        display_name="Raspberry Pi 4 Model B (8GB)",
        cpu_desc="Broadcom BCM2711, quad-core Cortex-A72 (ARM v8) 64-bit @ 1.5GHz",
        cores=4, ram_gb=8.0,
        power_w=15.0,
        power_note="15 W max: 5V/3A USB-C supply specification",
        accelerator=None, accel_note="",
        config_choice="8GB LPDDR4 SKU",
        source_url="https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/",
        source_note="Official specifications page + product brief PDF "
                    "(https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-product-brief.pdf)",
        deploy_ms_median=2500.0, deploy_ms_sigma=0.35),
    # NVIDIA Jetson Orin Nano Series datasheet DS-11105-001:
    # https://www.mouser.com/pdfDocs/Jetson_Orin_Nano_Series_DS-11105-001_v11.pdf
    "jetson_orin_nano": DeviceClass(
        key="jetson_orin_nano",
        display_name="NVIDIA Jetson Orin Nano (8GB)",
        cpu_desc="6-core Arm Cortex-A78AE",
        cores=6, ram_gb=8.0,
        power_w=15.0,
        power_note="configurable power modes 7W/15W/25W; 15 W (mid) mode used",
        accelerator="gpu", accel_note="integrated NVIDIA Ampere GPU",
        config_choice="8GB LPDDR5 module, 15W power mode",
        source_url="https://www.mouser.com/pdfDocs/Jetson_Orin_Nano_Series_DS-11105-001_v11.pdf",
        source_note="NVIDIA Jetson Orin Nano Series Data Sheet DS-11105-001",
        deploy_ms_median=1500.0, deploy_ms_sigma=0.35),
    # Google Coral Dev Board datasheet:
    # https://www.coral.ai/docs/dev-board/datasheet/
    "coral_dev_board": DeviceClass(
        key="coral_dev_board",
        display_name="Google Coral Dev Board (4GB)",
        cpu_desc="NXP i.MX 8M SoC, quad-core Cortex-A53 @ 1.5GHz + Cortex-M4F; "
                 "Edge TPU coprocessor (4 TOPS)",
        cores=4, ram_gb=4.0,
        power_w=4.0,
        power_note="~4 W typical operating power",
        accelerator="edge_tpu", accel_note="Google Edge TPU, 4 TOPS (int8)",
        config_choice="4GB LPDDR4 SKU",
        source_url="https://www.coral.ai/docs/dev-board/datasheet/",
        source_note="Coral Dev Board datasheet (smart-camera / NPU-class device)",
        deploy_ms_median=3000.0, deploy_ms_sigma=0.35),
    # Dell PowerEdge rack server spec sheet:
    # https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-rack-series-spec-sheet.pdf
    "rack_edge_server": DeviceClass(
        key="rack_edge_server",
        display_name="Dell PowerEdge rack server (2S, 64 cores)",
        cpu_desc="dual-socket; 64 total cores (representative high-core-count SKU)",
        cores=64, ram_gb=512.0,
        power_w=800.0,
        power_note="800 W PSU rating (sheet lists 800-1100 W options)",
        accelerator=None, accel_note="",
        config_choice="64 cores / 512 GB (sheet supports up to ~2 TB) / 800 W PSU",
        source_url="https://www.delltechnologies.com/asset/en-us/products/servers/"
                   "technical-support/poweredge-rack-series-spec-sheet.pdf",
        source_note="Dell PowerEdge Rack Servers spec sheet (official PDF)",
        deploy_ms_median=600.0, deploy_ms_sigma=0.30),
}

# Citation-only reference for low-power inference benchmarking methodology
# (NOT a source of any number above): MLPerf Tiny,
# https://github.com/mlcommons/tiny and https://github.com/mlcommons/tiny_results_v1.0
BENCHMARK_METHODOLOGY_REFS = [
    "https://github.com/mlcommons/tiny",
    "https://github.com/mlcommons/tiny_results_v1.0",
]


def get(key):
    return DEVICE_CLASSES[key]


def as_table():
    return [asdict(d) for d in DEVICE_CLASSES.values()]


def load_device_calibration(path=None):
    """Return {device_key: {"deploy_ms_median", "deploy_ms_sigma",
    "deploy_is_assumption", ...}} merging any measured container start-up
    times over the stated assumptions. Missing file -> assumptions only."""
    out = {k: {"deploy_ms_median": d.deploy_ms_median,
               "deploy_ms_sigma": d.deploy_ms_sigma,
               "deploy_is_assumption": d.deploy_is_assumption}
           for k, d in DEVICE_CLASSES.items()}
    if path and os.path.exists(path):
        cal = json.load(open(path))
        for k, v in cal.get("devices", {}).items():
            if k in out and v.get("deploy_ms_median"):
                out[k].update(deploy_ms_median=float(v["deploy_ms_median"]),
                              deploy_ms_sigma=float(v.get("deploy_ms_sigma", 0.3)),
                              deploy_is_assumption=False,
                              measured_on=v.get("measured_on", ""))
    return out
