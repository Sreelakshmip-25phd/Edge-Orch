"""Tier 1 - edge devices: the origin of requests.

A device is deliberately cheap: an id, a (moving) position expressed as
the zone it is in at simulated time t, and the device->zone transmission
delay. What it sends is *only* natural-language intent text. The zone a
request lands in is decided by the network (the device's current zone),
not by anything inside the payload; the device id travels in telemetry
for per-device analysis but no orchestrator logic reads it.
"""
import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentRequest:
    """The complete payload an orchestrator receives. Pure NL + routing."""
    req_id: str
    zone_id: str          # receiving zone (network routing, not payload)
    text: str
    t: float


class EdgeDevice:
    def __init__(self, device_id, home_zone, trajectory, slot_s, transport_ms):
        self.device_id = device_id
        self.home_zone = home_zone
        # trajectory: [[slot, zone_idx], ...] change points, slot-ascending
        self._slots = [int(s) for s, _ in trajectory]
        self._zones = [f"z{int(z)}" for _, z in trajectory]
        self.slot_s = float(slot_s)
        self.transport_ms = float(transport_ms)

    def zone_at(self, t):
        slot = int(t // self.slot_s)
        i = bisect.bisect_right(self._slots, slot) - 1
        return self._zones[max(i, 0)]

    def send(self, req_id, text, t):
        return IntentRequest(req_id=req_id, zone_id=self.zone_at(t),
                             text=text, t=float(t))


class DeviceFleet:
    def __init__(self, devices):
        self.devices = {d.device_id: d for d in devices}

    @classmethod
    def from_workload(cls, wl, topo):
        slot_s = wl["meta"].get("slot_s", 600.0)
        return cls([EdgeDevice(d["device_id"], d["home_zone"], d["trajectory"],
                               slot_s, topo["intra_zone_ms"])
                    for d in wl["devices"]])

    def __getitem__(self, k):
        return self.devices[k]

    def __len__(self):
        return len(self.devices)
