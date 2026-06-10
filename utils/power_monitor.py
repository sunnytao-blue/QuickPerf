import subprocess
import os
import time
import re
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PowerSample:
    timestamp: float
    cpu_power_watts: float
    gpu_power_watts: float


class PowerMonitor:
    def __init__(self):
        self._is_macos = os.name == "posix" and os.uname().sysname == "Darwin"
        self._is_linux = os.name == "posix" and os.uname().sysname == "Linux"
        self._is_windows = os.name == "nt"
        self._samples: list[PowerSample] = []
        self._monitoring = False
        self._prev_energy = None
        self._prev_time = None

    def _get_cpu_power_linux(self) -> float:
        try:
            with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
                return int(f.read()) / 1_000_000
        except Exception:
            pass

        try:
            with open("/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj", "r") as f:
                return int(f.read()) / 1_000_000
        except Exception:
            pass

        return 0.0

    def _get_gpu_power_nvidia(self) -> float:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    return float(lines[0])
        except Exception:
            pass
        return 0.0

    def _get_gpu_power_intel(self) -> float:
        try:
            with open("/sys/class/drm/card0/device/power1_average", "r") as f:
                return int(f.read()) / 1_000_000
        except Exception:
            pass

        try:
            with open("/sys/class/drm/card1/device/power1_average", "r") as f:
                return int(f.read()) / 1_000_000
        except Exception:
            pass

        return 0.0

    def _get_power_darwin(self) -> Tuple[float, float]:
        cpu_power = 0.0
        gpu_power = 0.0

        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    match = re.search(r"(\d+\.?\d*)\s*W", line)
                    if match:
                        cpu_power = float(match.group(1))
                        break
        except Exception:
            pass

        if cpu_power == 0:
            try:
                result = subprocess.run(
                    ["system_profiler", "SPPowerDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "Power" in line and "Watts" in line:
                            match = re.search(r"(\d+\.?\d*)\s*Watts?", line)
                            if match:
                                cpu_power = float(match.group(1))
                                break
            except Exception:
                pass

        if cpu_power == 0:
            try:
                result = subprocess.run(
                    ["ioreg", "-c", "AppleSmartBattery", "-r", "-l"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if '"CurrentCapacity"' in line:
                            current_cap = float(line.split("=")[1].strip())
                        elif '"MaxCapacity"' in line:
                            max_cap = float(line.split("=")[1].strip())
                        elif '"DesignCapacity"' in line:
                            design_cap = float(line.split("=")[1].strip())
                            try:
                                cpu_power = (current_cap / max_cap) * 100
                            except:
                                pass

                result = subprocess.run(
                    ["ioreg", "-n", "AppleACPIPlatformExpert", "-r", "-l"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "ACPower" in line and "Yes" in line:
                            try:
                                result2 = subprocess.run(
                                    ["pmset", "-g", "ps"],
                                    capture_output=True,
                                    text=True,
                                    timeout=1
                                )
                                if result2.returncode == 0:
                                    for l in result2.stdout.split("\n"):
                                        if "AC Power" in l:
                                            parts = l.split()
                                            for i, part in enumerate(parts):
                                                if part == "Watt" and i > 0:
                                                    try:
                                                        cpu_power = float(parts[i-1])
                                                    except ValueError:
                                                        pass
                            except Exception:
                                pass
            except Exception:
                pass

        return cpu_power, gpu_power

    def _get_power_windows(self) -> Tuple[float, float]:
        cpu_power = 0.0
        gpu_power = 0.0

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    gpu_power = float(lines[0])
        except Exception:
            pass

        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"SYSTEM\CurrentControlSet\Services\battery\Enum") as key:
                pass
        except Exception:
            pass

        return cpu_power, gpu_power

    def _get_cpu_energy_linux(self) -> float:
        try:
            total = 0
            for i in range(5):
                path = f"/sys/class/powercap/intel-rapl/intel-rapl:{i}/energy_uj"
                if os.path.exists(path):
                    with open(path, "r") as f:
                        total += int(f.read())
            return total / 1_000_000
        except Exception:
            return 0.0

    def sample(self) -> PowerSample:
        timestamp = time.perf_counter()
        cpu_power = 0.0
        gpu_power = 0.0

        if self._is_linux:
            current_energy = self._get_cpu_energy_linux()
            if self._prev_energy is not None and self._prev_time is not None and timestamp > self._prev_time:
                elapsed = timestamp - self._prev_time
                if elapsed > 0:
                    cpu_power = (current_energy - self._prev_energy) / elapsed
            self._prev_energy = current_energy
            self._prev_time = timestamp

            gpu_power = self._get_gpu_power_nvidia()
            if gpu_power == 0:
                gpu_power = self._get_gpu_power_intel()
        elif self._is_macos:
            cpu_power, gpu_power = self._get_power_darwin()
        elif self._is_windows:
            cpu_power, gpu_power = self._get_power_windows()

        return PowerSample(timestamp, cpu_power, gpu_power)

    def start(self):
        self._samples = []
        self._monitoring = True
        self._start_time = time.perf_counter()
        self._prev_energy = None
        self._prev_time = None

    def stop(self) -> Tuple[float, float]:
        self._monitoring = False

        if not self._samples:
            return 0.0, 0.0

        valid_cpu = [s.cpu_power_watts for s in self._samples if s.cpu_power_watts > 0]
        valid_gpu = [s.gpu_power_watts for s in self._samples if s.gpu_power_watts > 0]

        cpu_avg = sum(valid_cpu) / len(valid_cpu) if valid_cpu else 0.0
        gpu_avg = sum(valid_gpu) / len(valid_gpu) if valid_gpu else 0.0

        return cpu_avg, gpu_avg

    def add_sample(self):
        if self._monitoring:
            self._samples.append(self.sample())


class PowerMonitorContext:
    def __init__(self, sample_interval: float = 0.1):
        self.monitor = PowerMonitor()
        self.sample_interval = sample_interval
        self._thread = None
        self._running = False

    def __enter__(self):
        self.monitor.start()
        self._running = True

        def sampler():
            while self._running:
                self.monitor.add_sample()
                time.sleep(self.sample_interval)

        import threading
        self._thread = threading.Thread(target=sampler, daemon=True)
        self._thread.start()
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)


def get_cpu_power() -> float:
    monitor = PowerMonitor()
    return monitor.sample().cpu_power_watts


def get_gpu_power() -> float:
    monitor = PowerMonitor()
    return monitor.sample().gpu_power_watts


def measure_power_during(func, sample_interval: float = 0.1) -> Tuple[float, float, float]:
    start = time.perf_counter()
    with PowerMonitorContext(sample_interval) as monitor:
        result = func()
        elapsed = time.perf_counter() - start
    cpu_power, gpu_power = monitor.stop()
    return elapsed, cpu_power, gpu_power


if __name__ == "__main__":
    print("Testing power monitor...")
    monitor = PowerMonitor()
    
    print("\nSingle sample:")
    sample = monitor.sample()
    print(f"CPU Power: {sample.cpu_power_watts:.2f} W")
    print(f"GPU Power: {sample.gpu_power_watts:.2f} W")

    print("\nTesting with context manager (3 seconds)...")
    with PowerMonitorContext(sample_interval=0.1) as pm:
        time.sleep(1)
        print("Doing work...")
        import numpy as np
        a = np.random.rand(2000, 2000)
        for _ in range(5):
            _ = a @ a
        time.sleep(1)
    cpu_avg, gpu_avg = pm.stop()
    print(f"Average CPU Power: {cpu_avg:.2f} W")
    print(f"Average GPU Power: {gpu_avg:.2f} W")
