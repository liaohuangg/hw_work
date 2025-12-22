# 类型存根文件：为 gem5 m5.objects 模块提供类型提示
# 这些类在 gem5 编译时动态生成，此文件仅用于 IDE 类型检查

from typing import Any, List, Optional

# 基础 SimObject 类
class SimObject:
    pass

# 系统相关类
class System(SimObject):
    clk_domain: Any
    mem_mode: str
    mem_ranges: List[Any]
    cpu: Any
    membus: Any
    mem_ctrl: Any
    system_port: Any
    workload: Any

class SrcClockDomain(SimObject):
    clock: str
    voltage_domain: Any

class VoltageDomain(SimObject):
    pass

class AddrRange:
    def __init__(self, size: str) -> None: ...

# CPU 相关类
class X86TimingSimpleCPU(SimObject):
    icache_port: Any
    dcache_port: Any
    interrupts: List[Any]
    workload: Any
    
    def createInterruptController(self) -> None: ...
    def createThreads(self) -> None: ...

# 内存总线相关类
class SystemXBar(SimObject):
    cpu_side_ports: Any
    mem_side_ports: Any

# 内存控制器相关类
class MemCtrl(SimObject):
    dram: Any
    port: Any

class DDR3_1600_8x8(SimObject):
    range: Any

# 工作负载相关类
class SEWorkload:
    @staticmethod
    def init_compatible(binary: str) -> Any: ...

class Process(SimObject):
    cmd: List[str]

# 根对象
class Root(SimObject):
    def __init__(self, full_system: bool = False, system: Optional[System] = None) -> None: ...

