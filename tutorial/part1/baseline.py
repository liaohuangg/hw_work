# Copyright (c) 2024
# Baseline gem5 configuration script for X86 ISA using SimpleBoard
# 基准 gem5 X86 ISA 配置脚本（使用 SimpleBoard）
# 此脚本创建一个基本系统并运行本地二进制文件 conv2d

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

# 设置缓存层次结构（这里使用无缓存配置）
cache_hierarchy = NoCache()

# 设置内存系统（单通道 DDR3 1600）
memory = SingleChannelDDR3_1600(size="512MiB")

# 创建处理器（X86 Timing CPU，单核）
processor = SimpleProcessor(
    cpu_type=CPUTypes.TIMING,
    isa=ISA.X86,
    num_cores=1,
)

# 创建 SimpleBoard（用于 SE 模式模拟）
board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# 设置要运行的本地二进制文件路径
binary_path = "/home/huangl/workspace/hw_work/benchmark/conv2d"

# 创建 BinaryResource 对象
binary_resource = BinaryResource(local_path=binary_path)

# 使用 set_se_binary_workload 设置 SE 模式工作负载
board.set_se_binary_workload(binary=binary_resource)

# 创建模拟器并运行
simulator = Simulator(board=board)
print("开始模拟！")
simulator.run()
print(
    "在 tick {} 退出，原因：{}".format(
        simulator.get_current_tick(),
        simulator.get_last_exit_event_cause(),
    )
)

