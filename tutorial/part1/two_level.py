# Copyright (c) 2024
# Simple gem5 configuration script for X86 ISA
# 简单的 gem5 X86 ISA 配置脚本
# 此脚本创建一个基本系统并运行 "Hello World" 应用程序

import m5
import os
from caches import *
# 导入所有 SimObject（gem5 编译时动态生成的对象）
# 注意：这些对象在 gem5 编译时动态生成，IDE 通过类型存根文件识别
from m5.objects import (
    System,
    SrcClockDomain,
    VoltageDomain,
    AddrRange,
    X86TimingSimpleCPU,
    SystemXBar,
    MemCtrl,
    DDR3_1600_8x8,
    SEWorkload,
    Process,
    Root, 
    L2XBar
)

# 创建要模拟的系统
system = System()

# 设置系统的时钟频率（及其所有子组件的时钟频率）
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()

# 设置系统参数
system.mem_mode = "timing"  # 使用时序访问模式
system.mem_ranges = [AddrRange("512MiB")]  # 创建地址范围：512MB

# 创建一个简单的 CPU
system.cpu = X86TimingSimpleCPU()

# 创建内存总线，这里使用系统交叉开关（System Crossbar）
system.membus = SystemXBar()

# 有了两级缓存后，不需要将 CPU 的端口连接到内存总线
# system.cpu.icache_port = system.membus.cpu_side_ports  # 指令缓存端口
# system.cpu.dcache_port = system.membus.cpu_side_ports  # 数据缓存端口

# 为 CPU 创建中断控制器并连接到内存总线
system.cpu.createInterruptController()

# 对于 X86 架构，确保中断连接到内存
# 注意：这些中断直接连接到内存总线，不经过缓存
# pio: 程序化 I/O 端口，连接到内存总线的内存侧端口
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
# int_requestor: 中断请求端口，连接到内存总线的 CPU 侧端口
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
# int_responder: 中断响应端口，连接到内存总线的内存侧端口（不是 CPU 侧！）
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# 创建 DDR3 内存控制器并连接到内存总线
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# 将系统端口连接到内存总线
system.system_port = system.membus.cpu_side_ports

# 设置 X86 "hello world" 二进制文件路径
thispath = os.path.dirname(os.path.realpath(__file__))
# binary = os.path.join(
#     thispath,
#     "../../../",
#     "/home/huangl/workspace/gem5/tests/test-progs/hello/bin/x86/linux/hello",
# )
binary = os.path.join(
    thispath,
    "../../../",
    "/home/huangl/workspace/hw_work/benchmark/conv2d",
)


# ========== 两级缓存配置 ==========
# 连接顺序：CPU -> L1 Cache -> L2 Bus -> L2 Cache -> Memory Bus -> Memory

# 创建 L1 指令和数据缓存
system.cpu.icache = L1ICache()
system.cpu.dcache = L1DCache()

# 步骤 1: 将 L1 缓存连接到 CPU 端口
# connectCPU 方法会将 CPU 的 icache_port/dcache_port 连接到 L1 缓存的 cpu_side
system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)

# 步骤 2: 创建 L2 总线，用于连接 L1 缓存和 L2 缓存
system.l2bus = L2XBar()

# 步骤 3: 将 L1 缓存连接到 L2 总线
# connectBus 方法会将 L1 缓存的 mem_side 连接到 L2 总线的 cpu_side_ports
system.cpu.icache.connectBus(system.l2bus)
system.cpu.dcache.connectBus(system.l2bus)

# 步骤 4: 创建 L2 缓存并将其连接到 L2 总线（CPU 侧）
system.l2cache = L2Cache()
# connectCPUSideBus 方法会将 L2 缓存的 cpu_side 连接到 L2 总线的 mem_side_ports
system.l2cache.connectCPUSideBus(system.l2bus)

# 步骤 5: 将 L2 缓存连接到内存总线（内存侧）
# connectMemSideBus 方法会将 L2 缓存的 mem_side 连接到内存总线的 cpu_side_ports
# 注意：membus 已经在第 42 行创建，不要重新创建！
system.l2cache.connectMemSideBus(system.membus)


# 设置工作负载（SE 模式：系统调用模拟模式）
system.workload = SEWorkload.init_compatible(binary)

# 创建一个简单的 "Hello World" 应用程序进程
process = Process()
# 设置命令
# cmd 是一个列表，以可执行文件开头（类似于 argv）
process.cmd = [binary]
# 设置 CPU 使用该进程作为其工作负载，并创建线程上下文
system.cpu.workload = process
system.cpu.createThreads()

# 设置根 SimObject 并启动模拟
root = Root(full_system=False, system=system)
# 实例化上面创建的所有对象
m5.instantiate()

print("开始模拟！")
exit_event = m5.simulate()
print(f"在 tick {m5.curTick()} 退出，原因：{exit_event.getCause()}")