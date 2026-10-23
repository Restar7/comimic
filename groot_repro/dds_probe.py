import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
try:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
except Exception:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
count = 0
def cb(msg):
    global count; count += 1
ChannelFactoryInitialize(0, "lo")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(cb, 10)
print("subscribed rt/lowstate, waiting 8s...")
time.sleep(8)
print("messages:", count)
