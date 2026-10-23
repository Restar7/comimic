# 四个任务的 prompt

评测/部署时喂给 policy 的 `annotation.human.task_description` 原文：

1. **apple**
   > Pick up the apple from the table in front of you, turn around, walk to the table behind you, and place the apple into the box.

2. **carton**
   > Pick up the carton from the table in front of you, turn around, walk to the table behind you, and place the carton into the box.

3. **red cup**
   > Pick up the red cup, place it into the upper drawer of the cabinet, and close the drawer.

4. **handover**
   > pick up the carton with the right hand, hand it over to the left hand, and place it into the box

5. **task4（地上捡罐子）**
   > Walk to the front of the platform, crouch down, pick up the can from the floor, stand up, and place the can on the platform.

这些字符串直接来自部署脚本和评测日志：

- `GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py`
- `SONIC_DATA/deployment/bin/start_handover_*_client.sh`
- `SONIC_DATA/groot_eval/**/client.log.gz` 里的 `language prompt:`
