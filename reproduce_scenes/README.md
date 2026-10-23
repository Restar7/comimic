# 场景与初始位姿（可完全复现）

这里放的是**部署/评测时实际使用的 MuJoCo 场景**，以及对应的初始位姿文件。
拿到这个目录就能把四个任务的场景一字不差地重建出来。

```
reproduce_scenes/
├── g1/                          # 与部署机上完全一致的目录结构
│   ├── g1_29dof_with_hand.xml   # 机器人本体（MJCF）
│   ├── g1_29dof_with_hand.urdf  # 机器人本体（URDF，备用）
│   ├── g1_29dof_with_hand_carton_aligned.xml
│   ├── scene_43dof.xml           # 红杯任务：柜子+抽屉
│   ├── scene_pickplace.xml       # 苹果任务：苹果+箱子
│   ├── scene_43dof_two_table.xml # carton 任务：橙桌纸箱 + 白箱桌
│   ├── scene_43dof_handover.xml  # handover 任务：托盘 + 目标箱
│   ├── scene_43dof_bend_and_pick.xml  # task4：地上捡罐子放台面
│   ├── scene_twotables.xml / scene_43dof_two_table_carton_aligned.xml / ...
│   └── meshes/                   # 146 个网格文件（STL + YCB 的 obj/png，163 MB）
├── initial_states/              # 初始位姿（见下）
├── prompts.md                   # 四个任务的文本 prompt
├── verify_scenes.py             # 一键验证所有场景能加载、能渲染
└── README.md
```

## 为什么是这些文件

- 场景 XML 里通过 `<include file="g1_29dof_with_hand.xml"/>` 引用机器人，
  通过 `<compiler meshdir="meshes"/>` 引用网格 —— **保持原来的相对路径**，
  所以整个 `g1/` 目录可以原样搬走。
- `meshes/` 里既有 G1 本体的 STL，也有 YCB 物体（糖盒、番茄罐头、苹果、马克杯…）的
  obj + 纹理。缺了它场景加载会直接报 `Error opening file`。

## 场景 ↔ 任务对照

| 任务 | 场景文件 | 场景里的目标物体 / 容器 |
|---|---|---|
| apple | `scene_pickplace.xml` | `apple` → `box` |
| carton | `scene_43dof_two_table.xml` | `sugar_box` → `box`（另一张桌子） |
| red cup | `scene_43dof.xml` | `mug` → `drawer`（柜子抽屉） |
| handover | `scene_43dof_handover.xml` | `sugar_box` → `tgt_box`（经 `src_tray`） |
| task4（地上捡罐子） | `scene_43dof_bend_and_pick.xml` | `can` → `table_target` |

每个场景自带 5 个相机：`ego_view`、`head_camera`、`side_view`、`top_down`，
部分还有 `front_view` / `task_view` / `q3_view`。
**policy 用的是 `head_camera`**（sim 里发布成 `ego_view` 话题）。

## 初始位姿文件（三种格式）

### A. 单帧初始位姿 —— `*_initial_state.json` / `*_init_state.json` / `*_identity_state.json`

```json
{
  "format_version": 1,
  "source": "<录制 parquet 路径>",
  "source_sha256": "...",
  "episode_index": 0,
  "frame_index": 0,
  "base_position": [0.0, 0.0, 0.793],
  "base_quaternion_wxyz": [...],
  "recorded_heading_init_quaternion_wxyz": [...],
  "joint_positions": [ ... 43 维 ... ]
}
```

`joint_positions` 是 **43 维**（29 本体 + 14 手），顺序与数据集 `observation.state` 一致：
`左腿6 右腿6 腰3 左臂7 左手7 右臂7 右手7`。

用法：

```python
import json, mujoco

m = mujoco.MjModel.from_xml_path("g1/scene_43dof_two_table.xml")
d = mujoco.MjData(m)
s = json.load(open("initial_states/handover_episode0_initial_state.json"))

d.qpos[0:3] = s["base_position"]
d.qpos[3:7] = s["base_quaternion_wxyz"]
body = [0,1,2,3,4,5, 6,7,8,9,10,11, 12,13,14, 15,16,17,18,19,20,21, 29,30,31,32,33,34,35]
hand = [22,23,24,25,26,27,28, 36,37,38,39,40,41,42]
jp = s["joint_positions"]
d.qpos[7:36] = [jp[i] for i in body]
d.qpos[36:50] = [jp[i] for i in hand]
mujoco.mj_forward(m, d)
```

注意那个索引置换：**数据集的 43 维顺序不是 MJCF 的关节顺序**
（MJCF 把左手插在左臂和右臂之间）。这是复现时最容易错的地方之一。

### B. warmstart 序列 —— `*_warmstart*.json`

评测在切到 POSE 模式前会先播一段"planner 预热"，这批文件就是那段序列：

```json
{
  "format_version": 1,
  "source": "2026-07-14_put_carton_into_box_clean/.../episode_0",
  "episode_index": 0,
  "source_frame_indices": [0, 1, 2, 3, 4],
  "collection_frequency_hz": 50,
  "stream_mode": "PLANNER_VR_3PT",
  "target_observation_state": [ ... 43 维 ... ],
  "frames": [ { "planner_mode": 0, "planner_movement": [...], "planner_facing": [...] }, ... ]
}
```

### C. task4 初始 motion token —— `task4/*.json`

task4 的随机初始状态池用的是另一种协议：`initial_motion_token`（64 维）+
`recorded_init_base_quat_wxyz` + `source_frame_index`，对应评测里的
`task4-random-clean34-initial-token-v1`。

## 验证

```bash
cd reproduce_scenes
python verify_scenes.py          # 打印每个场景的 nq/ngeom/相机
python verify_scenes.py --render # 额外渲染每场景一张 png
```

预期输出：

```
scene_43dof.xml             nq=72  ngeom=124  cams=['ego_view','side_view','top_down','task_view','head_camera']
scene_pickplace.xml         nq=78  ngeom=118  cams=['ego_view','agentview','side_view','top_down','head_camera']
scene_43dof_two_table.xml   nq=78  ngeom=123  cams=['ego_view','side_view','top_down','tableA_view','head_camera']
scene_43dof_handover.xml    nq=71  ngeom=122  cams=['ego_view','front_view','top_down','q3_view','head_camera']
scene_43dof_bend_and_pick.xml nq=57 ngeom=112 cams=['ego_view','side_view','top_down','q3_view','head_camera']
```
