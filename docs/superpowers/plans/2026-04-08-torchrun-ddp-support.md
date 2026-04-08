# torchrun DDP Support for RecBole

**Goal:** 让 RecBole 支持 `torchrun` 启动多 GPU 训练，同时保留原有 `mp.spawn` 方式。

**Architecture:** 在 `configurator.py` 检测 torchrun 环境变量（`LOCAL_RANK` 等），走 `init_method="env://"` 路径；在 `quick_start.py` 检测到 torchrun 时跳过 `mp.spawn` 直接调用 `run_recbole()`。

**Tech Stack:** PyTorch DDP, torchrun, torch.distributed

---

### Task 1: 修改 `configurator.py` 的 DDP 初始化逻辑

**Files:**
- Modify: `/rmbs_1/RecBole/recbole/config/configurator.py:495-515`

- [ ] **Step 1: 在 DDP 初始化分支新增 torchrun 检测**

将 `configurator.py` 第 497-511 行的 `torch.distributed.init_process_group` 调用改为：

```python
# Check if launched by torchrun (env variables set automatically)
if "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    # torchrun path
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    self.final_config_dict["device"] = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    self.final_config_dict["single_spec"] = False
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    self.final_config_dict["local_rank"] = int(os.environ["LOCAL_RANK"])
    self.final_config_dict["world_size"] = int(os.environ["WORLD_SIZE"])
    self.final_config_dict["rank"] = int(os.environ["RANK"])
    if int(os.environ["LOCAL_RANK"]) != 0:
        self.final_config_dict["state"] = "error"
        self.final_config_dict["show_progress"] = False
        self.final_config_dict["verbose"] = False
else:
    # Original mp.spawn path (TCP init)
    assert len(gpu_id.split(",")) >= self.final_config_dict["nproc"]
    torch.distributed.init_process_group(
        backend="nccl",
        rank=self.final_config_dict["local_rank"]
        + self.final_config_dict["offset"],
        world_size=self.final_config_dict["world_size"],
        init_method="tcp://"
        + self.final_config_dict["ip"]
        + ":"
        + str(self.final_config_dict["port"]),
    )
    self.final_config_dict["device"] = torch.device(
        "cuda", self.final_config_dict["local_rank"]
    )
    self.final_config_dict["single_spec"] = False
    torch.cuda.set_device(self.final_config_dict["local_rank"])
    if self.final_config_dict["local_rank"] != 0:
        self.final_config_dict["state"] = "error"
        self.final_config_dict["show_progress"] = False
        self.final_config_dict["verbose"] = False
```

需要在文件顶部 `import os`（如果还没有的话）。

---

### Task 2: 修改 `quick_start.py` 的 run() 函数跳过 mp.spawn

**Files:**
- Modify: `/rmbs_1/RecBole/recbole/quick_start/quick_start.py:39-92`

- [ ] **Step 1: 在 `run()` 函数开头检测 torchrun 环境变量**

在 `run()` 函数的第一行 `if nproc == 1 and world_size <= 0:` 之前插入：

```python
# Detect if launched by torchrun
if "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    # torchrun: each process is a separate entry, just call run_recbole() directly
    res = run_recbole(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
        saved=saved,
    )
    return res
```

并确认文件顶部有 `import os`。

---

### Task 3: 创建 `run_ddp.sh` 示例脚本

**Files:**
- Create: `/rmbs_1/RecBole/run_ddp.sh`

- [ ] **Step 1: 编写 torchrun 启动脚本**

```bash
#!/bin/bash
# Multi-GPU training using torchrun

torchrun \
    --nproc_per_node=$1 \
    --master_port=29500 \
    run_recbole.py \
    --model=$2 \
    --dataset=$3 \
    --config_files="$4"
```

用法：`bash run_ddp.sh 2 LightGCN amazon`

---

### Task 4: 验证

- [ ] 单 GPU（不变）：`python run_recbole.py --model BPR --dataset ml-100k`
- [ ] mp.spawn（不变）：`python run_recbole.py --model LightGCN --dataset amazon --nproc=2`
- [ ] torchrun（新增）：`torchrun --nproc_per_node=2 run_recbole.py --model LightGCN --dataset amazon`

