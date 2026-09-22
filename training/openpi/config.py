"""OpenPI 训练配置；可用配置列表见 `_CONFIGS`。"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.nero_bimanual_policy as nero_bimanual_policy
import openpi.policies.nero_policy as nero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# 规避 tyro 直接使用 nnx.filterlib.Filter 时的问题。
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """定义构建数据管线所需资产（例如归一化统计）的位置。

    这些资产会复制到 checkpoint 的 `assets/asset_id` 目录中。

    也可从其他 checkpoint（例如基础模型）或集中存储位置加载资产。例如，微调时可用
    下列方式从基础模型 checkpoint 加载 Trossen 机器人的归一化统计：

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # 资产目录。未指定时使用配置中的 assets_dirs；可用于从其他 checkpoint 或集中
    # 存储位置加载资产。
    assets_dir: str | None = None

    # 资产 ID。未指定时使用 repo id，以便引用不同机器人平台的资产。
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot 仓库 ID；为 None 时创建伪数据。
    repo_id: str | None = None
    # 资产目录中存放数据资产的子目录。
    asset_id: str | None = None
    # 预计算归一化统计；为 None 时不执行归一化。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 将数据集专用输入结构整理为数据变换所需的统一结构。
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 数据变换通常包含机器人专用变换，在归一化之前应用。归一化数据定义见
    # `model.Observation` 和 `model.Actions`。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型专用变换，在归一化之后应用。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 为 true 时使用分位数归一化，否则使用普通 z-score 归一化。
    use_quantile_norm: bool = False

    # 数据加载器用于生成 action 序列的字段名。序列长度由模型配置的 `action_horizon`
    # 定义；LeRobot 数据集使用其他 action 字段时需要调整。
    action_sequence_keys: Sequence[str] = ("actions",)

    # 为 true 时使用 LeRobot 数据集的 task 字段定义 prompt。
    prompt_from_task: bool = False

    # 仅供 RLDS 数据加载器使用，目前只用于 DROID。
    rlds_data_dir: str | None = None
    # DROID 数据集的 action 空间。
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # 采样数据集列表：名称、版本、权重及可选的 filter_dict_path。
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建变换组。"""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """为标准 π0 模型创建变换。"""

    # 指定后作为模型使用的默认 prompt。
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # LeRobot 仓库 ID。
    repo_id: str = tyro.MISSING
    # 定义资产加载方式。
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # 工厂函数用于更新的基础配置。
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建数据配置。"""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # 数据变换工厂。
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # 模型变换工厂。
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # 为 true 时，在送入模型前将关节维度转换为相对当前状态的 delta；夹爪保持绝对值。
    use_delta_joint_actions: bool = True
    # 输入中没有 `prompt` 字段时注入该默认值。
    default_prompt: str | None = None
    # 为 true 时，将关节和夹爪值从标准 Aloha 空间转换到训练基础模型所用的 π 内部
    # 运行时空间。使用标准 Aloha 数据时应启用。
    adapt_to_pi: bool = True

    # 字段重组变换。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 从数据集读取 action 序列时使用的字段名。
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # repack 只作用于数据集，不用于推理。它把数据集字段映射成推理环境传给策略
        # 服务器的字段结构；接入新数据集时应按实际推理输入修改下列映射。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 数据变换同时用于训练数据和推理。inputs 负责模型输入，outputs 负责推理输出；
        # 具体实现位于 `libero_policy.py`，接入自定义数据集时替换为对应变换。
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # π0 使用相对 action chunk 首帧状态的 delta action。绝对关节目标需要在此转换，
        # 夹爪维度始终保持绝对值。Libero 原始 action 已是 delta，因此无需额外转换。

        # LIBERO 已使用 delta action；该开关仅兼容曾额外应用 delta 变换的旧 π0 checkpoint。
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 模型变换包括 prompt 和 action 目标的 token 化，自定义数据集通常无需修改。
        model_transforms = ModelTransformFactory()(model_config)

        # 返回训练与推理共用的全部数据变换。
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotNeroDataConfig(DataConfigFactory):
    """NERO 关节位置示教数据管线。"""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/external_image": "observation.images.external",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                nero_policy.NeroInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                nero_policy.NeroOutputs(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotNeroBimanualDataConfig(DataConfigFactory):
    """双 NERO 机械臂与三路 RGB 相机数据管线。"""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/world_image": "observation.images.world",
                        "observation/left_wrist_image": "observation.images.left_wrist",
                        "observation/right_wrist_image": "observation.images.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # 关节 action 使用相对值，两侧夹爪保持绝对值。
        delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                nero_bimanual_policy.NeroBimanualInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                nero_bimanual_policy.NeroBimanualOutputs(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # 过滤配置可传入字典路径，把 episode 映射到需要保留的时间步区间 `(start, end)`。
    # episode 使用 RLDS 元数据中的 `recording_folderpath--file_path` 唯一标识。

    # 采样数据集列表：名称、版本、权重及可选的 filter_dict_path。
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # 数据加载器返回绝对关节位置 action，训练前转换为 delta action。
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # 此处假设 action 为关节速度，因此不能再应用 delta 变换。
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 配置名称，必须唯一，用于引用该配置。
    name: tyro.conf.Suppress[str]
    # 项目名称。
    project_name: str = "openpi"
    # 实验名称，用于命名元数据和 checkpoint 目录。
    exp_name: str = tyro.MISSING

    # 模型配置。action_dim、action_horizon 和 max_token_len 等公共字段见
    # BaseModelConfig；具体模型可继承并增加字段。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # 模型初始化后可用权重加载器从磁盘加载完整或部分权重。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 可选的 PyTorch checkpoint 权重路径。
    pytorch_weight_path: str | None = None

    # PyTorch 训练精度。
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定需要冻结的权重。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 定义训练数据。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 配置资产（例如归一化统计）的根目录。
    assets_base_dir: str = "./assets"
    # checkpoint 根目录。
    checkpoint_base_dir: str = "./checkpoints"

    # 训练随机数生成器使用的种子。
    seed: int = 42
    # 全局 batch size。
    batch_size: int = 32
    # 数据加载器 worker 数量；提高该值可加速加载，但会增加内存和 CPU 占用。
    num_workers: int = 2
    # 训练步数（batch 数）。
    num_train_steps: int = 30_000

    # 训练指标日志间隔，单位为 step。
    log_interval: int = 100
    # checkpoint 保存间隔，单位为 step。
    save_interval: int = 1000
    # 设置后，满足 `step % keep_period == 0` 的 checkpoint 不会被删除。
    keep_period: int | None = 5000

    # 为 true 时覆盖已有 checkpoint 目录。
    overwrite: bool = False
    # 为 true 时从最后一个 checkpoint 恢复训练。
    resume: bool = False

    # 为 true 时启用 wandb 日志。
    wandb_enabled: bool = True

    # 传递给策略服务器的元数据。
    policy_metadata: dict[str, Any] | None = None

    # 大于 1 时启用 FSDP，并在指定数量的设备上分片。这样可降低单设备显存占用，但
    # 可能降低训练速度。例如 4 个设备、FSDP 设备数为 2 时，模型在每 2 个设备间
    # 分片，并在两组设备之间执行数据并行。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """返回该配置的资产目录。"""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """返回该配置的 checkpoint 目录。"""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """返回可训练参数过滤器。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# 代码中需要按名称读取配置时使用 `get_config`。
_CONFIGS = [
    #
    # Aloha 推理配置。
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # DROID 推理配置。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Libero 微调配置。
    #
    # 这些配置定义基础模型在自定义数据集上的微调超参数，包括数据集、基础 checkpoint、
    # 训练步数和学习率。接入新数据集时可复制该配置并修改数据集名称和数据变换。
    TrainConfig(
        # 名称应准确反映模型和数据集。
        name="pi0_libero",
        # 此处定义模型配置。示例使用 π0 做全量微调；后续示例展示低显存 LoRA 微调
        # 以及 π0-FAST 架构。
        model=pi0_config.Pi0Config(),
        # 此处定义训练数据集。示例使用 Libero；接入自定义数据集时修改 repo_id，并让
        # DataConfig 使用对应的数据配置。
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # 决定是否从 LeRobot 数据集的 `task` 字段读取任务 prompt。启用后，prompt
                # 会写入输入字典的 `prompt` 字段，推荐启用。
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # 指定用于初始化模型的预训练 checkpoint，必须与上方模型配置匹配。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 其余学习率、训练步数等超参数见基础 TrainConfig 类。
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # 加载 π0 模型并进行 LoRA 微调的示例。
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # freeze filter 定义训练时冻结的参数。模型配置提供 LoRA 默认过滤器，必须确保
        # 它与上方选择的模型配置一致。
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # LoRA 微调时关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # π0-FAST 全量微调示例。action_dim 和 action_horizon 必须匹配数据集，后者等于
        # action chunk 长度。max_token_len 是模型可处理的最大非图像 token 数，包括
        # prompt、机器人本体状态和 FAST action token。过小会截断序列，过大则浪费
        # 内存；单臂可从约 180 开始，双臂可从约 250 开始，再根据警告调整。
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # 此处加载 π0-FAST 基础模型 checkpoint。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # π0-FAST LoRA 微调示例；action_dim、action_horizon 和 max_token_len 见上文。
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # 提取 LoRA 冻结参数过滤器时，同样必须与上方模型配置保持一致。
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # LoRA 微调时关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_nero_smoke",
        exp_name="three_episode_smoke",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_pi05_smoke_v21",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=3,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_pick_water_bottle",
        exp_name="lora_10000_20260720",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_pick_water_bottle_50_v2",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=10_000,
        log_interval=10,
        save_interval=5_000,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_pick_bottle_calibrated_v2",
        exp_name="lora_10000_calibrated_20260721",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_pick_bottle_calibrated_50_v2",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=10_000,
        log_interval=10,
        save_interval=5_000,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_pick_bottle_9points_3bottles_v3",
        exp_name="lora_10000_9points_3bottles_20260724",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_pick_bottle_9points_3bottles_v3",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=10_000,
        log_interval=10,
        save_interval=5_000,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_pick_bottle_single_80_clean_v1",
        exp_name="lora_12000_single80_clean_20260727",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_pick_bottle_single_bottle_80_v3_clean_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=12_000,
        log_interval=10,
        save_interval=12_000,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_pick_bottle_single_80_clean_6h_v1",
        exp_name="lora_48000_single80_clean_6h_20260727",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_pick_bottle_single_bottle_80_v3_clean_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=48_000,
        log_interval=10,
        save_interval=48_000,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_pick_bottle_clean60_v1",
        exp_name="lora_12000_clean60_20260729",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_pick_bottle_clean_full_demo_60_v2",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=12_000,
        log_interval=10,
        save_interval=12_000,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_towel_bimanual_50_v1",
        exp_name="lora_20000_towel_bimanual_50_20260817",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroBimanualDataConfig(
            repo_id="local/nero_towel_fold_bimanual_50_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=20_000,
        log_interval=10,
        save_interval=4_000,
        keep_period=4_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_towel_fold_stage_50_h24_v1",
        exp_name="lora_20000_towel_fold_stage_50_h24_20260819",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroBimanualDataConfig(
            repo_id="local/nero_towel_fold_stage_50_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=20_000,
        log_interval=10,
        save_interval=4_000,
        keep_period=4_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_towel_stage23_fold_50_command_h24_v1",
        exp_name="lora_20000_towel_stage23_fold_50_command_h24_v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroBimanualDataConfig(
            repo_id="local/nero_towel_stage23_fold_50_command_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=20_000,
        log_interval=10,
        save_interval=4_000,
        keep_period=4_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_towel_stage23_only_50_event35_h24_v1",
        exp_name="lora_10000_towel_stage23_command_event35_h24_20260821",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroBimanualDataConfig(
            repo_id="local/nero_towel_stage23_fold_50_command_event35_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=10_000,
        log_interval=10,
        save_interval=2_000,
        keep_period=2_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_towel_multistage_99_command_event35_h24_v1",
        exp_name="lora_20000_towel_multistage_99_command_event35_h24_v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroBimanualDataConfig(
            repo_id="local/nero_towel_multistage_99_command_event35_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=20_000,
        log_interval=10,
        save_interval=4_000,
        keep_period=4_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_towel_multistage_99_next_feedback_event4_h24_v1",
        exp_name="lora_20000_towel_multistage_99_next_feedback_event4_h24_v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroBimanualDataConfig(
            repo_id="local/nero_towel_multistage_99_next_feedback_event4_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=20_000,
        log_interval=10,
        save_interval=4_000,
        keep_period=4_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_towel_fullflow_70_command_event4_h24_v1",
        exp_name="lora_40000_towel_fullflow_70_command_event4_h24_20260822",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroBimanualDataConfig(
            repo_id="local/nero_towel_fullflow_70_command_event4_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=40_000,
        log_interval=10,
        save_interval=4_000,
        keep_period=4_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1",
        exp_name="lora_40000_towel_fullflow_70_next_feedback_event4_h24_20260822",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroBimanualDataConfig(
            repo_id="local/nero_towel_fullflow_70_next_feedback_event4_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=40_000,
        log_interval=10,
        save_interval=4_000,
        keep_period=4_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    TrainConfig(
        name="pi05_nero_intervention_13_v1",
        exp_name="lora_4000_intervention13_from_single80_12k_20260728",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_intervention_pick_bottle_13_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        num_workers=0,
        num_train_steps=4_000,
        log_interval=10,
        save_interval=4_000,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/nero_training/checkpoints/"
            "pi05_nero_pick_bottle_single_80_clean_v1/"
            "lora_12000_single80_clean_20260727/11999/params"
        ),
    ),
    #
    # Aloha 微调配置。
    #
    # 自定义 LeRobot 数据集训练示例。Aloha 数据转换与训练说明见
    # `examples/aloha_real/README.md`。
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # DROID 微调配置。
    #
    TrainConfig(
        # 在完整 DROID 数据集上微调 π0-FAST-base；使用 RLDS 加载大型数据集。
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 设置为 DROID RLDS 数据集路径，即 `droid` 目录的父目录。
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # 在完整 DROID 数据集上微调 π0.5；使用 RLDS 加载大型数据集。
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 设置为 DROID RLDS 数据集路径，即 `droid` 目录的父目录。
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # 在较小的自定义 DROID 数据集上微调 π0.5-DROID。此处使用 LeRobot 格式；
        # 转换少于数十小时的自定义数据见 `examples/droid/convert_droid_data_to_lerobot.py`。
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # 替换为自定义 DROID LeRobot 数据集 repo id。
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # 重要：微调时必须复用原始 DROID 归一化统计。
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA 仿真配置，用于展示简单仿真环境中的训练方法。
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # 调试配置。
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena 与 PolaRiS 配置。
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """按名称读取配置。"""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
