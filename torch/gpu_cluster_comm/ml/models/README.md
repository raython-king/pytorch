# 预训练模型目录

本目录用于存放训练好的ML模型权重。

## 模型文件命名规范

- `time_predictor_best.pt` - 通讯时间预测器的最佳模型
- `bandwidth_predictor_best.pt` - 带宽预测器的最佳模型
- `congestion_predictor_best.pt` - 拥塞预测器的最佳模型
- `algorithm_selector_best.pt` - 算法选择器的最佳模型
- `parameter_optimizer_best.pt` - 参数优化器的最佳模型
- `rl_agent_final.pt` - RL智能体的最终模型

## 模型文件格式

每个模型文件包含：
- `model_state_dict`: 模型权重
- `model_config`: 模型配置
- `training_info`: 训练信息（可选）

## 使用示例

```python
from torch.gpu_cluster_comm.ml import create_fast_inference

# 加载所有模型
inference_engine = create_fast_inference(model_dir='./models')

# 进行推理
prediction = inference_engine.predict_time(features)
```

## 模型版本控制

建议使用版本号命名：
- `time_predictor_v1.0.pt`
- `time_predictor_v1.1.pt`

## 模型性能基准

| 模型 | 参数量 | 推理时间 | 准确率/误差 |
|------|--------|----------|-------------|
| TimePredictor | 500K | <0.5ms | MAPE<15% |
| BandwidthPredictor | 400K | <0.8ms | RMSE<10GB/s |
| AlgorithmSelector | 600K | <0.3ms | Acc>85% |
| ParameterOptimizer | 450K | <0.4ms | - |
| RLAgent | 800K | <1.0ms | - |
