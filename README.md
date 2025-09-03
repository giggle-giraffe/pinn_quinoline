# Physics-informed Neural Network for Predicting Reaction Kinetics of Quinoline Hydrodenitrogenation

An implementation of Physics-Informed Neural Networks (PINNs) for modeling and predicting reaction kinetics of quinoline hydrodenitrogenation. This project combines deep learning with chemical engineering principles to predict reaction outcomes and obtain kinetic parameters.

## 🔬 Features

- **Advanced PINN Architecture**: Custom neural network models that incorporate physical laws and chemical kinetics
- **Curriculum Learning**: Multi-stage training with adaptive loss weighting strategies
- **GradNorm Optimization**: Advanced multi-objective optimization using gradient normalization
- **Physics Constraints**: Built-in mass conservation and thermodynamic constraints
- **Adaptive Training**: Dynamic parameter freezing and learning rate scheduling
- **Comprehensive Logging**: Detailed training metrics and visualization tools

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zcbipt/pinn_quinoline.git
   cd pinn_quinoline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place training data in the `data/` directory
   - Update configuration files in `config/` as needed

## 📁 Project Structure

```
pinn_quinoline_os_dev/
├── src/                    # Source code
│   ├── train.py           # Training logic with curriculum learning
│   ├── model_pinn.py      # PINN model architecture
│   ├── loss.py            # Physics-informed loss functions
│   └── ...
├── config/                # Configuration files
│   ├── train_pinn_quinoline.yaml
│   ├── predict_pinn_quinoline.yaml
│   └── validate_pinn_quinoline.yaml
├── data/                  # Training and validation data
├── model/                 # Saved model checkpoints
└── output/               # Results and visualizations
```

## 🔧 Configuration

The project uses YAML configuration files to control training parameters:

- **Training Mode**: `config/train_pinn_quinoline.yaml` - For model training
- **Prediction Mode**: `config/predict_pinn_quinoline.yaml` - For inference
- **Validation Mode**: `config/validate_pinn_quinoline.yaml` - For model validation

Key configuration sections:
- `curriculum_config`: Curriculum learning and adaptive weighting
- `model_config`: Neural network architecture parameters
- `training_config`: Optimization and training hyperparameters

## 📊 Key Algorithms

### Physics-Informed Neural Networks (PINNs)
Our implementation combines:
- **Data-driven learning**: Learning from experimental quinoline synthesis data
- **Physics-based constraints**: Incorporating chemical kinetics and thermodynamics
- **Mass conservation**: Ensuring physical consistency in predictions

### Curriculum Learning
- **Multi-stage training**: Progressive complexity increase
- **Adaptive loss weighting**: Dynamic balancing of different loss components
- **Parameter freezing**: Strategic parameter updates during training phases

### GradNorm Optimization
- **Multi-objective balancing**: Automatic loss weight adjustment
- **Gradient magnitude normalization**: Balanced learning across objectives
- **Adaptive learning rates**: Dynamic optimization strategy

## 📈 Results and Visualization

The training process generates comprehensive visualizations:
- Loss evolution curves
- Weight adaptation dynamics
- Gradient magnitude analysis
- Physics constraint satisfaction metrics

Results are saved to the `output/` directory with detailed plots and metrics.

## 🤝 Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs and requesting features
- Setting up the development environment
- Code style and testing requirements
- Submitting pull requests

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2025pinn_quinoline,
  title={Physics-informed Neural Network for Predicting Reaction Kinetics of Quinoline Hydrodenitrogenation},
  author={Shihao Wang, Tao Li, Yuelin Xu, Wenbin Chen, Xiaoqian Dang, Wei Zhang, Chen Zhang, Cuiqing Li, Yong Luo, Feng Liu and Mingfeng Li},
  journal={Under Review},
  year={2025},
  note={Manuscript submitted for publication}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Tao Li**: tao.li@gigglegiraffe.com
- **Xiaoqian Dang**: dangxiaoqian@gigglegiraffe.com
- **Chen Zhang**: zhangc@bipt.edu.cn

For questions about the methodology or collaboration opportunities, please open an issue or contact the maintainers directly.

## 🔗 Related Work

This project builds upon advances in:
- Physics-Informed Neural Networks (PINNs)
- Multi-objective optimization in deep learning
- Chemical process modeling and optimization
- Curriculum learning strategies

---

**Note**: This is an active research project. While we strive for stability, the API may evolve as we incorporate new features and improvements.