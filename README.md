# Specialized Deep Residual Policy Safe Reinforcement Learning (CoL-SDRPRL)

## Overview
This repository contains the implementation of the **Specialized Deep Residual Policy Safe Reinforcement Learning (CoL-SDRPRL)** framework for process control in continuous state-action spaces. The framework integrates:
- **Residual Policy Learning (RPL)** for hybrid DRL and conventional controller collaboration.
- **Cycle of Learning (CoL)** for improved policy initialization and stability.
- **Specialized Reinforcement Learning Agent (SRLA)** with an extended **Input-Output Hidden Markov Model (IOHMM)** for targeted learning in abnormal states.

The framework is validated on the **Tennessee Eastman Process (TEP)** control problem.

## Features
- Autonomous synchronization between controllers.
- Targeted DRL activation for critical states.
- Ablation study to assess the impact of RPL, CoL, and SRLA.
- Comparative evaluation with baselines and recent safe-RL approaches.
- Scalability analysis for increasing system complexity.

## Installation
To set up the environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/CoL-SDRPRL.git
cd CoL-SDRPRL

# Create a virtual environment (optional)
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Running the Training and Evaluation Pipeline
```bash
python CoL-SDRPRL.py
```

## Methodology
Our approach addresses the challenges of conventional and deep reinforcement learning (DRL)-based controllers by integrating specialized learning mechanisms:
1. **Residual Policy Learning (RPL):** Enhances a conventional controller by learning a corrective DRL policy.
2. **Cycle of Learning (CoL):** Reduces distributional drift and stabilizes training by initializing DRL from expert demonstrations.
3. **Specialized Reinforcement Learning Agent (SRLA):** Uses an extended **IOHMM** for detecting critical states and activating DRL selectively.


## Results
- **Improved stability**: The hybrid approach prevents instability observed in standalone DRL methods.
- **Efficient learning**: The CoL strategy accelerates convergence compared to naive reinforcement learning.
- **Safety validation**: The specialized activation reduces risk in critical states.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or collaborations, reach out to:
- **Ammar N. Abbas** - ammar.abbas@eu4m.eu

