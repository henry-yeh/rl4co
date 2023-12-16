from rl4co.envs import TSPEnv
from rl4co.models import AttentionModel_NAR
from rl4co.utils import RL4COTrainer

# Environment, Model, and Lightning Module
env = TSPEnv(num_loc=20)
model = AttentionModel_NAR(
    env,
    baseline="rollout",
    train_data_size=100_000,
    test_data_size=10_000,
    optimizer_kwargs={"lr": 1e-6},
    policy_kwargs={"n_encoder_layers": 6},
)

# Trainer
trainer = RL4COTrainer(max_epochs=20)

# Fit the model
trainer.fit(model)

# Test the model
trainer.test(model)
