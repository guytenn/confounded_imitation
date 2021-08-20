from src.rllib.env.wrappers.model_vector_env import model_vector_env as mve
from src.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    old="src.rllib.env.model_vector_env.model_vector_env",
    new="src.rllib.env.wrappers.model_vector_env.model_vector_env",
    error=False,
)

model_vector_env = mve
