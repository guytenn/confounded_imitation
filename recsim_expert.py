from ray.rllib.env.wrappers.recsim_wrapper import make_recsim_env
import numpy as np
from tqdm import tqdm


def compute_action(env):
    inner_env = env.unwrapped.environment
    all_slates = list(inner_env._current_documents.keys())
    documents = inner_env._candidate_set.get_documents(all_slates)
    user_state = inner_env.user_model._user_state
    doc_obs = [doc.create_observation() for doc in documents]
    expected_utility = [user_state.score_document(doc_obs[i]) for i in range(len(doc_obs))]
    quality = [doc.quality for doc in documents]
    doc_scores = [user_state.user_quality_factor * expected_utility[i] + user_state.document_quality_factor * quality[i]
                  for i in range(len(documents))]

    action = np.array(doc_scores).argsort()[-2:][::-1]
    return action

# env = make_recsim_env({})
#
# inner_env = env.unwrapped.environment
#
# tot_rewards = []
# for epoch in tqdm(range(100)):
#     rewards = []
#     done = False
#     s = env.reset()
#     while not done:
#         all_slates = list(inner_env._current_documents.keys())
#         documents = inner_env._candidate_set.get_documents(all_slates)
#         user_state = inner_env.user_model._user_state
#         doc_obs = [doc.create_observation() for doc in documents]
#         expected_utility = [user_state.score_document(doc_obs[i]) for i in range(len(doc_obs))]
#         quality = [doc.quality for doc in documents]
#         doc_scores = [user_state.user_quality_factor * expected_utility[i] + user_state.document_quality_factor * quality[i] for i in range(len(documents))]
#
#         action = np.array(doc_scores).argsort()[-2:][::-1]
#         s, r, done, i = env.step(action)
#         # s, r, done, i = env.step(env.action_space.sample())
#         rewards.append(r)
#     tot_rewards.append(np.sum(rewards))
# print(np.mean(tot_rewards), np.std(tot_rewards))