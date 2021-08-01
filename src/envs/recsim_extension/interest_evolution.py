# coding=utf-8
# coding=utf-8
# Copyright 2019 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes to represent the interest evolution documents and users."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
import gin.tf
from gym import spaces
import numpy as np
from recsim import choice_model
from recsim import document
from recsim import user
from recsim import utils
from recsim.simulator import environment
from src.rllib_extensions import recsim_gym

FLAGS = flags.FLAGS


class IEvResponse(user.AbstractResponse):
  """Class to represent a user's response to a video.

  Attributes:
    clicked: A boolean indicating whether the video was clicked.
    watch_time: A float for fraction of the video watched.
    liked: A boolean indicating whether the video was liked.
    quality: A float indicating the quality of the video.
    cluster_id: A integer representing the cluster ID of the video.
  """

  # The min quality score.
  MIN_QUALITY_SCORE = -100
  # The max quality score.
  MAX_QUALITY_SCORE = 100

  def __init__(self,
               clicked=False,
               watch_time=0.0,
               liked=False,
               quality=0.0,
               cluster_id=0.0):
    """Creates a new user response for a video.

    Args:
      clicked: A boolean indicating whether the video was clicked
      watch_time: A float for fraction of the video watched
      liked: A boolean indicating whether the video was liked
      quality: A float for document quality
      cluster_id: a integer for the cluster ID of the document.
    """
    self.clicked = clicked
    self.watch_time = watch_time
    self.liked = liked
    self.quality = quality
    self.cluster_id = cluster_id

  def create_observation(self):
    return {
        'click': int(self.clicked),
        'watch_time': np.array(self.watch_time),
        'liked': int(self.liked),
        'quality': np.array(self.quality),
        'cluster_id': int(self.cluster_id)
    }

  @classmethod
  def response_space(cls):
    # `clicked` feature range is [0, 1]
    # `watch_time` feature range is [0, MAX_VIDEO_LENGTH]
    # `liked` feature range is [0, 1]
    # `quality`: the quality of the document and range is specified by
    #    [MIN_QUALITY_SCORE, MAX_QUALITY_SCORE].
    # `cluster_id`: the cluster the document belongs to and its range is
    # .  [0, IEvVideo.NUM_FEATURES].
    return spaces.Dict({
        'click':
            spaces.Discrete(2),
        'watch_time':
            spaces.Box(
                low=0.0,
                high=IEvVideo.MAX_VIDEO_LENGTH,
                shape=tuple(),
                dtype=np.float32),
        'liked':
            spaces.Discrete(2),
        'quality':
            spaces.Box(
                low=cls.MIN_QUALITY_SCORE,
                high=cls.MAX_QUALITY_SCORE,
                shape=tuple(),
                dtype=np.float32),
        'cluster_id':
            spaces.Discrete(IEvVideo.NUM_FEATURES)
    })


class IEvVideo(document.AbstractDocument):
  """Class to represent a interest evolution Video.

  Attributes:
    features: A numpy array that stores video features.
    cluster_id: An integer that represents.
    video_length : A float for video length.
    quality: a float the represents document quality.
  """

  # The maximum length of videos.
  MAX_VIDEO_LENGTH = 100.0

  # The number of features to represent each video.
  NUM_FEATURES = 20

  def __init__(self,
               doc_id,
               features,
               cluster_id=None,
               video_length=None,
               quality=None):
    """Generates a random set of features for this interest evolution Video."""

    # Document features (i.e. distribution over topics)
    self.features = features

    # Cluster ID
    self.cluster_id = cluster_id

    # Length of video
    self.video_length = video_length

    # Document quality (i.e. trashiness/nutritiousness)
    self.quality = quality

    # doc_id is an integer representing the unique ID of this document
    super(IEvVideo, self).__init__(doc_id)

  def create_observation(self):
    """Returns observable properties of this document as a float array."""
    return self.features

  @classmethod
  def observation_space(cls):
    return spaces.Box(
        shape=(cls.NUM_FEATURES,), dtype=np.float32, low=-1.0, high=1.0)


class IEvVideoSampler(document.AbstractDocumentSampler):
  """Class to sample interest_evolution videos."""

  def __init__(self,
               doc_ctor=IEvVideo,
               min_feature_value=-1.0,
               max_feature_value=1.0,
               video_length_mean=4.3,
               video_length_std=1.0,
               **kwargs):
    """Creates a new interest evolution video sampler.

    Args:
      doc_ctor: A class/constructor for the type of videos that will be sampled
        by this sampler.
      min_feature_value: A float for the min feature value.
      max_feature_value: A float for the max feature value.
      video_length_mean: A float for the mean of the video length.
      video_length_std: A float for the std deviation of video length.
      **kwargs: other keyword parameters for the video sampler.
    """
    super(IEvVideoSampler, self).__init__(doc_ctor, **kwargs)
    self._doc_count = 0
    self._min_feature_value = min_feature_value
    self._max_feature_value = max_feature_value
    self._video_length_mean = video_length_mean
    self._video_length_std = video_length_std

  def sample_document(self):
    doc_features = {}
    doc_features['doc_id'] = self._doc_count
    # For now, assume the document properties are uniform random.
    # It will probably make more sense to concentrate the interests around a few
    # (e.g. 5?) categories or have a more sophisticated generative process?
    doc_features['features'] = self._rng.uniform(
        self._min_feature_value, self._max_feature_value,
        self.get_doc_ctor().NUM_FEATURES)
    doc_features['video_length'] = min(
        self._rng.normal(self._video_length_mean, self._video_length_std),
        self.get_doc_ctor().MAX_VIDEO_LENGTH)
    doc_features['quality'] = 1.0
    self._doc_count += 1
    return self._doc_ctor(**doc_features)


class UtilityModelVideoSampler(document.AbstractDocumentSampler):
  """Class that samples videos for utility model experiment."""

  def __init__(self,
               doc_ctor=IEvVideo,
               min_utility=-3.0,
               max_utility=3.0,
               video_length=4.0,
               **kwargs):
    """Creates a new utility model video sampler.

    Args:
      doc_ctor: A class/constructor for the type of videos that will be sampled
        by this sampler.
      min_utility: A float for the min utility score.
      max_utility: A float for the max utility score.
      video_length: A float for the video_length in minutes.
      **kwargs: other keyword parameters for the video sampler.
    """
    super(UtilityModelVideoSampler, self).__init__(doc_ctor, **kwargs)
    self._doc_count = 0
    self._num_clusters = self.get_doc_ctor().NUM_FEATURES
    self._min_utility = min_utility
    self._max_utility = max_utility
    self._video_length = video_length

    # Linearly space utility according to cluster ID
    # cluster 0 will get min_utility. cluster
    # NUM_FEATURES - 1 will get max_utility
    # In between will be spaced as follows
    trashy = np.linspace(self._min_utility, 0, int(self._num_clusters * 0.7))
    nutritious = np.linspace(0, self._max_utility,
                             int(self._num_clusters * 0.3))
    self.cluster_means = np.concatenate((trashy, nutritious))

  def sample_document(self):
    doc_features = {}
    doc_features['doc_id'] = self._doc_count

    # Sample a cluster_id. Assumes there are NUM_FEATURE clusters.
    cluster_id = self._rng.randint(0, self._num_clusters)
    doc_features['cluster_id'] = cluster_id

    # Features are a 1-hot encoding of cluster id
    features = np.zeros(self._num_clusters)
    features[cluster_id] = 1.0
    doc_features['features'] = features

    # Fixed video lengths (in minutes)
    doc_features['video_length'] = self._video_length

    # Quality
    quality_mean = self.cluster_means[cluster_id]

    # Variance fixed
    quality_variance = 0.1
    doc_features['quality'] = self._rng.normal(quality_mean, quality_variance)

    self._doc_count += 1
    return self._doc_ctor(**doc_features)


class IEvUserState(user.AbstractUserState):
  """Class to represent interest evolution users."""

  # Number of features in the user state representation.
  NUM_FEATURES = 20

  def __init__(self,
               user_interests,
               time_budget=None,
               score_scaling=None,
               attention_prob=None,
               no_click_mass=None,
               keep_interact_prob=None,
               min_doc_utility=None,
               user_update_alpha=None,
               watched_videos=None,
               impressed_videos=None,
               liked_videos=None,
               step_penalty=None,
               min_normalizer=None,
               user_quality_factor=None,
               document_quality_factor=None):
    """Initializes a new user."""

    # Only user_interests is required, since it is needed to create an
    # observation. It is the responsibility of the designer to make sure any
    # other variables that are needed in the user choice/transition model are
    # also provided.

    ## User features
    #######################

    # The user's interests (1 = very interested, -1 = disgust)
    # Another option could be to represent in [0,1] e.g. by dirichlet
    self.user_interests = user_interests

    # Amount of time in minutes this user has left in session.
    self.time_budget = time_budget

    # Probability of interacting with another element on the same slate
    self.keep_interact_prob = keep_interact_prob

    # Min utility to interact with a document
    self.min_doc_utility = min_doc_utility

    # Convenience wrapper
    self.choice_features = {
        'score_scaling': score_scaling,
        # Factor of attention to give for subsequent items on slate
        # Item i on a slate will get attention (attention_prob)^i
        'attention_prob': attention_prob,
        # Mass that user does not click on any item in the slate
        'no_click_mass': no_click_mass,
        # If using the multinomial proportion model with negative scores, this
        # negative value will be subtracted from all scores to make a valid
        # distribution for sampling.
        'min_normalizer': min_normalizer
    }

    ## Transition model parameters
    ##############################

    # Step size for updating user interests based on watched videos (small!)
    # We may want to have different values for different interests
    # to represent how malleable those interests are (e.g. strong dislikes may
    # be less malleable).
    self.user_update_alpha = user_update_alpha

    # A step penalty applied when no item is selected (e.g. the time wasted
    # looking through a slate but not clicking, and any loss of interest)
    self.step_penalty = step_penalty

    # How much to weigh the user quality when updating budget
    self.user_quality_factor = user_quality_factor
    # How much to weigh the document quality when updating budget
    self.document_quality_factor = document_quality_factor

    # Observable user features (these are just examples for now)
    ###########################

    # Video IDs of videos that have been watched
    self.watched_videos = watched_videos

    # Video IDs of videos that have been impressed
    self.impressed_videos = impressed_videos

    # Video IDs of liked videos
    self.liked_videos = liked_videos

  def score_document(self, doc_obs):
    if self.user_interests.shape != doc_obs.shape:
      raise ValueError('User and document feature dimension mismatch!')
    return np.dot(self.user_interests, doc_obs)

  def create_observation(self):
    """Return an observation of this user's observable state."""
    return self.user_interests

  @classmethod
  def observation_space(cls):
    return spaces.Box(
        shape=(cls.NUM_FEATURES,), dtype=np.float32, low=-10.0, high=10.0)


class IEvUserDistributionSampler(user.AbstractUserSampler):
  """Class to sample users by a hardcoded distribution."""

  def __init__(self, user_ctor=IEvUserState, **kwargs):
    """Creates a new user state sampler."""
    logging.debug('Initialized IEvUserDistributionSampler')
    super(IEvUserDistributionSampler, self).__init__(user_ctor, **kwargs)

  def sample_user(self):
    """Samples a new user, with a new set of features."""

    features = {}
    features['user_interests'] = self._rng.uniform(
        -1.0, 1.0,
        self.get_user_ctor().NUM_FEATURES)
    features['time_budget'] = 30
    features['score_scaling'] = 0.05
    features['attention_prob'] = 0.9
    features['no_click_mass'] = 1
    features['keep_interact_prob'] = self._rng.beta(1, 3, 1)
    features['min_doc_utility'] = 0.1
    features['user_update_alpha'] = 0
    features['watched_videos'] = set()
    features['impressed_videos'] = set()
    features['liked_videos'] = set()
    features['step_penalty'] = 1.0
    features['min_normalizer'] = -1.0
    features['user_quality_factor'] = 1.0
    features['document_quality_factor'] = 1.0
    return self._user_ctor(**features)




DEFAULT_ALPHA = [1.5, 4]
DEFAULT_BETA = [4, 4]

@gin.configurable
class UtilityModelUserSampler(user.AbstractUserSampler):
  """Class that samples users for utility model experiment."""

  def __init__(self,
               user_ctor=IEvUserState,
               document_quality_factor=1.0,
               no_click_mass=1.0,
               min_normalizer=-1.0,
               alpha=(1.5, 4),
               beta=(4, 4),
               n_confounders=0,
               **kwargs):
    """Creates a new user state sampler."""
    logging.debug('Initialized UtilityModelUserSampler')
    self._no_click_mass = no_click_mass
    self._min_normalizer = min_normalizer
    self._document_quality_factor = document_quality_factor
    self.alpha = alpha
    self.beta = beta
    self.n_confounders = n_confounders
    super(UtilityModelUserSampler, self).__init__(user_ctor, **kwargs)

  def sample_user(self):
    features = {}
    # Interests are distributed uniformly randomly
    num_features = self.get_user_ctor().NUM_FEATURES
    features_default = features['user_interests'] = \
        self._rng.beta(np.linspace(DEFAULT_ALPHA[0], DEFAULT_ALPHA[1], num_features),
                       np.linspace(DEFAULT_BETA[0], DEFAULT_BETA[1], num_features),
                       num_features)
    features_confounded = \
        self._rng.beta(np.linspace(self.alpha[0], self.alpha[1], num_features),
                       np.linspace(self.beta[0], self.beta[1], num_features),
                       num_features)
    features['user_interests'] = features_default
    features['user_interests'][0:self.n_confounders] = features_confounded[0:self.n_confounders]
    features['user_interests'] = 2 * (features['user_interests'] - 0.5)
    # features['user_interests'] = self._rng.uniform(
    #     -1.0, 1.0,
    #     self.get_user_ctor().NUM_FEATURES)
    # Assume all users have fixed amount of time
    features['time_budget'] = 200.0  # 120.0
    features['no_click_mass'] = self._no_click_mass
    features['step_penalty'] = 0.5
    features['score_scaling'] = 0.05
    features['attention_prob'] = 0.65
    features['min_normalizer'] = self._min_normalizer
    features['user_quality_factor'] = 0.0
    features['document_quality_factor'] = self._document_quality_factor

    # Fraction of video length we can extend (or cut) budget by
    # Maybe this should be a parameter that varies by user?
    alpha = 0.9
    # In our setup, utility is just doc_quality as user_quality_factor is 0.
    # doc_quality is distributed normally ~ N([-3,3], 0.1) for a 3 sigma range
    # of [-3.3,3.3]. Therefore, we normalize doc_quality by 3.4 (adding a little
    # extra in case) to get in [-1,1].
    utility_range = 1.0 / 3.4
    features['user_update_alpha'] = alpha * utility_range
    return self._user_ctor(**features)


class IEvUserModel(user.AbstractUserModel):
  """Class to model an interest evolution user.

  Assumes the user state contains:
    - user_interests
    - time_budget
    - no_click_mass
  """

  def __init__(self,
               slate_size,
               choice_model_ctor=None,
               response_model_ctor=IEvResponse,
               user_state_ctor=IEvUserState,
               no_click_mass=1.0,
               alpha=0.5,
               beta=0.5,
               n_confounders=0,
               seed=0,
               alpha_x_intercept=1.0,
               alpha_y_intercept=0.3):
    """Initializes a new user model.

    Args:
      slate_size: An integer representing the size of the slate
      choice_model_ctor: A contructor function to create user choice model.
      response_model_ctor: A constructor function to create response. The
        function should take a string of doc ID as input and returns a
        IEvResponse object.
      user_state_ctor: A constructor to create user state
      no_click_mass: A float that will be passed to compute probability of no
        click.
      seed: A integer used as the seed of the choice model.
      alpha_x_intercept: A float for the x intercept of the line used to compute
        interests update factor.
      alpha_y_intercept: A float for the y intercept of the line used to compute
        interests update factor.

    Raises:
      Exception: if choice_model_ctor is not specified.
    """
    super(IEvUserModel, self).__init__(
        response_model_ctor,
        UtilityModelUserSampler(
            user_ctor=user_state_ctor, no_click_mass=no_click_mass, alpha=alpha, beta=beta, n_confounders=n_confounders, seed=seed),
        slate_size)
    if choice_model_ctor is None:
      raise Exception('A choice model needs to be specified!')
    self.choice_model = choice_model_ctor(self._user_state.choice_features)

    self._alpha_x_intercept = alpha_x_intercept
    self._alpha_y_intercept = alpha_y_intercept

  def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return self._user_state.time_budget <= 0

  def update_state(self, slate_documents, responses):
    """Updates the user state based on responses to the slate.

    This function assumes only 1 response per slate. If a video is watched, we
    update the user's interests some small step size alpha based on the
    user's interest in that topic. The update is either towards the
    video's features or away, and is determined stochastically by the user's
    interest in that document.

    Args:
      slate_documents: a list of IEvVideos representing the slate
      responses: a list of IEvResponses representing the user's response to each
        video in the slate.
    """

    user_state = self._user_state

    # Step size should vary based on interest.
    def compute_alpha(x, x_intercept, y_intercept):
      return (-y_intercept / x_intercept) * np.absolute(x) + y_intercept

    for doc, response in zip(slate_documents, responses):
      if response.clicked:
        self.choice_model.score_documents(
            user_state, [doc.create_observation()])
        # scores is a list of length 1 since only one doc observation is set.
        expected_utility = self.choice_model.scores[0]
        ## Update interests
        target = doc.features - user_state.user_interests
        mask = doc.features
        alpha = compute_alpha(user_state.user_interests,
                              self._alpha_x_intercept, self._alpha_y_intercept)

        update = alpha * mask * target
        positive_update_prob = np.dot((user_state.user_interests + 1.0) / 2,
                                      mask)
        flip = np.random.rand(1)
        if flip < positive_update_prob:
          user_state.user_interests += update
        else:
          user_state.user_interests -= update
        user_state.user_interests = np.clip(user_state.user_interests, -1.0,
                                            1.0)
        ## Update budget
        received_utility = (
            user_state.user_quality_factor * expected_utility) + (
                user_state.document_quality_factor * doc.quality)
        user_state.time_budget -= response.watch_time
        user_state.time_budget += (
            user_state.user_update_alpha * response.watch_time *
            received_utility)
        return

    # Step penalty if no selection
    user_state.time_budget -= user_state.step_penalty

  def simulate_response(self, documents):
    """Simulates the user's response to a slate of documents with choice model.

    Args:
      documents: a list of IEvVideo objects

    Returns:
      responses: a list of IEvResponse objects, one for each document
    """
    # List of empty responses
    responses = [self._response_model_ctor() for _ in documents]

    # Sample some clicked responses using user's choice model and populate
    # responses.
    doc_obs = [doc.create_observation() for doc in documents]
    self.choice_model.score_documents(self._user_state, doc_obs)
    selected_index = self.choice_model.choose_item()

    for i, response in enumerate(responses):
      response.quality = documents[i].quality
      response.cluster_id = documents[i].cluster_id

    if selected_index is None:
      return responses
    self._generate_click_response(documents[selected_index],
                                  responses[selected_index])

    return responses

  def _generate_click_response(self, doc, response):
    """Generates a response to a clicked document.

    Right now we assume watch_time is a fixed value that is the minium value of
    time_budget and video_length. In the future, we may want to try more
    variations of watch_time definition.

    Args:
      doc: an IEvVideo object
      response: am IEvResponse for the document
    Updates: response, with whether the document was clicked, liked, and how
      much of it was watched
    """
    user_state = self._user_state
    response.clicked = True
    response.watch_time = min(user_state.time_budget, doc.video_length)




REWARD_MATRIX = np.array([[  3.36,  -1.99,  -9.57,  -9.13,  -9.12,  -7.71,   6.  ,  -7.77,
         14.41,  14.85,  20.61,  16.94,  18.55,   0.97,  -6.16, -24.57,
        -25.95,  18.33,   6.72,   8.26],
       [  8.62,  10.63,  -2.55,  14.36, -12.15,  -3.46,   5.71,   3.74,
         20.08,  -4.54,  -3.18,  -0.26,  -6.31,  -6.38,  -8.39,  -8.43,
         -2.54,  -9.92,  -3.34, -10.24],
       [  6.86,   2.  ,   4.71,   4.72,  -4.77,  -0.78,   3.04,   1.56,
          7.42,  -4.82,   5.72,   2.17,  16.8 ,   4.68,  -4.35,  20.67,
        -11.45, -15.54,   8.04, -14.09],
       [-17.42,  -5.34,  11.89,  -0.63,  -1.99,   4.83,   4.6 ,  -4.19,
        -12.46,  -2.48,  21.36,  19.91, -12.7 ,   6.92,  16.53,  -0.23,
         -2.95, -13.23,  18.45,  -4.5 ],
       [  5.14,   2.8 , -14.2 ,  -2.36,  10.09,  17.28,  -1.43,   1.85,
          4.14,  -8.  , -10.3 , -14.21, -13.84, -15.67,   8.2 ,   4.21,
          1.42,  -9.95,   0.44,  13.25],
       [ -4.26,  -4.96,  -2.64,  24.81,  -1.77,   6.05,   2.96,  14.35,
         20.88,   5.13,   1.69,  10.66,   2.8 ,  19.75,  -8.59,   8.61,
          7.4 ,  14.04,   1.96,   0.41],
       [  6.73,  18.5 ,   4.44,  -5.28,   4.91,  -6.22,   8.85,   9.47,
        -15.06,   0.53,   4.83,  14.47, -13.83,   9.56,  12.56, -15.61,
          7.34,  -9.18,  10.99,  -1.82],
       [ -4.41,   2.08, -11.06,  -1.7 , -17.35,  14.91,   5.38,   0.7 ,
         -4.13,  -6.13,  -9.16,  14.82,  -9.25,   7.86,  -4.4 ,  15.04,
        -11.84,  -8.53,   6.34,   4.82],
       [  7.46,  -2.26,  15.44,  -2.05,   2.42,  -1.24, -13.66,  10.94,
         -7.47,  -0.32,  -1.66,   1.76,   5.35,  -6.92,  -0.07,   3.95,
         25.05, -19.08,   8.84,   2.14],
       [-17.18,   1.26,  10.72,  10.26,   8.19,   1.15,  -7.3 , -13.96,
         -4.81,  20.79,   9.54,  10.71,  -8.26,  -6.3 ,   0.53,   5.84,
         -8.97, -14.26,  -0.2 ,  -2.35],
       [  3.23, -22.16,  12.48, -15.67,   6.05,  -2.29, -12.32,   8.96,
         19.34,  -0.86,  22.82,  12.59,  15.57,  17.45,  -3.98,  -2.42,
          4.47,  -8.8 ,   8.55,  -8.09],
       [ -0.87, -17.48,  22.47,  -3.44,  -4.36,  -2.76,  -2.97,  -3.39,
         -7.85,   2.61, -21.11,  -2.57,  15.42,  16.44,  -6.94,   3.04,
          7.38,  21.56,  -7.21, -12.21],
       [ -9.44,  -1.67, -15.72,  -5.88,  -6.55,  -5.3 ,  -2.23,   2.94,
         13.09,   5.36,   8.61,   5.99, -12.31,  -5.85,   3.19,  -4.55,
          3.02,  -6.38,  -1.54,  -2.64],
       [ -3.17,  17.97,  -4.09,   5.22,  -7.83, -12.03,  -1.81,   7.9 ,
         23.8 ,  26.44,  -9.88,   0.92,  -7.09,  13.62, -20.11,   2.41,
         -4.08, -10.37,  -9.85,  10.56],
       [ -9.35,   9.39,   4.98,  13.13, -14.74,  15.87,  13.25,  -8.49,
         -0.72, -19.69,  -0.55,  17.93,  -9.16,  -8.13,  -6.94,  -2.67,
         11.06,  11.75,  17.13,   8.62],
       [ -1.65,  21.43,   1.4 , -10.9 , -17.15,   8.95,   4.3 ,  -0.58,
         -4.63, -20.14,  -2.71,   5.83, -16.97, -10.64,  -0.57,  -3.47,
         -1.19,  -1.9 ,   1.93,  -5.  ],
       [  2.64,  -2.77,  23.66,  -4.98,  -0.08,  -7.2 ,   8.03,   2.59,
          8.81,  -2.57,  18.11,  -2.24,   4.73,   2.04,   1.37, -20.51,
         14.96,   0.11,   3.8 ,  -9.59],
       [ -3.22,  -2.08,   7.66,   2.03,  -8.4 ,  -3.28,  10.35,   5.11,
          3.65, -11.28,  -3.07, -16.3 ,  -1.97,   2.02,  -3.82,  10.85,
         -1.6 ,   7.38,  -0.25,  -5.2 ],
       [ -7.7 ,  14.02,  -7.61,   3.68,   5.47,  -2.32,   5.77,  -5.85,
          4.02,  -4.32,  -2.67,   1.43,  -5.12,  -1.74,  -8.55,   9.07,
         -2.76,   8.63,  -1.3 ,   1.2 ],
       [ -8.45,  -2.88,   3.02, -11.88,   5.4 ,   5.81,  -1.55,  20.12,
        -12.91,   2.59,   0.64,   4.56,  16.58,   0.53,   5.68,  10.89,
          6.14,  10.51,   0.54,  -2.09]])


def clicked_watchtime_reward(responses, user_obs=None, doc_obs=None):
  """Calculates the total clicked watchtime from a list of responses.

  Args:
    responses: A list of IEvResponse objects

  Returns:
    reward: A float representing the total watch time from the responses
  """
  reward = 0.0
  # reward = np.tanh(REWARD_MATRIX[0] @ (np.cos(REWARD_MATRIX @ user_obs / 30) * np.sin(REWARD_MATRIX @ doc_obs[0] / 30)) / 10)
  # print(reward)
  reward = np.sum(user_obs @ doc_obs.T)
  for response in responses:
    if response.clicked:
      reward += 0 * response.watch_time
  return reward


def total_clicks_reward(responses):
  """Calculates the total number of clicks from a list of responses.

  Args:
     responses: A list of IEvResponse objects

  Returns:
    reward: A float representing the total clicks from the responses
  """
  reward = 0.0
  for r in responses:
    reward += r.clicked
  return reward


def create_environment(env_config):
  """Creates an interest evolution environment."""

  user_model = IEvUserModel(
      env_config['slate_size'],
      choice_model_ctor=choice_model.MultinomialProportionalChoiceModel,
      response_model_ctor=IEvResponse,
      user_state_ctor=IEvUserState,
      alpha=env_config['alpha'],
      beta=env_config['beta'],
      n_confounders=env_config['n_confounders'],
      seed=env_config['seed'])

  document_sampler = UtilityModelVideoSampler(
      doc_ctor=IEvVideo, seed=env_config['seed'])

  ievenv = environment.Environment(
      user_model,
      document_sampler,
      env_config['num_candidates'],
      env_config['slate_size'],
      resample_documents=env_config['resample_documents'])

  return recsim_gym.RecSimGymEnv(ievenv, clicked_watchtime_reward,
                                 utils.aggregate_video_cluster_metrics,
                                 utils.write_video_cluster_metrics)
