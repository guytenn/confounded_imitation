from typing import Dict, Callable, List
import numpy as np
from types import SimpleNamespace


class GeneralDist:
    def __init__(self, params):
        self.params = params

    def sample(self):
        raise NotImplementedError


class Multinomial(GeneralDist):
    def __init__(self, p, fields=None):
        super(Multinomial, self).__init__(params=dict(p=p, fields=fields))
        self.p = p
        self.fields = fields

    def sample(self):
        choice = np.random.choice(len(self.p), p=self.p)
        if self.fields is None:
            return choice, None
        return choice, self.fields[choice]


class GaussianDist(GeneralDist):
    def __init__(self, mu, sigma, clip=None):
        super(GaussianDist, self).__init__(params=dict(mu=mu, sigma=sigma, clip=None))
        self.mu = mu
        self.sigma = sigma
        self.clip = clip
        self.n = 1 if np.isscalar(mu) else len(mu)

    def sample(self):
        res = self.mu + self.sigma * np.random.randn(self.n)
        if self.clip is None:
            return res
        return np.clip(res, self.clip[0], self.clip[1])


class UniformDist(GeneralDist):
    def __init__(self, a, b):
        super(UniformDist, self).__init__(params=dict(a=a, b=b))
        self.a = a
        self.b = b
        self.n = 1 if np.isscalar(a) else len(a)

    def sample(self):
        return self.a + (self.b - self.a) * np.random.rand(self.n)


class FieldDependentDist(GeneralDist):
    def __init__(self, base_name: str, base_dist: Multinomial, dependent_dists: Dict[str, Dict[str, GeneralDist]]):
        super(FieldDependentDist, self).__init__(params=dict(origin_dist=base_dist, dependent_dists=dependent_dists))
        assert(base_dist.fields is not None)
        self.base_name = base_name
        self.base_dist = base_dist
        self.dependent_dists = dependent_dists

    def sample(self):
        sampled_idx, sampled_field = self.base_dist.sample()
        values = {self.base_name: (sampled_idx, sampled_field)}
        for dist_key, dist_dict in self.dependent_dists[sampled_field].items():
            values[dist_key] = self.dependent_dists[sampled_field][dist_key].sample()
        return values


class Context:
    def __init__(self, fields: Dict[str, GeneralDist]):
        self.fields = fields

    def add_field(self, name: str, dist: GeneralDist):
        self.fields.update({name: dist})

    def sample(self):
        values = dict()
        for name, dist in self.fields.items():
            value = dist.sample()
            if isinstance(value, Dict):
                values.update(value)
            else:
                values[name] = dist.sample()
        return values
