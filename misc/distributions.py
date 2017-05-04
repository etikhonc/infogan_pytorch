# original:  InfoGAN/infogan/misc/distributions.py
# modified: Ekaterina Sutter, May 2017

import itertools
import torch
import torch.nn.functional as F

import numpy as np

TINY = 1e-8
# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor

# ToDo change all classes to work/produce variables instead of tensors.
#  This should help to get rid of the is_tensor check


# https://github.com/pytorch/pytorch/issues/812
def embedding_lookup(embeddings, indices):
    if not torch.is_tensor(indices):
        indices = indices.data
    return embeddings.index_select(0, indices.view(-1)).view(*(indices.size() + embeddings.size()[1:]))


class Distribution(object):
    @property
    def dist_flat_dim(self):
        """
        :rtype: int
        """
        raise NotImplementedError

    @property
    def dim(self):
        """
        :rtype: int
        """
        raise NotImplementedError

    @property
    def effective_dim(self):
        """
        The effective dimension when used for rescaling quantities. This can be different from the
        actual dimension when the actual values are using redundant representations (e.g. for categorical
        distributions we encode it in onehot representation)
        :rtype: int
        """
        raise NotImplementedError

    def kl_prior(self, dist_info):
        return self.kl(dist_info, self.prior_dist_info(dist_info.values()[0].get_shape()[0]))

    def logli(self, x_var, dist_info):
        """
        :param x_var:
        :param dist_info:
        :return: log likelihood of the data
        """
        raise NotImplementedError

    def logli_prior(self, x_var):
        return self.logli(x_var, self.prior_dist_info(x_var.size(0)))

    def nonreparam_logli(self, x_var, dist_info):
        """
        :param x_var:
        :param dist_info:
        :return: the non-reparameterizable part of the log likelihood
        """
        raise NotImplementedError

    def activate_dist(self, flat_dist):
        """
        :param flat_dist: flattened dist info without applying nonlinearity yet
        :return: a dictionary of dist infos
        """
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        """
        :rtype: list[str]
        """
        raise NotImplementedError

    def entropy(self, dist_info):
        """
        :return: entropy for each minibatch entry
        """
        raise NotImplementedError

    def marginal_entropy(self, dist_info):
        """
        :return: the entropy of the mixture distribution averaged over all minibatch entries. Will return in the same
        shape as calling `:code:Distribution.entropy`
        """
        raise NotImplementedError

    def marginal_logli(self, x_var, dist_info):
        """
        :return: the log likelihood of the given variable under the mixture distribution averaged over all minibatch
        entries.
        """
        raise NotImplementedError

    def sample(self, dist_info):
        raise NotImplementedError

    def sample_prior(self, batch_size):
        return self.sample(self.prior_dist_info(batch_size))

    def prior_dist_info(self, batch_size):
        """
        :return: a dictionary containing distribution information about the standard prior distribution, the shape
                 of which is jointly decided by batch_size and self.dim
        """
        raise NotImplementedError


class Categorical(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self.dim

    @property
    def effective_dim(self):
        return 1

    def logli(self, x_var, dist_info):
        prob = dist_info["prob"]
        if not torch.is_tensor(prob):
            prob = prob.data
        if not torch.is_tensor(x_var):
            x_var = x_var.data
        return torch.sum(torch.log(prob + TINY) * x_var, 1)
        # return tf.reduce_sum(tf.log(prob + TINY) * x_var, reduction_indices=1)

    def prior_dist_info(self, batch_size):
        prob = torch.from_numpy(np.ones((batch_size, self.dim), dtype=np.float32) / float(self.dim)).type(dtype)
        # prob = tf.ones([batch_size, self.dim]) * floatX(1.0 / self.dim)
        return dict(prob=prob)

    def marginal_logli(self, x_var, dist_info):
        prob = dist_info["prob"]
        avg_prob = torch.mean(prob, 0)  # average over the batch
        avg_prob = avg_prob.reshape((1,) + prob.size(1))
        avg_prob = avg_prob.repeat(prob.size(0), 1)  # repeat for each entry
        # avg_prob = tf.tile(
        #     tf.reduce_mean(prob, reduction_indices=0, keep_dims=True),
        #     tf.stack([tf.shape(prob)[0], 1])
        # )
        return self.logli(x_var, dict(prob=avg_prob))

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    def kl(self, p, q):
        """
        :param p: left dist info
        :param q: right dist info
        :return: KL(p||q)
        """
        p_prob = p["prob"]
        q_prob = q["prob"]
        return torch.sum(p_prob * (torch.log(p_prob + TINY) - torch.log(q_prob + TINY)), dim=1)

    def sample(self, dist_info):
        prob = dist_info["prob"]
        ids = torch.multinomial(prob, num_samples=1)[:, 0]
        # ids = tf.multinomial(tf.log(prob + TINY), num_samples=1)[:, 0]
        onehot = torch.from_numpy(np.eye(self.dim, dtype=np.float32)).type(dtype)
        return embedding_lookup(onehot, ids)

    def activate_dist(self, flat_dist):
        return dict(prob=F.softmax(flat_dist))
        # return dict(prob=F.softmax(flat_dist).data)

    def entropy(self, dist_info):
        prob = dist_info["prob"]
        return -torch.sum(prob * torch.log(prob + TINY), dim=1)

    def marginal_entropy(self, dist_info):
        prob = dist_info["prob"]
        avg_prob = torch.mean(prob, 0)  # average over the batch
        avg_prob = avg_prob.reshape((1,) + prob.size(1))
        avg_prob = avg_prob.repeat(prob.size(0), 1)  # repeat for each entry
        # avg_prob = tf.tile(
        #     tf.reduce_mean(prob, reduction_indices=0, keep_dims=True),
        #     tf.stack([tf.shape(prob)[0], 1])
        # )
        return self.entropy(dict(prob=avg_prob))

    @property
    def dist_info_keys(self):
        return ["prob"]


class Gaussian(Distribution):
    def __init__(self, dim, fix_std=False):
        self._dim = dim
        self._fix_std = fix_std

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    def logli(self, x_var, dist_info):
        mean = dist_info["mean"]
        stddev = dist_info["stddev"]
        if not torch.is_tensor(x_var):
            x_var = x_var.data
        if not torch.is_tensor(mean):
            mean = mean.data
        if not torch.is_tensor(stddev):
            stddev = stddev.data
        epsilon = (x_var - mean) / (stddev + TINY)
        return torch.sum(- 0.5 * np.log(2 * np.pi) - torch.log(stddev + TINY) - 0.5 * epsilon*epsilon, dim=1)

    def prior_dist_info(self, batch_size):
        mean = torch.zeros([batch_size, self.dim]).type(dtype)
        stddev = torch.ones([batch_size, self.dim]).type(dtype)
        return dict(mean=mean, stddev=stddev)

    def nonreparam_logli(self, x_var, dist_info):
        return torch.zeros(x_var[:, 0].size()).type(dtype)
        # return tf.zeros_like(x_var[:, 0])

    def kl(self, p, q):
        p_mean = p["mean"]
        p_stddev = p["stddev"]
        q_mean = q["mean"]
        q_stddev = q["stddev"]
        # means: (N*D)
        # std: (N*D)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) + ln(\sigma_2/\sigma_1)
        numerator = torch.square(p_mean - q_mean) + torch.square(p_stddev) - torch.square(q_stddev)
        denominator = 2. * torch.square(q_stddev)
        return torch.sum(
            numerator / (denominator + TINY) + torch.log(q_stddev + TINY) - torch.log(p_stddev + TINY),
            dim=1
        )

    def sample(self, dist_info):
        mean = dist_info["mean"]
        stddev = dist_info["stddev"]
        epsilon = torch.randn(mean.size()).type(dtype)

        if not torch.is_tensor(mean):
            mean = mean.data
        if not torch.is_tensor(stddev):
            stddev = stddev.data
        return mean + epsilon * stddev


    @property
    def dist_info_keys(self):
        return ["mean", "stddev"]

    def activate_dist(self, flat_dist):
        mean = flat_dist[:, :self.dim].type(dtype)
        if self._fix_std:
            stddev = torch.ones(mean.size()).type(dtype)
        else:
            stddev = torch.sqrt(torch.exp(flat_dist[:, self.dim:]))
        return dict(mean=mean, stddev=stddev)


class Uniform(Gaussian):
    """
    This distribution will sample prior data from a uniform distribution, but
    the prior and posterior are still modeled as a Gaussian
    """

    def kl_prior(self):
        raise NotImplementedError

    # def prior_dist_info(self, batch_size):
    #     raise NotImplementedError

    # def logli_prior(self, x_var):
    #     #
    #     raise NotImplementedError

    def sample_prior(self, batch_size):
        return torch.Tensor(batch_size, self.dim).uniform_(-1., 1.).type(dtype)


class Bernoulli(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["p"]

    def logli(self, x_var, dist_info):
        p = dist_info["p"]
        return torch.sum(x_var * torch.log(p + TINY) + (1.0 - x_var) * torch.log(1.0 - p + TINY), dim=1)

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    def activate_dist(self, flat_dist):
        return dict(p=F.sigmoid(flat_dist))
        # return dict(p=F.sigmoid(flat_dist).data)

    def sample(self, dist_info):
        p = dist_info["p"]
        ind = np.random.rand(p.get_shape()) < p
        return torch.from_numpy(ind).type(dtype)

    def prior_dist_info(self, batch_size):
        return dict(p=0.5 * torch.ones([batch_size, self.dim]).type(dtype))


class MeanBernoulli(Bernoulli):
    """
    Behaves almost the same as the usual Bernoulli distribution, except that when sampling from it, directly
    return the mean instead of sampling binary values
    """

    def sample(self, dist_info):
        return dist_info["p"]

    def nonreparam_logli(self, x_var, dist_info):
        return torch.zeros(x_var.shape).type(dtype)  #  tf.zeros_like(x_var[:, 0])


# class MeanCenteredUniform(MeanBernoulli):
#     """
#     Behaves almost the same as the usual Bernoulli distribution, except that when sampling from it, directly
#     return the mean instead of sampling binary values
#     """


class Product(Distribution):
    def __init__(self, dists):
        """
        :type dists: list[Distribution]
        """
        self._dists = dists

    @property
    def dists(self):
        return list(self._dists)

    @property
    def dim(self):
        return sum(x.dim for x in self.dists)

    @property
    def effective_dim(self):
        return sum(x.effective_dim for x in self.dists)

    @property
    def dims(self):
        return [x.dim for x in self.dists]

    @property
    def dist_flat_dims(self):
        return [x.dist_flat_dim for x in self.dists]

    @property
    def dist_flat_dim(self):
        return sum(x.dist_flat_dim for x in self.dists)

    @property
    def dist_info_keys(self):
        ret = []
        for idx, dist in enumerate(self.dists):
            for k in dist.dist_info_keys:
                ret.append("id_%d_%s" % (idx, k))
        return ret

    def split_dist_info(self, dist_info):
        ret = []
        for idx, dist in enumerate(self.dists):
            cur_dist_info = dict()
            for k in dist.dist_info_keys:
                cur_dist_info[k] = dist_info["id_%d_%s" % (idx, k)]
            ret.append(cur_dist_info)
        return ret

    def join_dist_infos(self, dist_infos):
        ret = dict()
        for idx, dist, dist_info_i in zip(itertools.count(), self.dists, dist_infos):
            for k in dist.dist_info_keys:
                ret["id_%d_%s" % (idx, k)] = dist_info_i[k]
        return ret

    def split_var(self, x):
        """
        Split the tensor variable or value into per component.
        """
        cum_dims = list(np.cumsum(self.dims))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.dists):
            sliced = x[:, slice_from:slice_to]
            out.append(sliced)
        return out

    def join_vars(self, xs):
        """
        Join the per component tensor variables into a whole tensor
        """
        return torch.cat(xs, dim=1)

    def split_dist_flat(self, dist_flat):
        """
        Split flat dist info into per component
        """
        cum_dims = list(np.cumsum(self.dist_flat_dims))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.dists):
            sliced = dist_flat[:, slice_from:slice_to]
            out.append(sliced)
        return out

    def prior_dist_info(self, batch_size):
        ret = []
        for dist_i in self.dists:
            ret.append(dist_i.prior_dist_info(batch_size))
        return self.join_dist_infos(ret)

    def kl(self, p, q):
        ret = 0.  # tf.constant(0.)
        for p_i, q_i, dist_i in zip(self.split_dist_info(p), self.split_dist_info(q), self.dists):
            ret += dist_i.kl(p_i, q_i)
        return ret

    def activate_dist(self, dist_flat):
        ret = dict()
        for idx, dist_flat_i, dist_i in zip(itertools.count(), self.split_dist_flat(dist_flat), self.dists):
            dist_info_i = dist_i.activate_dist(dist_flat_i)
            for k, v in dist_info_i.iteritems():
                ret["id_%d_%s" % (idx, k)] = v
        return ret

    def sample(self, dist_info):
        ret = []
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret.append((dist_i.sample(dist_info_i)).double())
            # ret.append(tf.cast(dist_i.sample(dist_info_i), tf.float32))
        return torch.cat(ret, dim=1)

    def sample_prior(self, batch_size):
        ret = []
        for dist_i in self.dists:
            ret.append((dist_i.sample_prior(batch_size).double()))
            # ret.append(tf.cast(dist_i.sample_prior(batch_size), tf.float32))
        return torch.cat(ret, dim=1)

    def logli(self, x_var, dist_info):
        ret = 0.  # tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.logli(x_i, dist_info_i)
        return ret

    def marginal_logli(self, x_var, dist_info):
        ret = 0.  # tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.marginal_logli(x_i, dist_info_i)
        return ret

    def entropy(self, dist_info):
        ret = 0.  # tf.constant(0.)
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret += dist_i.entropy(dist_info_i)
        return ret

    def marginal_entropy(self, dist_info):
        ret = 0.  # tf.constant(0.)
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret += dist_i.marginal_entropy(dist_info_i)
        return ret

    def nonreparam_logli(self, x_var, dist_info):
        ret = 0.  # tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.nonreparam_logli(x_i, dist_info_i)
        return ret
