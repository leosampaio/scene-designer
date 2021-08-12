import numpy as np
import pickle
import tensorflow as tf


class VectorRepRandomizerPool:

    def __init__(self, pool_size, dim):
        self.pool_size = pool_size
        self.vectors = {}
        self.dim = dim

    def query(self, objs):
        if self.pool_size == 0:
            return np.zeros((len(objs), self.dim))
        return_vectors = []
        for obj in objs:
            if obj not in self.vectors:
                self.vectors[obj] = []
            obj_pool_size = len(self.vectors[obj])
            if obj_pool_size == 0:
                return_vectors.append(np.zeros((self.dim,)))
            else:
                random_id = 0 if obj_pool_size == 1 else np.random.randint(0, obj_pool_size - 1)
                return_vectors.append(self.vectors[obj][random_id])

        return_vectors = np.stack(return_vectors)
        return return_vectors

    def query_avgs(self, objs):
        if self.pool_size == 0:
            return np.zeros((len(objs), self.dim))
        return_vectors = []
        for obj in objs:
            if obj not in self.vectors:
                self.vectors[obj] = []
            if len(self.vectors[obj]) == 0:
                return_vectors.append(np.zeros((self.dim,)))
            else:
                return_vectors.append(np.mean(self.vectors[obj], axis=0))

        return_vectors = np.stack(return_vectors)
        return return_vectors

    def update_pool(self, objs, vectors):
        if self.pool_size == 0:
            return
        for obj, vector in zip(objs, vectors):
            vector = vector.numpy()
            if obj not in self.vectors:
                self.vectors[obj] = []
            obj_pool_size = len(self.vectors[obj])
            if obj_pool_size < self.pool_size:
                self.vectors[obj].append(vector)
            else:
                random_id = np.random.randint(0, obj_pool_size - 1)
                self.vectors[obj][random_id] = vector

    def save(self, filepath):
        dict_repr = {"vectors": self.vectors}
        with open(filepath, 'wb') as f:
            pickle.dump(dict_repr, f)

    def load(self, filepath):
        try:
            with open(filepath, "rb") as f:
                dict_repr = pickle.load(f)
                self.vectors = dict_repr["vectors"]
        except IOError:
            print("[Checkpoint] Couldn't load representation vectors, file not found")


class TFVectorRepRandomizerPool(tf.keras.Model):

    def __init__(self, pool_size, dim, n_pools):
        super().__init__()
        self.pool_size = pool_size
        self.n_pools = n_pools
        self.vectors = tf.Variable(tf.zeros((n_pools, pool_size, dim)), trainable=False, synchronization=tf.VariableSynchronization.ON_READ,
                                   aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.lengths = tf.Variable(tf.zeros((n_pools,), dtype=tf.int32),  trainable=False, synchronization=tf.VariableSynchronization.ON_READ,
                                   aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    def query(self, objs):
        vecs = tf.gather(self.vectors, objs, axis=0)
        lens = tf.gather(self.lengths, objs, axis=0)

        def query_map(vcs_lens):
            vecs, lens = vcs_lens
            if lens == 0:
                return vecs[0]
            else:
                idx = tf.random.uniform((), minval=0, maxval=lens, dtype=tf.int32)
                return vecs[idx]
        return tf.map_fn(query_map, (vecs, lens), dtype=tf.float32)

    def query_avgs(self, objs):
        vecs = tf.gather(self.vectors, objs, axis=0)
        lens = tf.gather(self.lengths, objs, axis=0)

        return tf.reduce_sum(vecs, axis=1) / tf.expand_dims(tf.cast(lens, tf.float32) + 1e-5, -1)

    def update_pool(self, objs, vectors):
        for obj, new_vec in zip(objs, vectors):
            if self.lengths[obj] < self.pool_size:
                self.vectors[obj, self.lengths[obj]].assign(new_vec)
                self.lengths[obj].assign(self.lengths[obj] + 1)
            else:
                idx = tf.random.uniform((), minval=0, maxval=self.lengths[obj], dtype=tf.int32)
                self.vectors[obj, idx].assign(new_vec)

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass


class TFVectorRepQueue(tf.keras.Model):

    def __init__(self, pool_size, dim):
        super().__init__()
        self.pool_size = pool_size
        self.vectors = tf.Variable(tf.zeros((pool_size, dim)), trainable=False, synchronization=tf.VariableSynchronization.ON_READ,
                                   aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.cursor = tf.Variable(tf.zeros((), dtype=tf.int32),  trainable=False, synchronization=tf.VariableSynchronization.ON_READ,
                                  aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    def update_queue(self, vectors):
        for new_vec in vectors:
            if self.cursor >= self.pool_size:
                self.cursor.assign(0)
            self.vectors[self.cursor].assign(new_vec)
            self.cursor.assign(self.cursor + 1)
