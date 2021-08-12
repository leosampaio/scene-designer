import tensorflow as tf

from builders.layers.helpers import build_mlp
from builders.utils import gather


class GraphTripleConvLayer(tf.keras.layers.Layer):

    def __init__(self, input_dim, attributes_dim=0,
                 output_dim=None, hidden_dim=512,
                 pooling='avg', mlp_normalization='none', activation='relu'):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg'], "Invalid pooling '{}'".format(pooling)
        self.pooling = pooling
        net1_layers = [3 * input_dim + 2 * attributes_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization, activation=activation)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization, activation=activation)

    def call(self, obj_vecs, predi_vecs, edges):
        """
        Inputs:
        - obj_vecs: tensor of shape (O, D) giving vectors for all objects
        - predi_vecs: tensor of shape (T, D) giving vectors for all predicates
        - edges: tensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], predi_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: tensor of shape (O, D) giving new vectors for objects
        - new_predi_vecs: tensor of shape (T, D) giving new vectors for predicates
        """
        dtype = obj_vecs.dtype
        O, T = tf.shape(obj_vecs)[0], tf.shape(predi_vecs)[0]
        D_in, H, D_out = self.input_dim, self.hidden_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0]
        o_idx = edges[:, 1]

        # Get current vectors for subjects and objects; these have shape (T, D_in)
        cur_s_vecs = tf.gather(obj_vecs, s_idx)
        cur_o_vecs = tf.gather(obj_vecs, o_idx)

        # Get current vectors for triples; shape is (T, 3 * D_in)
        # Pass through net1 to get new triple vecs; shape is (T, 2 * H + D_out)
        cur_t_vecs = tf.concat([cur_s_vecs, predi_vecs, cur_o_vecs], axis=1)
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
        # p vecs have shape (T, D_out)
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H + D_out)]
        new_o_vecs = new_t_vecs[:, (H + D_out):(2 * H + D_out)]

        # Use scatter_nd to sum vectors triples;
        pooled_obj_vecs = tf.scatter_nd(
            indices=tf.reshape(s_idx, (-1, 1)), updates=new_s_vecs, shape=(O, H))
        pooled_obj_vecs = pooled_obj_vecs + tf.scatter_nd(
            indices=tf.reshape(o_idx, (-1, 1)), updates=new_o_vecs, shape=(O, H))

        # not sure if it was a good translation from:
        #   allocate space for pooled object vectors of shape (O, H)
        #   pooled_obj_vecs = tf.zeros((O, H), dtype=dtype)
        #   pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        #   pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            ones = tf.ones((T,), dtype=dtype)
            obj_counts = tf.scatter_nd(
                indices=tf.reshape(s_idx, (-1, 1)), updates=ones, shape=(O,))
            obj_counts = obj_counts + tf.scatter_nd(
                indices=tf.reshape(o_idx, (-1, 1)), updates=ones, shape=(O,))

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = tf.clip_by_value(obj_counts, 1, 1000)
            obj_counts = tf.reshape(obj_counts, (-1, 1))
            pooled_obj_vecs = pooled_obj_vecs / obj_counts

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (O, D_out)
        new_obj_vecs = self.net2(pooled_obj_vecs)

        return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(tf.keras.layers.Layer):

    def __init__(self, input_dim, num_layers=5,
                 hidden_dim=512, pooling='avg',
                 mlp_normalization='none', activation='relu'):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = []
        for _ in range(self.num_layers):
            gconv = GraphTripleConvLayer(input_dim=input_dim,
                                         hidden_dim=hidden_dim,
                                         pooling=pooling,
                                         mlp_normalization=mlp_normalization,
                                         activation=activation)
            self.gconvs.append(gconv)

    def call(self, obj_vecs, pred_vecs, edges):
        for gconv in self.gconvs:
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs
