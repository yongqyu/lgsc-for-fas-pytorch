import tensorflow as tf
import tensorflow.keras as K


# source code from: https://github.com/Yuol96/pytorch-triplet-loss
def pairwise_distances(embeddings, squared=False):
    """
    ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
    """
    # get dot product (batch_size, batch_size)
    dot_product = tf.linalg.matmul(embeddings, embeddings, transpose_b=True)

    # a vector
    square_sum = tf.linalg.diag_part(dot_product)

    distances = (
        tf.expand_dims(square_sum, 1) - 2 * dot_product + tf.expand_dims(square_sum, 0)
    )

    distances = tf.clip_by_value(distances, clip_value_min=0, clip_value_max=tf.reduce_max(distances))

    if not squared:
        epsilon = 1e-16
        mask = tf.cast(tf.math.equal(distances, 0), tf.float32)
        distances += mask * epsilon
        distances = tf.math.sqrt(distances)
        distances *= 1 - mask

    return distances


# source code from: https://github.com/Yuol96/pytorch-triplet-loss
def get_valid_triplets_mask(labels):
    """
    To be valid, a triplet (a,p,n) has to satisfy:
        - a,p,n are distinct embeddings
        - a and p have the same label, while a and n have different label
    """
    labels = 1-labels
    indices_equal = tf.cast(tf.eye(labels.shape[0]), tf.bool)
    indices_not_equal = ~indices_equal
    i_ne_j = tf.expand_dims(indices_not_equal,2)
    i_ne_k = tf.expand_dims(indices_not_equal,1)
    j_ne_k = tf.expand_dims(indices_not_equal,0)
    distinct_indices = i_ne_j & i_ne_k & j_ne_k

    label_equal = tf.math.equal(tf.expand_dims(labels,1), tf.expand_dims(labels,0))
    i_eq_j = tf.expand_dims(label_equal,2)
    i_eq_k = tf.expand_dims(label_equal,1)
    i_ne_k = ~i_eq_k
    valid_labels = i_eq_j & i_ne_k

    mask = distinct_indices & valid_labels
    return mask


# source code from: https://github.com/Yuol96/pytorch-triplet-loss
def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """
    get triplet loss for all valid triplets and average over those triplets whose loss is positive.
    """

    distances = pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(distances, 2)
    anchor_negative_dist = tf.expand_dims(distances, 1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # get a 3D mask to filter out invalid triplets
    mask = get_valid_triplets_mask(labels)

    triplet_loss = triplet_loss * tf.cast(mask, tf.float32)
    triplet_loss = tf.clip_by_value(triplet_loss, clip_value_min=0, clip_value_max=tf.reduce_max(triplet_loss))

    # count the number of positive triplets
    epsilon = 1e-16
    num_positive_triplets = tf.reduce_sum(tf.cast(triplet_loss > 0, tf.float32))
    num_valid_triplets = tf.reduce_sum(tf.cast(mask, tf.float32))
    fraction_positive_triplets = num_positive_triplets / (
        num_valid_triplets + epsilon
    )

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + epsilon)

    return triplet_loss, fraction_positive_triplets


class TripletLoss(K.Model):
    """Triplet Loss

    arXiv: https://arxiv.org/pdf/1703.07737.pdf
    """

    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, pred, target):
        loss, _ = batch_all_triplet_loss(
            target, pred, self.margin, squared=True
        )
        return loss
