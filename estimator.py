import tensorflow as tf
from input_pipeline import InputPipeline
from transformer import block, mlp, add_timing_signal_1d


def bayesianesque_embeddings(features, labels, mode, params):
    """
    Note: Labels will be max lengths.
    """
    seq_len = features["lens"]
    raw_seqs = features["seqs"]

    pdrop = params["pdrop"] if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    # TODO add partitioners to these
    mu_embed = tf.get_variable("mean_embed", shape=[params["vocab_size"], params["embed_dim"]], dtype=tf.float32, partitioner=tf.fixed_size_partitioner(params["num_shards"]))
    mean_embedded_input = tf.nn.embedding_lookup(mu_embed, raw_seqs)

    if not params["raw_word2vec"]:
        cov_embed = tf.get_variable("cov_embed", shape=[params["vocab_size"], params["variance_size"], params["embed_dim"]], partitioner=tf.fixed_size_partitioner(params["num_shards"]))

        cov_embed_input = tf.nn.embedding_lookup(cov_embed, raw_seqs, partition_strategy="div")

        transformer_in = add_timing_signal_1d(mean_embedded_input)
        h = block(transformer_in, params["n_heads"], seq_len, pdrop, "trans_block")

        cov = mlp(h, "mlp", params["embed_dim"] * 2, pdrop, nx=params["variance_size"])  # batch, seq, cov_dim
        mean, variance = tf.nn.moments(cov, [0, 1])
        divergence_loss = tf.abs(mean) + tf.abs(variance - 1.0)
        # TODO minimise divergence

        embedding = mean_embedded_input + tf.reduce_sum(tf.expand_dims(cov, 3) * cov_embed_input,
                                                        2)  # batch, seq_len, embed_dim
    else:
        embedding = mean_embedded_input
        divergence_loss = 0.0

    seq_len_with_pad = tf.shape(embedding)[1]
    if mode == tf.estimator.ModeKeys.PREDICT:
        embedding *= tf.sequence_mask(seq_len, seq_len_with_pad, dtype=tf.float32)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=embedding)

    window = params["window_size"]
    pf = tf.pad(raw_seqs, [[0, 0], [window, window]], constant_values=params["pad_id"])
    targets = tf.map_fn(lambda i: tf.concat((pf[:, i - window:i], pf[:, i + 1: i + 1 + window]), axis=-1),
                        tf.range(window, window + seq_len_with_pad), dtype=tf.int32)
    embedding = tf.reshape(embedding, shape=[-1, params["embed_dim"]])
    mask = tf.reshape(tf.sequence_mask(seq_len, seq_len_with_pad, dtype=tf.float32), [-1])
#    targets = tf.Print(targets, [tf.reshape(targets, shape=[-1, window * 2])[0], raw_seqs[0], mask], summarize=1000)

    loss = tf.nn.nce_loss(
        tf.get_variable("output_embed", shape=[params["vocab_size"], params["embed_dim"]], dtype=tf.float32, partitioner=tf.fixed_size_partitioner(params["num_shards"])),
        tf.get_variable("output_bias", shape=[params["vocab_size"]], dtype=tf.float32),
        tf.reshape(targets, shape=[-1, window * 2]),
        embedding,
        params["sampled_softmax_size"],
        params["vocab_size"],
        num_true=window * 2,
        remove_accidental_hits=True,
        partition_strategy='div',
        name='nce_loss',
    )

    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get("optimizer", tf.train.AdamOptimizer)
        optimizer = optimizer(params.get("learning_rate", None))
        train_op = optimizer.minimize(
            loss=loss + divergence_loss * params["divergence_weight"], global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss)


def main(argv):
    pipeline = InputPipeline("data/train/*", val_filename="data/test/*", vocab_size=100000, tokens_per_batch=1200,
                             max_lengths_per_bucket=[10, 20, 30, 40, 50, 60, 80, 100, 140, 160, 180, 200, 300, 400, 500,
                                 600])

    conf = tf.ConfigProto(log_device_placement=False)
    conf.gpu_options.allow_growth = True

    config = tf.estimator.RunConfig(
        model_dir="model_fixed",
        tf_random_seed=None,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        save_checkpoints_secs=None,
        session_config=conf,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=100,
        #     train_distribute=None,
        #    device_fn=None
    )

    model = tf.estimator.Estimator(
        model_fn=bayesianesque_embeddings,
        params={
            "learning_rate": 0.001,
            "optimizer": tf.train.AdamOptimizer,
            "embed_dim": 300,
            "vocab_size": pipeline.vocab_size,
            "variance_size": 3,
            "pdrop": 0.1,
            "n_heads": 12,
            "pad_id": pipeline.pad_id,
            "sampled_softmax_size": 20,
            "window_size": 4,
            "divergence_weight": 0.1,
            "num_shards": 10,
            "raw_word2vec": False
        },
        config=config
    )
    train_spec = tf.estimator.TrainSpec(input_fn=pipeline.get_input_fn())
    eval_spec = tf.estimator.EvalSpec(input_fn=pipeline.get_input_fn("val"))
    tf.estimator.train_and_evaluate(model, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
