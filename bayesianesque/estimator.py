import os
import tensorflow as tf
from bayesianesque.input_pipeline import InputPipeline
from bayesianesque.transformer import block, mlp, add_timing_signal_1d, decoder, class_reweighting, noam_lr


def bayesianesque_embeddings(features, labels, mode, params):
    """
    Note: Labels will be max lengths.
    """
    seq_len = features["lens"]
    raw_seqs = features["seqs"]

    pdrop = params["pdrop"] if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    # TODO add partitioners to these
    mu_embed = tf.get_variable("mean_embed", shape=[params["vocab_size"], params["embed_dim"]], dtype=tf.float32, partitioner=tf.fixed_size_partitioner(params["num_shards"]))
    mean_embedded_input = tf.nn.embedding_lookup(mu_embed, raw_seqs, partition_strategy="div")

    if not params["raw_word2vec"]:
        cov_embed = tf.get_variable("cov_embed", shape=[params["vocab_size"], params["variance_size"], params["embed_dim"]], partitioner=tf.fixed_size_partitioner(params["num_shards"]))
        cov_embed_input = tf.nn.embedding_lookup(cov_embed, raw_seqs, partition_strategy="div")

        transformer_in = add_timing_signal_1d(mean_embedded_input)
        h = block(transformer_in, params["n_heads"], seq_len, pdrop, "trans_block")

        cov = mlp(h, "mlp", params["embed_dim"] * 2, pdrop, nx=params["variance_size"])  # batch, seq, cov_dim
        mean, variance = tf.nn.moments(cov, [0, 1])
        divergence_loss = tf.reduce_mean(tf.abs(mean) + tf.abs(variance - 1.0))
        # TODO minimise divergence

        embedding = mean_embedded_input + tf.reduce_sum(tf.expand_dims(cov, 3) * cov_embed_input,
                                                        2)  # batch, seq_len, embed_dim
    else:
        embedding = mean_embedded_input
        divergence_loss = 0.0

    seq_len_with_pad = tf.shape(embedding)[1]
    embed_mask = tf.expand_dims(tf.sequence_mask(seq_len, seq_len_with_pad, dtype=tf.float32), -1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        embedding *= embed_mask
        embedding = tf.reduce_mean(embedding, axis=1)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=embedding)

    if params["reweight"]:
        freqs = tf.gather(params["frequencies"], raw_seqs)
        weights = tf.sqrt(1/freqs)
        embedding = class_reweighting(tf.expand_dims(weights, -1))(embedding)

    output_embed = tf.get_variable("output_embed", shape=[params["vocab_size"], params["embed_dim"]], dtype=tf.float32,
                                   partitioner=tf.fixed_size_partitioner(params["num_shards"]))
    output_bias = tf.get_variable("output_bias", shape=[params["vocab_size"]], dtype=tf.float32)

    if params["reconstruction_loss"]:
        reconstruction_loss = decoder(embedding, raw_seqs, tf.reduce_mean(embedding * embed_mask, axis=1),
                                      output_embed, output_bias,
                                      params["n_heads"], params["num_decoder_blocks"], pdrop, seq_len,
                                      params["vocab_size"], params["sampled_softmax_size"])

    else:
        reconstruction_loss = 0.0

    window = params["window_size"]
    pf = tf.pad(raw_seqs, [[0, 0], [window, window]], constant_values=params["pad_id"])
    targets = tf.map_fn(lambda i: tf.concat((pf[:, i - window:i], pf[:, i + 1: i + 1 + window]), axis=-1),
                        tf.range(window, window + seq_len_with_pad), dtype=tf.int32)
    embedding = tf.reshape(embedding, shape=[-1, params["embed_dim"]])
    mask = tf.reshape(tf.sequence_mask(seq_len, seq_len_with_pad, dtype=tf.float32), [-1])
#    targets = tf.Print(targets, [tf.reshape(targets, shape=[-1, window * 2])[0], raw_seqs[0], mask], summarize=1000)

    loss = tf.nn.nce_loss(
        output_embed,
        output_bias,
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

    tf.summary.scalar("Reconstruction", reconstruction_loss)
    tf.summary.scalar("MainLoss", loss)
    tf.summary.scalar("DivergenceLoss", divergence_loss)

    loss = loss + reconstruction_loss * params["reconstruction_weight"]

    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = noam_lr(params.get("learning_rate", None), params["embed_dim"], tf.train.get_or_create_global_step(), params["warmup_steps"])
        optimizer = params.get("optimizer")
        train_op = tf.contrib.layers.optimize_loss(
            learning_rate=lr,
            loss=tf.reduce_mean(loss + divergence_loss * params["divergence_weight"]),
            global_step=tf.train.get_global_step(), optimizer=optimizer,
            clip_gradients=1.0, summaries=[
                "learning_rate",
                "loss",
                "gradients",
                "gradient_norm",
                "global_gradient_norm",
            ])

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss)

class EmbedModel:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
        self.pipeline = InputPipeline(
            os.path.join(dir_path, "data/train/*"),
            val_filename=os.path.join(dir_path, "data/test/*"),
            vocab_size=100000,
            tokens_per_batch=7000,
            max_lengths_per_bucket=[10, 20, 30, 40, 50, 60, 80, 100, 140, 160, 180, 200, 300, 400, 500, 600])

        conf = tf.ConfigProto(log_device_placement=False)
        conf.gpu_options.allow_growth = True

        config = tf.estimator.RunConfig(
            model_dir=os.path.join(dir_path, "subword_model"),
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

        self.model = tf.estimator.Estimator(
            model_fn=bayesianesque_embeddings,
            params={
                "learning_rate": 1e-4,
                "optimizer": "Adam",
                "embed_dim": 512,
                "vocab_size": self.pipeline.vocab_size,
                "variance_size": 3,
                "pdrop": 0.1,
                "n_heads": 16,
                "pad_id": self.pipeline.pad_id,
                "sampled_softmax_size": 20,
                "window_size": 4,
                "divergence_weight": 0.1,
                "num_shards": 10,
                "raw_word2vec": False,
                "reconstruction_loss": True,
                "reconstruction_weight": 1.0,
                "num_decoder_blocks": 3,
                "frequencies": self.pipeline.frequencies,
                "reweight": True,
                "warmup_steps": 100000
            },
            config=config
        )

    def train(self):
        train_spec = tf.estimator.TrainSpec(input_fn=self.pipeline.get_input_fn())
        eval_spec = tf.estimator.EvalSpec(input_fn=self.pipeline.get_input_fn("val"))
        tf.estimator.train_and_evaluate(self.model, train_spec=train_spec, eval_spec=eval_spec)

    def predict(self, list_of_text):
        return list(self.model.predict(input_fn=self.pipeline.predict_fn(list_of_text)))


def main(argv):
    embed = EmbedModel()
    embed.train()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
