import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

from absl import app
from absl import flags
from ml_collections import config_flags

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import train

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", ".", "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    "./configs/table_final_mlp_seed42.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    else:
        raise NotImplementedError(f"Unsupported mode {FLAGS.config.mode}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
