import os
import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from puts import printc, timestamp_seconds
from torch.utils.data import ConcatDataset
from torchinfo import summary

from DataLoader import lfcc, load_directory_split_train_test, mfcc
from models.cnn import ShallowCNN
from models.lstm import SimpleLSTM, WaveLSTM
from models.mlp import MLP
from models.rnn import WaveRNN
from models.tssd import TSSD
from trainer import ModelTrainer
from utils import set_seed_all

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)


# all feature classnames
FEATURE_CLASSNAMES: Tuple[str] = ("wave", "lfcc", "mfcc")
# all model classnames
MODEL_CLASSNAMES: Tuple[str] = (
    "MLP",
    "WaveRNN",
    "WaveLSTM",
    "SimpleLSTM",
    "ShallowCNN",
    "TSSD",
)
# all model keyword arguments
KWARGS_MAP: Dict[str, dict] = {
    "SimpleLSTM": {
        "lfcc": {"feat_dim": 40, "time_dim": 972, "mid_dim": 30, "out_dim": 1},
        "mfcc": {"feat_dim": 40, "time_dim": 972, "mid_dim": 30, "out_dim": 1},
    },
    "ShallowCNN": {
        "lfcc": {"in_features": 1, "out_dim": 1},
        "mfcc": {"in_features": 1, "out_dim": 1},
    },
    "MLP": {
        "lfcc": {"in_dim": 40 * 972, "out_dim": 1},
        "mfcc": {"in_dim": 40 * 972, "out_dim": 1},
    },
    "TSSD": {
        "wave": {"in_dim": 64600},
    },
    "WaveRNN": {
        "wave": {"num_frames": 10, "input_length": 64600, "hidden_size": 500},
    },
    "WaveLSTM": {
        "wave": {
            "num_frames": 10,
            "input_len": 64600,
            "hidden_dim": 30,
            "out_dim": 1,
        }
    },
}


def init_logger(log_file: Union[Path, str]) -> None:
    # create file handler
    fh = logging.FileHandler(log_file)
    # create console handler
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # clear handlers
    LOGGER.handlers = []
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)
    return None


def train(
    training_dir: Union[Path, str],
    validation_dir: Union[Path, str],
    amount_to_use: int = None,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available else "cpu",
    batch_size: int = 32,
    save_dir: Union[str, Path] = None,
    test_size: float = 0.2,
    feature_classname: str = "wave",
    model_classname: str = "SimpleLSTM",
    in_distribution: bool = True,
    checkpoint=None,
) -> None:
    """
    Train a model on WaveFake data.

    Args:
        training_dir:
            path to training dataset directory
        validation_dir:
            path to validation dataset directory
        amount_to_use:
            amount of data to use (if None, use all) (default: None)
        epochs:
            number of epochs to train for (default: 20)
        device:
            device to use (default: "cuda" if available)
        batch_size:
            batch size (default: 32)
        save_dir:
            directory to save model checkpoints to (default: None)
        test_size:
            ratio of test set / whole dataset (default: 0.2)
        feature_classname:
            classname of feature extractor (possible: "wave", "mfcc", "lfcc")
        model_classname:
            classname of model (possible: "SimpleLSTM", "ShallowCNN", "WaveLSTM", "MLP")
        in_distribution:
            whether to use in-distribution data (default: True)
                - True: use 1:1 real:fake data (split melgan for training and test)
                - False: use 1:7 real:fake data (use melgan for test only, others for training)

    Returns:
        None
    """
    feature_classname = feature_classname.lower()
    assert feature_classname in FEATURE_CLASSNAMES
    assert model_classname in MODEL_CLASSNAMES

    # get feature transformation function
    feature_fn = None if feature_classname == "wave" else eval(feature_classname)
    assert feature_fn in (None, lfcc, mfcc)
    # get model constructor
    Model = eval(model_classname)
    assert Model in (SimpleLSTM, ShallowCNN, WaveLSTM, MLP, TSSD, WaveRNN)

    model_kwargs: dict = KWARGS_MAP.get(model_classname).get(feature_classname)
    if model_kwargs is None:
        raise ValueError(
            f"model_kwargs not found for {model_classname} and {feature_classname}"
        )
    model_kwargs.update({"device": device})

    LOGGER.info(f"Training model: {model_classname}")
    LOGGER.info(f"Input feature : {feature_classname}")
    LOGGER.info(f"Model kwargs  : {json.dumps(model_kwargs, indent=2)}")

    ###########################################################################

    # TODO: FIXME
    real_train_dir = "for2sec/training/real"
    real_val_dir = "for2sec/validation/real"
    fake_train_dir = "for2sec/training/fake"
    fake_val_dir = "for2sec/validation/fake"

    real_train_dir = Path(real_train_dir)
    real_val_dir = Path(real_val_dir)
    fake_train_dir = Path(fake_train_dir)
    fake_val_dir = Path(fake_val_dir)
    assert real_train_dir.is_dir()
    assert real_val_dir.is_dir()
    assert fake_train_dir.is_dir()
    assert fake_val_dir.is_dir()

    LOGGER.info("Loading data...")

    _, real_dataset_train = load_directory_split_train_test(
        path=real_train_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=1.0,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=1,
        amount_to_use=amount_to_use,
    )

    _, real_dataset_validate = load_directory_split_train_test(
        path=real_val_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=1.0,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=1,
        amount_to_use=amount_to_use,
    )

    _, fake_dataset_train= load_directory_split_train_test(
        path=fake_train_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=1.0,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=0,
        amount_to_use=amount_to_use,
    )

    _, fake_dataset_validate = load_directory_split_train_test(
        path=fake_val_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=1.0,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=0,
        amount_to_use=amount_to_use,
    )

    print("real_dataset_train", len(real_dataset_train))
    print("real_dataset_validate", len(real_dataset_validate))
    print("fake_dataset_train", len(fake_dataset_train))
    print("fake_dataset_validate", len(fake_dataset_validate))

    dataset_train, dataset_validate = None, None

    # ljspeech (real) and melgan (fake) are split into train and validate
    dataset_train = ConcatDataset([real_dataset_train, fake_dataset_train])
    dataset_validate = ConcatDataset([real_dataset_validate, fake_dataset_validate])
    pos_weight = len(real_dataset_train) / len(fake_dataset_validate)


    ###########################################################################

    LOGGER.info(f"Training model on {len(dataset_train)} audio files.")
    LOGGER.info(f"Testing model on  {len(dataset_validate)} audio files.")
    LOGGER.info(f"Train/Validate ratio: {len(dataset_train) / len(dataset_validate)}")
    LOGGER.info(f"Real/Fake ratio in training: {round(pos_weight, 3)} (pos_weight)")

    pos_weight = torch.Tensor([pos_weight]).to(device)

    model = Model(**model_kwargs).to(device)
    input_size = (
        (batch_size, 64600) if feature_classname == "wave" else (batch_size, 40, 972)
    )
    model_stats = summary(model, input_size, verbose=0)
    summary_str = str(model_stats)
    LOGGER.info(f"Model summary:\n{summary_str}")

    ###########################################################################

    ModelTrainer(
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        lr=0.0005,
        optimizer_kwargs={"weight_decay": 0.0001},
    ).train(
        model=model,
        dataset_train=dataset_train,
        dataset_test=dataset_validate,
        save_dir=save_dir,
        pos_weight=pos_weight,
        checkpoint=checkpoint,
    )


def evaluate_only(
    testing_dir: Union[Path, str],
    amount_to_use: int = None,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available else "cpu",
    batch_size: int = 32,
    save_dir: Union[str, Path] = None,
    test_size: float = 0.2,
    feature_classname: str = "wave",
    model_classname: str = "SimpleLSTM",
    in_distribution: bool = True,
    checkpoint=None,
) -> None:
    """
    Train a model on WaveFake data.

    Args:
        testing_dir:
            path to testing dataset directory
        amount_to_use:
            amount of data to use (if None, use all) (default: None)
        epochs:
            number of epochs to train for (default: 20)
        device:
            device to use (default: "cuda" if available)
        batch_size:
            batch size (default: 32)
        save_dir:
            directory to save model checkpoints to (default: None)
        test_size:
            ratio of test set / whole dataset (default: 0.2)
        feature_classname:
            classname of feature extractor (possible: "wave", "mfcc", "lfcc")
        model_classname:
            classname of model (possible: "SimpleLSTM", "ShallowCNN", "WaveLSTM", "MLP")
        in_distribution:
            whether to use in-distribution data (default: True)
                - True: use 1:1 real:fake data (split melgan for training and test)
                - False: use 1:7 real:fake data (use melgan for test only, others for training)

    Returns:
        None
    """
    feature_classname = feature_classname.lower()
    assert feature_classname in FEATURE_CLASSNAMES
    assert model_classname in MODEL_CLASSNAMES

    # get feature transformation function
    feature_fn = None if feature_classname == "wave" else eval(feature_classname)
    assert feature_fn in (None, lfcc, mfcc)
    # get model constructor
    Model = eval(model_classname)
    assert Model in (SimpleLSTM, ShallowCNN, WaveLSTM, MLP, TSSD, WaveRNN)

    model_kwargs: dict = KWARGS_MAP.get(model_classname).get(feature_classname)
    if model_kwargs is None:
        raise ValueError(
            f"model_kwargs not found for {model_classname} and {feature_classname}"
        )
    model_kwargs.update({"device": device})

    LOGGER.info(f"Evaluating model: {model_classname}")
    LOGGER.info(f"Input feature : {feature_classname}")
    LOGGER.info(f"Model kwargs  : {json.dumps(model_kwargs, indent=2)}")

    ###########################################################################

    real_dir = Path(os.path.join(testing_dir, "real"))
    fake_dir = Path(os.path.join(testing_dir, "fake"))
    assert real_dir.is_dir()
    assert fake_dir.is_dir()

    LOGGER.info("Loading data...")

    _, real_dataset_test = load_directory_split_train_test(
        path=real_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=1.0,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=1,
        amount_to_use=amount_to_use,
    )

    _, fake_dataset_test = load_directory_split_train_test(
        path=fake_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=1.0,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=0,
        amount_to_use=amount_to_use,
    )

    dataset_test = ConcatDataset([real_dataset_test, fake_dataset_test])

    ###########################################################################

    # LOGGER.info(f"Training model on {len(dataset_train)} audio files.")
    LOGGER.info(f"Testing model on  {len(dataset_test)} audio files.")
    # LOGGER.info(f"Train/Test ratio: {len(dataset_train) / len(dataset_test)}")
    # LOGGER.info(f"Real/Fake ratio in training: {round(pos_weight, 3)} (pos_weight)")

    # pos_weight = torch.Tensor([pos_weight]).to(device)

    model = Model(**model_kwargs).to(device)
    input_size = (
        (batch_size, 64600) if feature_classname == "wave" else (batch_size, 40, 972)
    )
    model_stats = summary(model, input_size, verbose=0)
    summary_str = str(model_stats)
    LOGGER.info(f"Model summary:\n{summary_str}")

    ###########################################################################

    ModelTrainer(batch_size=batch_size, epochs=epochs, device=device).eval(
        model=model,
        dataset_test=dataset_test,
        save_dir=save_dir,
        checkpoint=checkpoint,
    )


def experiment(
    name: str,
    training_dir: str,
    validation_dir: str,
    testing_dir: str,
    epochs: int,
    batch_size: int,
    feature_classname: str,
    model_classname: str,
    in_distribution: bool,
    device: str,
    seed: Optional[int] = None,
    amount_to_use: Union[int, None] = None,
    restore: bool = False,
    eval_only: bool = False,
    **kwargs,
):

    root_save_dir = Path("saved")
    save_dir = root_save_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / f"{timestamp_seconds()}.log"
    restore_path = save_dir / "best.pt"
    if restore and restore_path.is_file():
        LOGGER.info(f"Restoring from {restore_path}")
        ckpt = torch.load(restore_path, map_location=lambda storage, loc: storage)
    else:
        ckpt = None

    init_logger(log_file)
    if seed is not None:
        set_seed_all(seed)

    LOGGER.info(f"Batch size: {batch_size}, seed: {seed}, epochs: {epochs}")

    if eval_only:
        evaluate_only(
            testing_dir=testing_dir,
            amount_to_use=amount_to_use,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            save_dir=save_dir,
            feature_classname=feature_classname,
            model_classname=model_classname,
            in_distribution=in_distribution,
            checkpoint=ckpt,
        )
    else:
        train(
            training_dir=training_dir,
            validation_dir=validation_dir,
            amount_to_use=amount_to_use,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            save_dir=save_dir,
            feature_classname=feature_classname,
            model_classname=model_classname,
            in_distribution=in_distribution,
            checkpoint=ckpt,
        )


def debug(real_dir: str, fake_dir: str, device: str):
    for model_classname in KWARGS_MAP.keys():
        for feature_classname in KWARGS_MAP[model_classname].keys():
            for in_distribution in [True, False]:
                exp_setup = "I" if in_distribution else "O"
                exp_name = f"{model_classname}_{feature_classname}_{exp_setup}"
                try:
                    printc(f">>>>> DEBUGGING: {exp_name}")
                    experiment(
                        name="debug",
                        seed=0,
                        epochs=3,
                        batch_size=16,
                        device=device,
                        feature_classname=feature_classname,
                        model_classname=model_classname,
                        in_distribution=in_distribution,
                        real_dir=real_dir,
                        fake_dir=fake_dir,
                        amount_to_use=160,
                    )
                    printc(f">>>>> DEBUGGING Done: {exp_name}\n\n")
                except Exception as e:
                    printc(f">>>>> DEBUGGING Failed: {exp_name}\n\n", color="red")
                    LOGGER.exception(e)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training",
        help="Directory containing training data. (default: 'for2sec/training')",
        type=str,
        default="for2sec/training",
    )
    parser.add_argument(
        "--validation",
        help="Directory containing validation data. (default: 'for2sec/validation')",
        type=str,
        default="for2sec/validation",
    )
    parser.add_argument(
        "--testing",
        help="Directory containing testing data. (default: 'for2sec/testing')",
        type=str,
        default="for2sec/testing",
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size. (default: 256)",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--epochs",
        help="Number of maximum epochs to train. (default: 20)",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        help="Random seed. (default: 42)",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--feature_classname",
        help="Feature classname. (default: 'lfcc')",
        choices=FEATURE_CLASSNAMES,
        type=str,
        default="lfcc",
    )
    parser.add_argument(
        "--model_classname",
        help="Model classname. (default: 'ShallowCNN')",
        choices=MODEL_CLASSNAMES,
        type=str,
        default="ShallowCNN",
    )
    parser.add_argument(
        "--in_distribution",
        "--in_dist",
        help="Whether to use in distribution experiment setup. (default: True)",
        choices=["True", "False"],
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--device",
        help="Device to use. (default: 'cuda' if possible)",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--deterministic",
        help="Whether to use deterministic training (reproducible results).",
        action="store_true",
    )
    parser.add_argument(
        "--restore",
        help="Whether to restore from checkpoint.",
        action="store_true",
    )
    parser.add_argument(
        "--eval_only",
        help="Whether to evaluate only.",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Whether to use debug mode.",
        action="store_true",
    )
    parser.add_argument(
        "--debug_all",
        help="Whether to use debug mode for all models.",
        action="store_true",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.debug_all:
        debug(args.real_dir, args.fake_dir, args.device)
        return

    exp_setup = "I" if args.in_distribution else "O"
    exp_name = f"{args.model_classname}_{args.feature_classname}_{exp_setup}"

    if args.debug:
        exp_name = "debug"
    try:
        printc(f">>>>> Starting experiment: {exp_name}")
        experiment(
            name=exp_name,
            training_dir=args.training,
            validation_dir=args.validation,
            testing_dir=args.testing,
            epochs=args.epochs,
            batch_size=args.batch_size,
            feature_classname=args.feature_classname,
            model_classname=args.model_classname,
            in_distribution=args.in_distribution,
            device=args.device,
            seed=args.seed if args.deterministic else None,
            amount_to_use=160 if args.debug else None,
            restore=args.restore,
            eval_only=args.eval_only,
        )
        printc(f">>>>> Experiment Done: {exp_name}\n\n")
    except Exception as e:
        printc(f">>>>> Experiment Failed: {exp_name}\n\n", color="red")
        LOGGER.exception(e)


if __name__ == "__main__":
    main()
