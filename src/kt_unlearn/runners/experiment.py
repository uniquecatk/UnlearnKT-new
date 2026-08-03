from __future__ import annotations

import argparse
import copy
import tracemalloc
from pathlib import Path

from kt_unlearn.core.factory_base import ConfigurableFactory
from kt_unlearn.utils.config.global_ctx import Global, bcolors
from kt_unlearn.utils.config.local_ctx import Local


def run_config(config_file: str | Path):
    config_path = Path(config_file).resolve()
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    global_ctx = Global(str(config_path))
    global_ctx.factory = ConfigurableFactory(global_ctx)

    data_manager = global_ctx.factory.get_object(Local(global_ctx.config.data))
    global_ctx.dataset = data_manager

    current = Local(global_ctx.config.predictor)
    current.dataset = data_manager
    predictor = global_ctx.factory.get_object(current)
    global_ctx.predictor = predictor
    global_ctx.logger.info("Global Predictor: %s", predictor)

    unlearners = []
    for unlearner_cfg in global_ctx.config.unlearners:
        current = Local(unlearner_cfg)
        current.dataset = data_manager
        current.predictor = copy.deepcopy(predictor)
        unlearners.append(global_ctx.factory.get_object(current))

    current = Local(global_ctx.config.evaluator)
    current.unlearners = unlearners
    evaluator = global_ctx.factory.get_object(current)

    for unlearner in unlearners:
        global_ctx.logger.info(
            "%s####\t\t Evaluating Unlearner %s \t\t####%s",
            bcolors.OKGREEN,
            unlearner.__class__.__name__,
            bcolors.ENDC,
        )
        evaluator.evaluate(unlearner, predictor)

    return {
        "global_ctx": global_ctx,
        "dataset": data_manager,
        "predictor": predictor,
        "unlearners": unlearners,
        "evaluator": evaluator,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KT Unlearning Framework.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    return parser


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    run_config(args.config_file)


if __name__ == "__main__":
    main()
