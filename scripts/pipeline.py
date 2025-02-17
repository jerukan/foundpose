#!/usr/bin/env python3
import click
import yaml

from utils import config_util
from utils.run import opts, templates, repre, infer


@click.command()
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--gen-templates",
    "gen_templates",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Generate render templates",
)
@click.option(
    "--gen-repre",
    "gen_repre",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Generate BoW representations for templates",
)
@click.option(
    "--infer",
    "_infer",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Run inference on a dataset",
)
def main(cfg_path, gen_templates, gen_repre, _infer):
    """
    Yes _infer is the variable name. No I don't care.
    """
    with open(cfg_path, "rt") as f:
        cfg = yaml.safe_load(f)
    allopts = config_util.load_opts_from_raw_dict(
        cfg,
        {
            "common_opts": opts.CommonOpts,
            "gen_templates_opts": opts.GenTemplatesOpts,
            "gen_repre_opts": opts.GenRepreOpts,
            "infer_opts": opts.InferOpts
        }
    )
    commonopts = allopts["common_opts"]
    templateopts = allopts["gen_templates_opts"]
    repreopts = allopts["gen_repre_opts"]
    inferopts = allopts["infer_opts"]
    if gen_templates:
        templates.synthesize_templates(commonopts, templateopts)
    if gen_repre:
        repre.generate_repre_from_list(commonopts, repreopts)
    if _infer:
        infer.infer(commonopts, inferopts)


if __name__ == "__main__":
    main()
