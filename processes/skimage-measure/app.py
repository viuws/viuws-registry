from typing import Optional, Union

import click
import pandas as pd
from imageio import imread
from skimage.measure import regionprops_table
from upath import UPath


@click.command()
@click.argument(
    "labels_files_uri",
    metavar="LABELS",
    type=click.STRING,
)
@click.argument(
    "regions_files_uri",
    metavar="REGIONS",
    type=click.STRING,
)
@click.option(
    "-i",
    "--images",
    "image_files_uri",
    type=click.STRING,
    help="URI of images or image file list.",
)
@click.option(
    "-p",
    "--property",
    "properties",
    type=click.STRING,
    multiple=True,
    help=(
        "Property that will be included in the resulting table. "
        "Specify multiple times to include multiple properties. "
        "The `label` property is automatically included. "
        "See https://scikit-image.org/docs/stable/api/skimage.measure.html"
    ),
)
@click.option(
    "--by-name/--alphabetically",
    "by_name",
    default=False,
    show_default=True,
    help="Determines whether to match images to labels by name or alphabetically.",
)
@click.option(
    "--cache/--no-cache",
    "cache",
    default=True,
    show_default=True,
    help=(
        "Determine whether to cache calculated properties. "
        "The computation is much faster for cached properties, "
        "whereas the memory consumption increases."
    ),
)
@click.option(
    "--spacing",
    "spacing_str",
    type=click.STRING,
    help=(
        "The pixel spacing along each axis of the image. "
        "Specify as a comma-separated list, with one value for each axis."
    ),
)
def cli(
    labels_files_uri: str,
    regions_files_uri: str,
    image_files_uri: Optional[str],
    properties: Optional[list[str]],
    by_name: bool,
    cache: bool,
    spacing_str: Optional[str],
) -> None:
    labels_files = get_files(labels_files_uri)
    labels_file_names = [labels_file.name for labels_file in labels_files]
    regions_files = get_files(
        regions_files_uri, file_names=labels_file_names, suffix=".csv"
    )
    if image_files_uri is not None:
        image_files = get_files(
            image_files_uri, file_names=labels_file_names if by_name else None
        )
        if len(image_files) != len(labels_files):
            raise click.BadParameter(
                "Number of image files does not match number of labels files"
            )
    else:
        image_files = [None] * len(labels_files)
    if properties:
        if "label" not in properties:
            properties.append("label")
    else:
        properties = None
    if spacing_str is not None:
        spacing = tuple(float(s) for s in spacing_str.split(","))
    else:
        spacing = None
    for labels_file, image_file, regions_file in zip(
        labels_files, image_files, regions_files
    ):
        with labels_file.open("rb") as f:
            labels = imread(f)
        if image_file is not None:
            with image_file.open("rb") as f:
                image = imread(f)
        else:
            image = None
        regions = pd.DataFrame(
            data=regionprops_table(
                labels,
                intensity_image=image,
                properties=properties,
                cache=cache,
                spacing=spacing,
            ),
            index_col="label",
        )
        with regions_file.open("w") as f:
            regions.to_csv(f)
        del labels, image, regions


def get_files(
    path: Union[str, UPath],
    pattern: str = "*",
    file_names: Optional[list[str]] = None,
    suffix: Optional[str] = None,
) -> list[UPath]:
    if not isinstance(path, UPath):
        path = UPath(path).resolve()
    if path.is_file():
        if path.suffix.lower() == ".list":
            file_paths = [
                UPath(line).resolve() for line in path.read_text().splitlines() if line
            ]
        else:
            file_paths = [path]
    elif path.is_dir():
        if file_names is not None:
            file_paths = [path / file_name for file_name in file_names]
        else:
            file_paths = sorted(p for p in path.glob(pattern) if p.is_file())
    else:
        raise ValueError("path")
    if suffix is not None:
        file_paths = [file_path.with_suffix(suffix) for file_path in file_paths]
    return file_paths


if __name__ == "__main__":
    cli(prog_name="skimage-measure")
