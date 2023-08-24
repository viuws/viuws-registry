from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import click
import pandas as pd
from imageio import imread
from skimage.measure import regionprops_table

FILE_LIST_NAME = "files.txt"


@click.command()
@click.argument(
    "labels_path",
    metavar="LABELS",
    type=click.Path(exists=True, readable=True, path_type=Path),
)
@click.argument(
    "regions_path",
    metavar="REGIONS",
    type=click.Path(writable=True, path_type=Path),
)
@click.option(
    "--images",
    "images_path",
    type=click.Path(exists=True, readable=True, path_type=Path),
)
@click.option(
    "-p",
    "--property",
    "properties",
    multiple=True,
    type=click.STRING,
)
@click.option(
    "--cache/--no-cache",
    "cache",
    default=True,
    show_default=True,
)
@click.option(
    "--separator",
    "separator",
    type=click.STRING,
    default="-",
    show_default=True,
)
@click.option(
    "--spacing",
    "spacing_str",
    type=click.STRING,
)
def cli(
    labels_path: Path,
    regions_path: Path,
    images_path: Optional[Path],
    properties: Optional[list[str]],
    cache: bool,
    separator: str,
    spacing_str: Optional[str],
) -> None:
    labels_urls = _get_urls(labels_path)
    regions_urls = _get_urls(regions_path, names=_get_names(labels_urls, suffix=".csv"))
    images_urls: list[Union[str, None]]
    if images_path is not None:
        images_urls = _get_urls(images_path)
        if len(images_urls) != len(labels_urls):
            raise click.BadParameter("Number of labels and images do not match")
    else:
        images_urls = [None] * len(labels_urls)
    if not properties:
        properties = None
    spacing: Optional[tuple[float, ...]]
    if spacing_str is not None:
        spacing = tuple(float(s) for s in spacing_str.split(","))
    else:
        spacing = None
    for labels_url, images_url, regions_url in zip(
        labels_urls, images_urls, regions_urls
    ):
        labels = imread(labels_url)
        image = imread(images_url) if images_url is not None else None
        regions = pd.DataFrame(
            data=regionprops_table(
                labels,
                intensity_image=image,
                properties=properties,
                cache=cache,
                separator=separator,
                spacing=spacing,
            ),
            index_col="label",
        )
        regions.to_csv(regions_url)
        click.echo(regions_url)
        del labels, image, regions


def _get_urls(path: Path, names: Optional[list[str]] = None) -> list[str]:
    if path.is_file():
        if path.name == FILE_LIST_NAME:
            return path.read_text().splitlines()
        return [str(path)]
    if path.is_dir():
        if names is not None:
            return [str(path / name) for name in names]
        return sorted(str(p) for p in path.glob("*") if p.is_file())
    raise click.BadParameter(f"Not a file or directory: {path}")


def _get_names(urls: list[str], suffix: Optional[str] = None) -> list[str]:
    names: list[str] = []
    for url in urls:
        path = Path(urlparse(url).path)
        if suffix is not None:
            path = path.with_suffix(suffix)
        names.append(path.name)
    return names


if __name__ == "__main__":
    cli()
