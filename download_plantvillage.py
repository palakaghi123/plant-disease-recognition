import argparse
import binascii
import struct
import zlib
from pathlib import Path, PurePosixPath

import requests
from remotezip import RemoteZip

from plant_disease import CLASS_NAMES


DATASET_URL = (
    "https://huggingface.co/datasets/mohanty/PlantVillage/"
    "resolve/main/data.zip"
)
RANGE_CHUNK_SIZE = 2 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class RangeCache:
    def __init__(self, session, file_size):
        self._session = session
        self._file_size = file_size
        self._blocks = {}

    def read(self, start, length):
        content = bytearray()
        while length:
            block_start = (start // RANGE_CHUNK_SIZE) * RANGE_CHUNK_SIZE
            if block_start not in self._blocks:
                block_end = min(
                    block_start + RANGE_CHUNK_SIZE - 1,
                    self._file_size - 1,
                )
                response = self._session.get(
                    DATASET_URL,
                    headers={"Range": f"bytes={block_start}-{block_end}"},
                    timeout=120,
                )
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {block_start}-"):
                    raise RuntimeError(
                        f"Dataset server returned an invalid range: {content_range!r}"
                    )
                self._blocks[block_start] = response.content

            block = self._blocks[block_start]
            offset = start - block_start
            count = min(length, len(block) - offset)
            if count <= 0:
                raise RuntimeError("Dataset server returned an incomplete byte range.")
            content.extend(block[offset : offset + count])
            start += count
            length -= count
        return bytes(content)


def read_image(range_cache, info):
    header = range_cache.read(info.header_offset, 30)
    (
        signature,
        _,
        _,
        compression,
        _,
        _,
        _,
        _,
        _,
        filename_length,
        extra_length,
    ) = struct.unpack("<IHHHHHIIIHH", header)
    if signature != 0x04034B50:
        raise RuntimeError(f"Invalid ZIP header for {info.filename}")

    data_offset = info.header_offset + 30 + filename_length + extra_length
    compressed = range_cache.read(data_offset, info.compress_size)
    if compression == 0:
        image_bytes = compressed
    elif compression == 8:
        image_bytes = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(
            f"Unsupported ZIP compression method {compression} for {info.filename}"
        )

    if len(image_bytes) != info.file_size or binascii.crc32(image_bytes) != info.CRC:
        raise RuntimeError(f"Dataset image failed ZIP integrity checks: {info.filename}")
    return image_bytes


def download_dataset(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = set(CLASS_NAMES)

    with requests.Session() as session, RemoteZip(DATASET_URL) as archive:
        range_cache = RangeCache(session, archive.size())
        image_files = []
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (
                len(path.parts) == 4
                and path.parts[:2] == ("raw", "color")
                and path.parts[2] in class_names
                and path.suffix.lower() in IMAGE_SUFFIXES
            ):
                image_files.append(info)

        if not image_files:
            raise RuntimeError("No matching color images were found in PlantVillage.")

        for index, info in enumerate(image_files, start=1):
            path = PurePosixPath(info.filename)
            destination = output_dir / path.parts[2] / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(read_image(range_cache, info))
            if index % 500 == 0 or index == len(image_files):
                print(f"Downloaded {index}/{len(image_files)} images")

    print(f"Dataset saved to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the four PlantVillage classes used by this project."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "PlantVillage",
    )
    args = parser.parse_args()
    download_dataset(args.output)


if __name__ == "__main__":
    main()
