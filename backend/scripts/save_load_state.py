# This file is purely for development use, not included in any builds
import argparse
import json
import os
import subprocess

import requests

from alembic import command
from alembic.config import Config
from danswer.configs.app_configs import POSTGRES_DB
from danswer.configs.app_configs import POSTGRES_HOST
from danswer.configs.app_configs import POSTGRES_PASSWORD
from danswer.configs.app_configs import POSTGRES_PORT
from danswer.configs.app_configs import POSTGRES_USER
from danswer.datastores.vespa.store import DOCUMENT_ID_ENDPOINT
from danswer.utils.logger import setup_logger

logger = setup_logger()


def save_postgres(filename: str) -> None:
    logger.info("Attempting to take Postgres snapshot")
    cmd = f"pg_dump -U {POSTGRES_USER} -h {POSTGRES_HOST} -p {POSTGRES_PORT} -W -F t {POSTGRES_DB} > {filename}"
    subprocess.run(
        cmd, shell=True, check=True, input=f"{POSTGRES_PASSWORD}\n", text=True
    )


def load_postgres(filename: str) -> None:
    logger.info("Attempting to load Postgres snapshot")
    try:
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
    except Exception:
        logger.info("Alembic upgrade failed, maybe already has run")
    cmd = f"pg_restore --clean -U {POSTGRES_USER} -h {POSTGRES_HOST} -p {POSTGRES_PORT} -W -d {POSTGRES_DB} -1 {filename}"
    subprocess.run(
        cmd, shell=True, check=True, input=f"{POSTGRES_PASSWORD}\n", text=True
    )


def save_vespa(filename: str) -> None:
    logger.info("Attempting to take Vespa snapshot")
    continuation = ""
    params = {}
    doc_jsons: list[dict] = []
    while continuation is not None:
        if continuation:
            params = {"continuation": continuation}
        response = requests.get(DOCUMENT_ID_ENDPOINT, params=params)
        response.raise_for_status()
        found = response.json()
        continuation = found.get("continuation")
        docs = found["documents"]
        for doc in docs:
            doc_json = {"update": doc["id"], "create": True, "fields": doc["fields"]}
            doc_jsons.append(doc_json)

    with open(filename, "w") as jsonl_file:
        for doc in doc_jsons:
            json_str = json.dumps(doc)
            jsonl_file.write(json_str + "\n")


def load_vespa(filename: str) -> None:
    headers = {"Content-Type": "application/json"}
    with open(filename, "r") as f:
        for line in f:
            new_doc = json.loads(line.strip())
            doc_id = new_doc["update"].split("::")[-1]
            response = requests.post(
                DOCUMENT_ID_ENDPOINT + "/" + doc_id, headers=headers, json=new_doc
            )
            response.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Danswer checkpoint saving and loading."
    )
    parser.add_argument(
        "--save", action="store_true", help="Save Danswer state to directory."
    )
    parser.add_argument(
        "--load", action="store_true", help="Load Danswer state from save directory."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join("..", "danswer_checkpoint"),
        help="A directory to store temporary files to.",
    )

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not args.save and not args.load:
        raise ValueError("Must specify --save or --load")

    if args.load:
        load_postgres(os.path.join(checkpoint_dir, "postgres_snapshot.tar"))
        load_vespa(os.path.join(checkpoint_dir, "vespa_snapshot.jsonl"))
    else:
        save_postgres(os.path.join(checkpoint_dir, "postgres_snapshot.tar"))
        save_vespa(os.path.join(checkpoint_dir, "vespa_snapshot.jsonl"))
