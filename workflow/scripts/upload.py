#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import requests


def upload_deposit(
    output,
    file_name,
    deposit_title,
    deposit_description,
    deposit_creators,
    token,
    sandbox=False,
    file_path=".",
):
    metadata = {
        "metadata": {
            "title": deposit_title,
            "upload_type": "dataset",
            "description": deposit_description,
            "creators": [{"name": name} for name in deposit_creators],
        }
    }

    with get_session(token) as session:
        # Search for an existing deposit with the given title
        deposit = find_deposit(session, deposit_title, sandbox=sandbox)

        if deposit:
            deposit_id = update_deposit(
                session,
                deposit,
                file_name,
                metadata,
                sandbox=sandbox,
                file_path=file_path,
            )
        else:
            deposit_id = new_deposit(
                session,
                file_name,
                metadata,
                sandbox=sandbox,
                file_path=file_path,
            )

    # Store the deposit URL
    deposit_url = f"{get_url(sandbox=sandbox)}/record/{deposit_id}"
    with open(output, "w") as f:
        f.write(deposit_url)


def get_session(token):
    session = requests.Session()
    session.params = {"access_token": token}
    return session


def get_url(sandbox=False):
    if sandbox:
        return "https://sandbox.zenodo.org"
    return "https://zenodo.org"


def find_deposit(session, deposit_title, sandbox=False):
    zenodo_url = get_url(sandbox=sandbox)

    # Search for an existing deposit with the given title
    print("Searching for existing deposit...")
    r = session.get(
        f"{zenodo_url}/api/deposit/depositions",
        params={"q": deposit_title},
    )
    if r.status_code != requests.codes.ok:
        return None

    for entry in r.json():
        if entry["title"] == deposit_title:
            return entry
    return None


def update_deposit(
    session,
    deposit,
    file_name,
    metadata,
    sandbox=False,
    file_path=".",
):
    zenodo_url = get_url(sandbox=sandbox)

    # Get the deposit id
    DEPOSIT_ID = deposit["id"]
    deposit_url = f"{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}"

    # Update the existing deposit
    print("Retrieving existing deposit...")

    # Create a new version draft
    r = session.post(f"{deposit_url}/actions/newversion")

    if r.status_code == 403:
        # Seems like we already have a draft. Let's use it
        DEPOSIT_ID = deposit["links"]["latest_draft"].split("/")[-1]
    else:
        r.raise_for_status()
        DEPOSIT_ID = r.json()["links"]["latest_draft"].split("/")[-1]

    # Get the ID of the previously uploaded file (if it exists),
    # then delete it so we can upload a new version.
    print("Deleting old file...")
    r = session.get(f"{deposit_url}/files")
    r.raise_for_status()
    for file in r.json():
        if file["filename"] == file_name:
            FILE_ID = file["id"]
            r = session.delete(f"{deposit_url}/files/{FILE_ID}")
            r.raise_for_status()
            break

    # Upload the new version of the file
    print("Uploading new file...")
    with open(os.path.join(file_path, file_name), "rb") as fp:
        r = session.post(
            f"{deposit_url}/files",
            data={"name": file_name},
            files={"file": fp},
        )
        r.raise_for_status()

    # Add some metadata
    print("Adding metadata...")
    r = session.put(
        deposit_url,
        data=json.dumps(metadata),
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()

    # Publish the deposit
    print("Publishing the deposit...")
    r = session.post(f"{deposit_url}/actions/publish")
    if r.status_code != requests.codes.ok:
        if "New version's files must differ" in r.json().get("message", ""):
            print("No change in the deposit's files. Aborting.")
            DEPOSIT_ID = deposit["links"]["latest_html"].split("/")[-1]
        else:
            r.raise_for_status()

    return DEPOSIT_ID


def new_deposit(
    session,
    file_name,
    metadata,
    sandbox=False,
    file_path=".",
):
    zenodo_url = get_url(sandbox=sandbox)

    print("Creating a new deposit...")
    r = session.post(f"{zenodo_url}/api/deposit/depositions", json={})
    r.raise_for_status()

    # Get the deposit id
    DEPOSIT_ID = r.json()["id"]

    # Upload the file
    print("Uploading the file...")
    bucket_url = r.json()["links"]["bucket"]
    with open(os.path.join(file_path, file_name), "rb") as fp:
        r = session.put(f"{bucket_url}/{file_name}", data=fp)
        r.raise_for_status()

    # Add some metadata
    print("Adding metadata...")
    r = session.put(
        f"{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}",
        data=json.dumps(metadata),
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()

    # Publish the deposit
    print("Publishing the deposit...")
    r = session.post(
        f"{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/actions/publish"
    )
    r.raise_for_status()
    return DEPOSIT_ID


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-d", "--metadata", required=True, type=str)
    parser.add_argument("-c", "--creds", required=True, type=str)
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args()

    with open(args.creds, "r") as f:
        token = f.read().strip()

    with open(args.metadata, "r") as f:
        metadata = yaml.load(f.read(), Loader=yaml.FullLoader)

    file_path, file_name = os.path.split(args.input)
    upload_deposit(
        args.output,
        file_name,
        metadata["title"],
        metadata["description"],
        metadata["creators"],
        token,
        sandbox=args.sandbox,
        file_path=file_path,
    )
