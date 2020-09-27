# This script requires pip install --upgrade google-cloud-vision to run
# This script uses the google cloud credentials in config.json at conf["google-cloud-vision"]["oauth2-path"]

import argparse
import json
import sys

from google.protobuf.json_format import MessageToDict

import io
import os

from google.cloud import vision
from google.cloud.vision import enums
from google.cloud.vision import types


def set_env_var(conf):
    var = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if var != conf["google-cloud-vision"]["oauth2-path"]:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = conf["google-cloud-vision"]["oauth2-path"]
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute facial attributes use Google Vision API")
    parser.add_argument('-m', '--meta_path', action='store', required=True, type=str)
    parser.add_argument('-f', '--index_folder', action='store', default="facenet-vgg2_10", type=str)
    parser.add_argument('-r', '--force_redo', action='store_true')
    parser.add_argument('-p', '--path_config', action='store', type=str, default="config.json")

    script_name = sys.argv[0]
    args = parser.parse_args(sys.argv[1:])

    assert "meta" in args.meta_path

    conf_face = json.load(open(args.path_config, "r"))
    conf_split = json.load(open(os.path.join(conf_face["paths"]["data"], "results", args.index_folder, "ccc.json"), "r"))
    set_env_var(conf_face)

    users = sorted(conf_split["victims"])

    try:
        # Instantiates a client
        client = vision.ImageAnnotatorClient()

        features = [
            types.Feature(type=enums.Feature.Type.LABEL_DETECTION),
            types.Feature(type=enums.Feature.Type.FACE_DETECTION),
        ]

        for u in users:

            output_file = os.path.join(args.meta_path, u, "vision.json")
            if os.path.isfile(output_file) and args.force_redo == False:
                print("[WARN] - skip user %s because vision.json exists already" % u)
                continue

            if not os.path.isdir(os.path.join(args.meta_path, u)):
                print("[WARN] - user %s does not exist" % u)
                continue

            u_folder = os.path.join(args.meta_path, u)

            requests = []

            images_fn = os.listdir(u_folder)
            images_fn = sorted(filter(lambda x: x[-4:] == ".png", images_fn))
            print(u, "%d images" % len(images_fn))

            for image_fn in images_fn:
                image_fp = os.path.join(u_folder, image_fn)
                image_content = io.open(image_fp, 'rb').read()
                image_type = types.Image(content=image_content)
                image_request = types.AnnotateImageRequest(image=image_type, features=features)
                requests.append(image_request)

            n_batches = int(len(requests)/16)
            if len(requests) % 16 != 0:
                n_batches += 1

            try:
                new_dict = {}

                for i in range(n_batches):
                    response = client.batch_annotate_images(requests[i*16:(i+1)*16])
                    response = MessageToDict(response, preserving_proto_field_name=True)

                    for j, r in enumerate(response["responses"]):
                        new_dict[images_fn[i*16:(i+1)*16][j]] = r

                json.dump(new_dict, open(output_file, "w"), indent=2)
            except Exception as e:
                print(e, type(e))
                pass


    except Exception as e:
        print(e)

