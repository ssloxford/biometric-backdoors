# this script uses the estimator at https://github.com/IBM/MAX-Facial-Age-Estimator
# you can have a local copy of the model by getting it from doker with
# docker run -it -p 5000:5000 codait/max-facial-age-estimator

import argparse
import json
import sys
import io
import os
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimates age for facial images using the estimator at"
                                                 "https://github.com/IBM/MAX-Facial-Age-Estimator")
    parser.add_argument('-m', '--meta_path', action='store', required=True, type=str)
    parser.add_argument('-f', '--index_folder', action='store', default="facenet-vgg2_10", type=str)
    parser.add_argument('-r', '--force_redo', action='store_true')
    parser.add_argument('-p', '--path_config', action='store', type=str, default="config.json")

    script_name = sys.argv[0]
    args = parser.parse_args(sys.argv[1:])

    assert "meta" in args.meta_path

    conf_face = json.load(open(args.path_config, "r"))
    conf_split = json.load(open(os.path.join(conf_face["paths"]["data"], "results", args.index_folder, "ccc.json"), "r"))

    users = sorted(conf_split["victims"])

    try:

        for u in users:

            output_file = os.path.join(args.meta_path, u, "max-age-estimate.json")
            if os.path.isfile(output_file) and args.force_redo == False:
                print("[WARN] - skip user %s because max-age-estimate.json exists already" % u)
                continue

            if not os.path.isdir(os.path.join(args.meta_path, u)):
                print("[WARN] - user %s does not exist" % u)
                continue

            u_folder = os.path.join(args.meta_path, u)

            images_fn = os.listdir(u_folder)
            images_fn = sorted(filter(lambda x: x[-4:] == ".png", images_fn))
            print(u, "%d images" % len(images_fn))

            out_dict = {}

            for image_fn in images_fn:
                img_fp = os.path.join(u_folder, image_fn)
                image_content = io.open(img_fp, 'rb').read()

                command = "curl -F 'image=@%s' -XPOST http://localhost:5000/model/predict > here.tmp" % img_fp
                print(command)
                os.system(command)
                time.sleep(.01)
                out = json.load(open("here.tmp", "r"))
                out_dict[image_fn] = out

            json.dump(out_dict, open(output_file, "w"), indent=2)

    except Exception as e:
        print(e)

