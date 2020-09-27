# Brief Descriptions
 * `age-estimation.py`: uses a model to estimate age from facial images
 * `create_data_split.py`: splits a folder containing a dataset into training/dev/testing
 * `face_utils.py`: just utils
 * `models.py`: implementation of face recognition models subclassing `cleverhans.model.Model` to run adversarial optimization
 * `preprocess.py`: aligns face dataset and computes face landmarks for glasses positioning
 * `vision-api.py`: uses Google Vision API to extract face attributes