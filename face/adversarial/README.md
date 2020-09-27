## Main Scripts
 * `start_face.py`: runs the first optimization to setup the attack (Algorithm 1, Line 3 in the paper)
 * `gen_face.py`: runs the second optimization of the attack (Algorithm 1, Line 8 in the paper)
 * `atk_face.py`: runs the attack with consecutive injections (Algorithm 2 in the paper)
 * `countermeasure.py`: computes which samples are detected by our countermeasure.
 * `apply_cm.py`: applies the countermeasure to successful attacks, showing which attacks are detected.
 
## Other Functions
 * `get_pairs.py`: selects 1,000 adversary-victim pairs to use for evaluation
 * `run_many_face.py`: utility to run above scripts in one go
 * `tf_utils.py` and `adv_utils.py`: utilities
 
 