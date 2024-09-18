import os
import re
import yaml

# results_folder = "../scripts_paper/AAruns_hybrid_single_mlp32x5_256b_euler_lr4_150ep"
# conditions_to_check = {
#     "batch_size": 256,
#     "epochs": 150,
#     "hidden_size": [32, 32, 32, 32, 32],
#     "learning_rate": 0.0001
# }

# results_folder = "../scripts_paper/AAruns_hybrid_single_mlp32x5_256b_bosh3_lr4_200ep"
# conditions_to_check = {
#     "batch_size": 256,
#     "epochs": 200,
#     "hidden_size": [32, 32, 32, 32, 32],
#     "learning_rate": 0.0001,
#     "odesmethod": "bosh3"
# }

results_folder = "../../scripts_paper/569basins_single_mlp32x5_7305b_euler1d_lr4_150ep_1000pre_lr3"
conditions_to_check = {
    "batch_size": -1,
    "epochs": 150,
    "hidden_size": [32, 32, 32, 32, 32],
    "learning_rate": 0.0001
}


# results_folder = "../scripts_paper/AAruns_hybrid_single_mlp32x5_256b_bosh3_lr34_200ep"
# conditions_to_check = {
#     "batch_size": 256,
#     "epochs": 200,
#     "hidden_size": [32, 32, 32, 32, 32],
#     "learning_rate": {
#         "decay": 0.5,
#         "decay_step_fraction": 2,
#         "initial": 0.001
#     },
#     "odesmethod": "bosh3"
# }



ONLY_TEST_DONOT_DELETE = True

def main(results_folder):

    jobs = sorted([folder for folder in os.listdir(results_folder) if \
                   re.search(r'\d{8}', folder) is not None])
    print(f"Found {len(jobs)} jobs in {results_folder}")

    for job in jobs[:]:
        job_folder = os.path.join(results_folder, job)
        # print(f"Processing {job}")
        try:
            with open(os.path.join(job_folder, "config.yml"), "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            # print(f"config.yml not found in {job}. Skipping.")
            continue

        # Check if the conditions match
        match = all(config.get(key) == value for key, value in conditions_to_check.items())
        if not match:
            print(f"Deleting {job} as it does not meet conditions.")
            if os.path.exists(job_folder):
                print(f"Folder {job} exists. Deleting.")
                if not ONLY_TEST_DONOT_DELETE:
                    os.system(f"rm -r \"{job_folder}\"")
            else:
                print(f"Folder {job_folder} does not exist. Cannot delete.")

            # aux = input("Press enter to continue ...")
        else:
            # print(f"{job} meets the conditions. Keeping it.")
            continue


if __name__ == "__main__":

    main(results_folder)