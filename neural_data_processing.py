from glob import glob
import os
import numpy as np
import pandas as pd 
import nibabel as nib
import nilearn
from nilearn.maskers import NiftiMasker
from tqdm import tqdm


def load_mask(mask_path):
    """Load the hippocampus ROI mask."""
    return nib.load(mask_path)


def load_beta_images(betas_path, participant_id, run):
    """Load beta images for a specific participant and run."""
    beta_images_folder = os.path.join(betas_path, f"betas_{participant_id}_run{run}")
    image_paths = sorted(glob(f"{beta_images_folder}/beta_*.nii"))
    return [nib.load(image) for image in image_paths]


def resample_mask(mask, target_affine, target_shape):
    """Resample the mask to match target affine and shape."""
    return nilearn.image.resample_img(mask, target_affine=target_affine, target_shape=target_shape,
                                      interpolation='nearest', fill_value=0)


def mask_beta_images(beta_images, masker):
    """Mask the beta images using the given masker."""
    return [masker.fit_transform(image)[0] for image in beta_images]


def load_participant_data(general_path, participant_id, run):
    """Load behavioral data for a specific participant and run."""
    filename = f"p{participant_id}_Run{run}.csv"
    participant_path = os.path.join(general_path, f"sub-{participant_id:02d}", filename)
    return pd.read_csv(participant_path)


def process_participant(roi_mask, betas_path, general_path, participant_id, runs):
    """Process data for a single participant."""
    all_participant_data = []

    for run in tqdm(runs, desc=f"Processing participant {participant_id}"):
        # Load beta images
        beta_images = load_beta_images(betas_path, participant_id, run)

        # Resample the mask to match beta images
        mean_participant_mask = nib.load(f"Mean_Func_Participant/sub-{participant_id:02d}_Mean_Func.nii.gz")
        resampled_mask = resample_mask(roi_mask, target_affine=mean_participant_mask.affine,
                                       target_shape=mean_participant_mask.get_fdata().shape)

        # Create masker object
        masker = NiftiMasker(mask_img=resampled_mask)

        # Mask beta images
        masked_betas = mask_beta_images(beta_images, masker)

        # Load participant behavioral data
        participant_data = load_participant_data(general_path, participant_id, run)

        # Extract shape and condition information
        shape = participant_data["Shape"].iloc[0]
        condition = participant_data["Condition"].iloc[0]

        # Create run dictionary
        run_dictionary = {"Functional": masked_betas, "Participant": participant_id, "Run": run,
                          "Shape": shape, "Condition": condition}

        all_participant_data.append(run_dictionary)

    return all_participant_data


def main():
    # Set paths and parameters
    betas_path = "classifier_betas_encoding"
    general_path = "Behavior"
    roi_mask_path = "BilatHippocampalAAL.nii"
    people = [2, 3, 4, 7, 8, 9, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33]
    runs = [1, 2, 3, 4]

    # Load hippocampus ROI mask
    hc_mask = load_mask(roi_mask_path)

    # Process data for each participant
    all_data = []
    for participant_id in people:
        participant_data = process_participant(hc_mask, betas_path, general_path, participant_id, runs)
        all_data.extend(participant_data)

    # Save processed data
    with open('processed_data.pickle', 'wb') as f:
        pickle.dump(all_data, f)

    # Load and print processed data
    with open('processed_data.pickle', 'rb') as f:
        loaded_data = pickle.load(f)
    print(f"Processed data: {loaded_data}")


if __name__ == "__main__":
    main()
