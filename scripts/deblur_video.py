import numpy as np
from PIL import Image
import click
import cv2

from deblurgan.model import generator_model
from deblurgan.utils import deprocess_image, preprocess_image

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')


@click.command()
@click.option('--weight_path', help='Model weight')
@click.option('--input_video', help='Video to deblur path')
@click.option('--output_video', help='Deblurred video path')
def deblur_video(weight_path, input_video, output_video):
    g = generator_model()
    g.load_weights(weight_path)
    # Read input video
    cap = cv2.VideoCapture(input_video)
    # Get frame count, possible apriori if reading a file
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames in file are {0}".format(n_frames))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Desired shape into which output is to be reshaped
    desired_shape = (512, 288)
    # output writer object
    out = cv2.VideoWriter(output_video, fourcc, fps, desired_shape)
    # Frame number
    ctr = 1
    # While frame keep coming in
    while cap.isOpened():
        if (ctr - 1) % 25 == 0:
            print("Now processing frame number {0}".format(ctr))
        # Read input frame by frame
        ret, in_frame = cap.read()
        if ret:
            # Shape = (w, h), np array = rows (height), cols (width)
            in_frame1 = in_frame[60:-60, :, :]
            # Out_frame is the 256x256 de-blurred numpy array frame
            image = np.array([preprocess_image(Image.fromarray(in_frame1))])
            generated_images = g.predict(x=image)
            generated = np.array([deprocess_image(img) for img in generated_images])
            image = deprocess_image(image)
            for i in range(generated_images.shape[0]):
                x = image[i, :, :, :]
                img = generated[i, :, :, :]
                output_frame = np.concatenate((x, img), axis=1)
            # Hardcoded [:, 256, :] because shape of output numpy array is (256, 512, 3)
            # out_cv is still a numpy array since cv2 keeps it as numpy after reshape instead of converting to cv2 form
            out_cv = cv2.resize(output_frame[:, 256:, :], desired_shape, interpolation=cv2.INTER_CUBIC)
            # We can write a numpy array directly and it is written with a transformed shape
            out.write(out_cv)
        else:
            break
        ctr += 1
    # Release everything if job is finished
    cap.release()
    out.release()
    return


if __name__ == "__main__":
    deblur_video()
