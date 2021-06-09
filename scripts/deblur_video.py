import numpy as np
from PIL import Image
import click
import cv2

from deblurgan.model import generator_model
from deblurgan.utils import deprocess_image, preprocess_image

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')


def deblur(weight_path, input_frame):
    g = generator_model()
    g.load_weights(weight_path)
    image = np.array([preprocess_image(Image.fromarray(input_frame))])
    x_test = image
    generated_images = g.predict(x=x_test)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    for i in range(generated_images.shape[0]):
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output_frame = np.concatenate((x, img), axis=1)
    # Hardcoded bacuse shape of output numpy array is (256, 512, 3)
    return output_frame[:, 256:, :]

@click.command()
@click.option('--weight_path', help='Model weight')
@click.option('--input_video', help='Video to deblur path')
@click.option('--output_video', help='Deblurred video path')
def deblur_video(weight_path, input_video, output_video):
    # Read input video
    cap = cv2.VideoCapture(input_video)
    # Get frame count, possible apriori if reading a file
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames in file are {0}".format(n_frames))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # output writer object
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    # Frame number
    ctr = 1
    # While frame keep coming in
    while cap.isOpened():
        if (ctr - 1) % 25 == 0:
            print("Now processing frame number {0}".format(ctr))
        # Read input frame by frame
        ret, in_frame = cap.read()
        if ret:
            out_frame = deblur(weight_path, in_frame)
            out.write(cv2.resize(out_frame, (w, h), interpolation=cv2.INTER_CUBIC))
        else:
            break
        ctr += 1
    # Release everything if job is finished
    cap.release()
    out.release()
    return


if __name__ == "__main__":
    deblur_video()
