import numpy as np
from PIL import Image
import click
import os


from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image


def deblur(weight_path, input_dir, output_dir):
	g = generator_model()
	g.load_weights(weight_path)
	for image_name in os.listdir(input_dir):
	    image = np.array([preprocess_image(load_image(os.path.join(input_dir, image_name)))])
	    image=image[:,:,:, [0, 1, 2]]
	    x_test = image
	    generated_images = g.predict(x=x_test)
	    generated = np.array([deprocess_image(img) for img in generated_images])
	    x_test = deprocess_image(x_test)
	    for i in range(generated_images.shape[0]):
	        x = x_test[i, :, :, :]
	        img = generated[i, :, :, :]
	        output = np.concatenate((x, img), axis=1)
	        im = Image.fromarray(output.astype(np.uint8))
	        im.save(os.path.join(output_dir, image_name))


@click.command()
@click.option('--weight_path', help='Model weight')
@click.option('--input_dir', help='Image to deblur')
@click.option('--output_dir', help='Deblurred image')
def deblur_command(weight_path, input_dir, output_dir):
    return deblur(weight_path, input_dir, output_dir)


if __name__ == "__main__":
    deblur_command()
