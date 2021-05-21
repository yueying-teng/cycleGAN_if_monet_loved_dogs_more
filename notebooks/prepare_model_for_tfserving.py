# %%
import sys
sys.path.append('../')

from cycle_gan import models_resnet_based as models
import tensorflow as tf


# %%
monet_generator = models.Generator(img_height=None, img_width=None, name='m_gen', num_residual_blocks=9) # transforms photos to Monet-esque paintings
# TODO: replace with model_wight_path arg
monet_generator.load_weights('2021_04_19_RESNET_monet_dog_gen_dynamic_input_LSGAN_with_lrdecay_to0.h5')

# %%
## decode -> preprocess -> predict -> postprocess -> encode
@tf.function(
    input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="image")]
)
def serving_fn(input_img):
    def _base64_to_array(img):
#         img = tf.io.decode_base64(img)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (512, 512))
        img = img / 127.5 - 1

        return img

    img = tf.map_fn(_base64_to_array, input_img, dtype=tf.float32)
    predictions = monet_generator(img)
    predictions = predictions* 127.5 + 127.5

    output = tf.cast(predictions[0], dtype=tf.uint8)
    output = tf.io.encode_jpeg(output)
    output = tf.io.encode_base64(output)
    
    return output

# %%
def save(export_path=serving_model_path):
    signatures = {"serving_default": serving_fn}
    tf.saved_model.save(monet_generator, export_dir=export_path, signatures=signatures)
   
# %%
# TODO: replace with tfserving_model_path arg
serving_model_path = '/work/monet_gen_transform_for_serving/1/'
save()

# %%
# serving_model = tf.saved_model.load(serving_model_path)
# print(serving_model.signatures)
# sig_name = list(serving_model.signatures.keys())[0]
# print(sig_name)

