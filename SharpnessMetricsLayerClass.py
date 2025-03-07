import tensorflow as tf

class SharpnessMetricsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SharpnessMetricsLayer, self).__init__(**kwargs)

    def laplacian_variance(self, img):
        img_gray = tf.image.rgb_to_grayscale(img)
        laplacian_filter = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
        laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])  
        laplacian = tf.nn.conv2d(img_gray, laplacian_filter, strides=[1, 1, 1, 1], padding="SAME")
        laplacian_var = tf.math.reduce_variance(laplacian, axis=[1, 2, 3])
        return laplacian_var

    def variance_of_gradient(self, img):
        gx = tf.image.sobel_edges(img)[:, :, :, :, 0]
        gy = tf.image.sobel_edges(img)[:, :, :, :, 1]
        return tf.reduce_mean(gx ** 2 + gy ** 2, axis=[1, 2, 3])

    def tenengrad(self, img):
        sobelx = tf.image.sobel_edges(img)[:, :, :, :, 0]
        sobely = tf.image.sobel_edges(img)[:, :, :, :, 1]
        return tf.reduce_mean(tf.sqrt(sobelx ** 2 + sobely ** 2), axis=[1, 2, 3])

    def ssim_metric(self, img):
        img_gray = tf.image.rgb_to_grayscale(img)
        blurred_img = tf.nn.avg_pool(img_gray, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")
        ssim = tf.image.ssim(img_gray, blurred_img, max_val=1.0)
        return tf.expand_dims(ssim, axis=-1)  

    def psnr_metric(self, img):
        img_gray = tf.image.rgb_to_grayscale(img)
        blurred_img = tf.nn.avg_pool(img_gray, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")
        psnr = tf.image.psnr(img_gray, blurred_img, max_val=1.0)
        return tf.expand_dims(psnr, axis=-1)  

    def call(self, inputs):
        laplacian = tf.expand_dims(self.laplacian_variance(inputs), axis=-1)
        vog = tf.expand_dims(self.variance_of_gradient(inputs), axis=-1)
        tenengrad = tf.expand_dims(self.tenengrad(inputs), axis=-1)
        ssim = self.ssim_metric(inputs)
        psnr = self.psnr_metric(inputs)
        
        metrics = tf.concat([laplacian, vog, tenengrad, ssim, psnr], axis=1) 
        return metrics

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 5)  
