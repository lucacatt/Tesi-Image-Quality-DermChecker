import tensorflow as tf

class SharpnessMetricsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SharpnessMetricsLayer, self).__init__(**kwargs)

    @tf.function
    def laplacian_variance(self, img):
        img_gray = tf.image.rgb_to_grayscale(img)
        laplacian_filter = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
        laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])
        laplacian = tf.nn.conv2d(img_gray, laplacian_filter, strides=[1, 1, 1, 1], padding="SAME")
        laplacian_var = tf.math.reduce_variance(laplacian, axis=[1, 2, 3])
        return laplacian_var

    @tf.function
    def calculate_sobel(self,img):
        sobel = tf.image.sobel_edges(img)
        gx = sobel[:, :, :, :, 0]
        gy = sobel[:, :, :, :, 1]
        return gx, gy

    @tf.function
    def variance_of_gradient(self, img):
        gx, gy = self.calculate_sobel(img)
        return tf.reduce_mean(gx ** 2 + gy ** 2, axis=[1, 2, 3])

    @tf.function
    def tenengrad(self, img):
        gx, gy = self.calculate_sobel(img)
        return tf.reduce_mean(tf.sqrt(gx ** 2 + gy ** 2), axis=[1, 2, 3])

    @tf.function
    def ssim_metric(self, img):
        ssims = []
        for i in range(img.shape[-1]):  
          blurred_img = tf.nn.avg_pool(img[..., i:i+1], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")
          ssims.append(tf.image.ssim(img[..., i:i+1], blurred_img, max_val=1.0)) 
        return tf.reduce_mean(ssims, axis=0, keepdims=True) 

    @tf.function
    def psnr_metric(self, img):
        psnrs = []
        for i in range(img.shape[-1]):
            blurred_img = tf.nn.avg_pool(img[..., i:i+1], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")
            psnrs.append(tf.image.psnr(img[..., i:i+1], blurred_img, max_val=1.0)) 
        return tf.reduce_mean(psnrs, axis=0, keepdims=True) 

    def call(self, inputs):
       
        batch_size = tf.shape(inputs)[0]

        laplacian = tf.expand_dims(self.laplacian_variance(inputs), axis=-1)
        vog = tf.expand_dims(self.variance_of_gradient(inputs), axis=-1)
        tenengrad = tf.expand_dims(self.tenengrad(inputs), axis=-1)
        ssim = self.ssim_metric(inputs) 
        psnr = self.psnr_metric(inputs)  

        
        metrics = tf.concat([laplacian, vog, tenengrad, ssim, psnr], axis=-1)

        
        return metrics

    def compute_output_shape(self, input_shape):
      return (None, 5)  
