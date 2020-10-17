import argparse
import tensorflow as tf
import tqdm
import numpy as np
import cv2
import os
import glob
import facenet
import lfw
class Model():

    def __init__(self):
        import inception_resnet_v1
        self.network = inception_resnet_v1

        self.image_batch = tf.placeholder(tf.uint8, shape=[None, 160, 160, 3], name='images')
        image = (tf.cast(self.image_batch, tf.float32) - 127.5) / 128.0
        prelogits, _ ,t1,_= self.network.inference(image, 1.0, False, bottleneck_layer_size=128)
        self.embeddings = tf.nn.l2_normalize(t1, 1, 1e-10, name='embeddings')
        self.emb = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        pretrained_model = 'models\\facenet\\20170512-110547\\'
        self.sess = tf.Session()
        saver = tf.train.Saver()
        model_exp = os.path.expanduser(pretrained_model)
        print('Model directory: %s' % model_exp)
        _, ckpt_file = facenet.get_model_filenames(model_exp)
        print('Checkpoint file: %s' % ckpt_file)
        saver.restore(self.sess, os.path.join(model_exp, ckpt_file))

    def compute_victim(self, img):

        embeddings = self.eval_embeddings(img)
        embs=self.eval_emb(img)
        self.victim_embeddings = embeddings
        self.victim_emb=embs
        return embeddings,embs
    def structure(self, input_tensor):
        """
        Args:
            input_tensor: NHWC
        """
        rnd = tf.random_uniform((), 135, 160, dtype=tf.int32)
        rescaled = tf.image.resize_images(
            input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        h_rem = 160 - rnd
        w_rem = 160 - rnd
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [
                        pad_left, pad_right], [0, 0]])
        padded.set_shape((input_tensor.shape[0], 160, 160, 3))
        output = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.9),
                         lambda: padded, lambda: input_tensor)
        return output

    def eval_attack(self, img):
        # img: single HWC image
        out = self.sess.run(
            self.adv_image, feed_dict={self.image_batch: [img]})[0]
        return out
    def build_pgd_attack(self, eps,vic):
        # victim_embeddings = tf.constant(self.victim_embeddings, dtype=tf.float32)
        victim_embeddings = tf.constant(vic, dtype=tf.float32)
        # victim_emb = tf.constant(self.victim_emb, dtype=tf.float32)
        def one_step_attack(image, grad):
            orig_image = image
            image = self.structure(image)
            image = (image - 127.5) / 128.0
            image = image + tf.random_uniform(tf.shape(image), minval=-1e-2, maxval=1e-2)
            prelogits, _ ,t1,_= self.network.inference(image, 1.0, False, bottleneck_layer_size=128)         #512 to 128
            embeddings = tf.nn.l2_normalize(t1, 1, 1e-10, name='embeddings')
            # print(embeddings.shape)
            # loss=tf.reduce_mean(tf.abs(victim_embeddings-embeddings))
            loss = -1/2*tf.reduce_mean(tf.square(victim_embeddings - embeddings))
            noise, = tf.gradients(loss, orig_image)
            noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
            noise = 0.1 * grad + noise

            adv = tf.clip_by_value(orig_image + tf.sign(noise) * 1.0, lower_bound, upper_bound)
            return adv, noise

        input = tf.to_float(self.image_batch)
        lower_bound = tf.clip_by_value(input - eps, 0, 255.)
        upper_bound = tf.clip_by_value(input + eps, 0, 255.)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            adv, _ = tf.while_loop(
                lambda _, __: True, one_step_attack,
                (input, tf.zeros_like(input)),
                back_prop=False,
                maximum_iterations=10,
                parallel_iterations=1)
        self.adv_image = adv

        return adv
    def eval_embeddings(self, batch_arr):
        return self.sess.run(self.embeddings, feed_dict={self.image_batch: [batch_arr]})
    def eval_emb(self, batch_arr):
        return self.sess.run(self.emb, feed_dict={self.image_batch: [batch_arr]})
    def distance_to_victim(self, img1,img2):
        emb1 = self.eval_emb(img1)
        emb2=self.eval_emb(img2)

        distance=tf.reduce_sum(tf.square(emb1-emb2))
        return distance




filepath=''
pairs_path="diff.txt"
adv_path='youtube_advdiff'
model=Model()


def generate(eps,path):
    sess = tf.Session()
    pairs=lfw.read_pairs(path)
    print('pairs length is ')
    print(len(pairs))
    D1=0
    D2=0
    n=0
    c=0
    for pair in pairs:

        if not os.path.exists("youtube_advdiff/" + pair[0]):
            os.mkdir("youtube_advdiff/" + pair[0])

        if(len(pair))==4:
            path1=os.path.join(adv_path, pair[0],pair[1])
            path2 = os.path.join(filepath, pair[2], pair[3])
            # savepath=os.path.join('youtube_advdiff',pair[0],pair[1])
            # print(savepath)
            # print(path1)
            img1=cv2.imread(path1)[:, :, ::-1]
            img2=cv2.imread(path2)[:, :, ::-1]
            # victim, v_emb = model.compute_victim(img2)
            # model.build_pgd_attack(eps, victim)
            # out = model.eval_attack(img1)
            # cv2.imwrite(savepath, out[:, :, ::-1])
            d1= model.distance_to_victim(img1, img2)
            # d2= model.distance_to_victim(out, img2)
            # n=n+1
            D1=d1+D1
            # if d1>0.95
            #     c = c + 1
            # n = n + 1
    # print(D1/ n)
    print('n= ' + str(n))
    accuracy = c / n
    print('Accuracy: ' + str(accuracy * 100) + '%')

        # D2=d2+D2
    #         print("行： "+str(n)+"d1: "+str(sess.run(d1))+"d2: "+str(sess.run(d2)))
    #
    # print("完成")
    # print(D1)
    # print(D2)

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

if __name__ == '__main__':
    generate(8,pairs_path)




