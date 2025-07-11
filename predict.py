import tensorflow as tf
import numpy as np
from PIL import Image

# 模型目录（必须包含 saved_model.pb 文件）
model_dir = r"E:\mobile\mobile-deeplab-v3-plus-master\result\1748118415"

# 加载模型
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
    graph = tf.get_default_graph()

    # 打印所有操作名，帮助你确认输入输出 tensor 名称
    print("模型中的操作名如下（用于确认输入输出名称）：")
    for op in graph.get_operations():
        print(op.name)

    # 请根据你打印出来的结果手动替换以下 tensor 名
    input_tensor = graph.get_tensor_by_name('Input:0')
    output_tensor = graph.get_tensor_by_name('Output:0')

    # 读取图片
    image_path = r"E:\mobile\mobile-deeplab-v3-plus-master\datasets\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg"
    img = Image.open(image_path).resize((513, 513))
    img = np.array(img)

    # 统一为 RGB 三通道
    if img.ndim == 2:  # 灰度图
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:  # RGBA -> RGB
        img = img[:, :, :3]

    img = img[np.newaxis, ...]  # 添加 batch 维度 -> (1, 513, 513, 3)

    # 推理
    prediction = sess.run(output_tensor, feed_dict={input_tensor: img})
    print("模型输出 prediction.shape =", prediction.shape)

    # squeeze 处理维度
    prediction_image = np.squeeze(prediction).astype(np.uint8)

    # PIL 图像创建
    if prediction_image.ndim == 2:
        result = Image.fromarray(prediction_image)
    elif prediction_image.ndim == 3 and prediction_image.shape[2] == 1:
        result = Image.fromarray(prediction_image[:, :, 0])
    else:
        result = Image.fromarray(prediction_image)

    # 保存图像
    result.save("output.png")
    print("推理完成，结果已保存为 output.png")