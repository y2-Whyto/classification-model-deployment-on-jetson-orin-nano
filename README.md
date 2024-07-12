# 在Jetson Orin NANO上使用TensorRT部署分类模型并验证精度
Deployment and accuracy assessment of classification models on Jetson Orin NANO with TensorRT

## 相关信息 Info

- 编程语言 Programming Language：C++
- 额外库 Extra Libraries：
  - TensorRT (C++)
  - [nlohmann's json](https://github.com/nlohmann/json)

## 一些说明 Simple Explanations

本代码需要读取TensorRT engine文件（.trt），代码本身不包含生成engine的部分，请使用TensorRT的`trtexec`工具进行快速转换，或自行搜索其他使用代码生成engine的脚本。详情请参考TensorRT的官方文档（太长了我也没全看）。

This code needs to read a TensorRT engine file, whose extension is usually ".trt", and the code doesn't generates one, so please use `trtexec` from TensorRT for simple engine generation, or search for some script that generates engine files on your own. For details, see official documents from TensorRT. (To your amusement I didn't read it to the end cuz it's too long ... orz)

代码中包含了必要的英文注释，如有疑问欢迎交流。

The code contains necessary comments in it, for which if you have any question, discussions are welcome.

本代码并未使用对象式的编程，这是不好的。但是目前还有其他工作要做，所以就先不改了=。= 大家可以自行参考或修改。但请注意遵守代码许可。

The code was not written in OOP style, which is not very elegant. However the styling thing is not very urgent for me so... ¯\\\_(ツ)\_/¯ You can copy or modify the code by yourself as you like it. But please pay attention to the licenses.



## 参考 References

本代码参考了下面几篇博客，非常感谢这些乐于分享的前辈们：

References are listed below. Sincere gratitude to these helping seniors:

- https://blog.csdn.net/bobchen1017/article/details/129900569
- https://blog.csdn.net/qq_41263444/article/details/138301510
- https://blog.csdn.net/weixin_38241876/article/details/133177813
- https://blog.csdn.net/qq_26611129/article/details/132738109