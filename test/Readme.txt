test_1016.py是预测浓度的代码文件，fourier_c.pt是训练好的浓度预测模型；
test_up_1016.py是预测速度压力的代码文件，fourier.pt是训练好的速度压力预测模型。

因为本案例中输入数据的大小不统一，batchsize只能设为1。

我根据案例的最小单元数设定截断模数mode1,mode2,mode3。

需要注意输入和输出数据格式:

输入数据在matlab中格式为(numy,numx,numz)，但读入python中格式为(numy,numx,numz,1)

速度场和压力场输出数据格式为(1,numz,numx,numy,4)

浓度场为(1,numz,numx,numy)

其中numx,numy,numz分别代表x,y,z方向上的单元数

因此后处理时需注意数据格式的转换