from nn.layers import DenseLayer, ActivationFunc, LossFunc, MLP

lr = 0.25
true = [[1, -1, 1, -1, 1, -1]]		#(1,6) 
x1 = [[1, -1, 2, -2, 3, -3]]		#(1,6)

nn = MLP(x1, lr, True)

nn.add_layer(DenseLayer(6,3))		#(1,3)
nn.add_layer(ActivationFunc())
nn.add_layer(DenseLayer(3, 4))		#(1,4)
nn.add_layer(ActivationFunc())
nn.add_layer(DenseLayer(4, 5))		#(1,5)
nn.add_layer(ActivationFunc())
nn.add_layer(DenseLayer(5, 6)) 		#(1,6)
nn.add_layer(LossFunc(true))		#(1,1)

nn.epoch(10)
