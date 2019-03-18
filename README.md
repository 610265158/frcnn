# frcnn
A tensorflow faster rcnn

borrow tons of code from tensorpack

now is mainly optimised for face detect

####

it may a littel complicated to use, but i will work on it, :)

####1.prepare data
prepare your data like this:
/images/21_Festival_Festival_21_777.jpg| 497,477,512,493 575,473,593,491 

eg. ~~imgpath| x0,y0,x1,y1 x2,y2,x3,y3

#####2. change the config thing in train_config.py
if training, set config.MODEL.mode=True ,
             config.MODEL.pretrained_model=None
run python train.py


#####3. release a model as .pb file
after training ,set config.MODEL.mode=True ,
                    config.MODEL.pretrained_model='your ckpt model'
run python release.py

but the model is build in ckpt mode, then
run python tools/ckpt.py


put it into ./model

then u can play with it with python vis.py

 
