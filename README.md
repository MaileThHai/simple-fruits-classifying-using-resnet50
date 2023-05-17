# simpleAIclassify
fruits classification
This is a simple AI i make when learning application of AI,this using resnet50 from PyTorch, 
with loss function is MSE, optimizer is SGD, using pre-trained weights V2 so the accurancy is around 96% when training with only 10 epochs.
The train data is only contain 10 type of fresh fruits, around 100 images of each one.


Put img to test folder(it must have 10 folder that named 10 type of fruits inside the test folder, since my code make labels base on them),
run the testmodel after finished training.


![image](https://github.com/MaileThHai/simple-fruits-classifying-using-resnet50/assets/127375951/c5ef86cb-ca2f-44a7-a9df-41e8619c9665)

Can leave the output_test folder empty before test.
Change the fc-layer if need to expand more than 10 type of fruits.
